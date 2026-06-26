"""Line-of-sight geometry utilities for COSMIC-2 BP geolocation.

This module implements the LOS geometry layer required by the paper's
back-propagation geolocation method. It provides:

* Midpoint receiver-to-transmitter LOS direction for a 10-second window.
* Candidate ECEF positions along the LOS for magnetic-field evaluation.
* Minimum LOS tangent-point height estimation for pre-BP filtering.
* Conversion from accepted BP distance to geodetic geolocation.

Scientific convention:
    Positive LOS distance is measured from the LEO receiver toward the GNSS
    transmitter. This resolves the sign ambiguity in the parsed paper text and
    ensures that

        R_candidate = R_rx + L * unit(R_tx - R_rx)

    lies between receiver and transmitter whenever
    ``0 <= L <= |R_tx - R_rx|``.

Units:
    * ECEF positions are meters.
    * Input BP/geolocation distances are kilometers.
    * Tangent height and output altitude are kilometers.
    * Latitude and longitude are degrees.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from src.core.types import SignalWindow, StateVector, Vector3
from src.geometry.coordinates import CoordinateTransformer


_M_PER_KM: float = 1000.0
_KM_PER_M: float = 1.0 / _M_PER_KM
_MIN_LOS_RANGE_M: float = 1.0e-6
_DISTANCE_TOLERANCE_M: float = 1.0e-3
_STATE_TIME_MATCH_TOLERANCE_S: float = 1.0e-9
_TANGENT_OPT_XATOL: float = 1.0e-8
_TANGENT_OPT_MAXITER: int = 80


class LosGeometryError(ValueError):
    """Raised when LOS geometry cannot be computed reliably."""


class LosGeometry:
    """Line-of-sight geometry for COSMIC-2 RO back-propagation processing.

    Args:
        transformer: Coordinate transformer used for ECEF/geodetic conversion.

    The class intentionally does not store application configuration. Distance
    grids and QC thresholds are supplied by upstream/downstream modules.
    """

    def __init__(self, transformer: CoordinateTransformer) -> None:
        """Initialize LOS geometry utilities.

        Args:
            transformer: Coordinate transformer instance.

        Raises:
            TypeError: If ``transformer`` is not a ``CoordinateTransformer``.
        """
        if not isinstance(transformer, CoordinateTransformer):
            raise TypeError(
                "transformer must be a CoordinateTransformer, got "
                f"{type(transformer).__name__}."
            )
        self.transformer: CoordinateTransformer = transformer

    def midpoint_los(self, window: SignalWindow) -> Vector3:
        """Return midpoint receiver-to-transmitter LOS unit vector.

        The returned vector points from the LEO receiver toward the GNSS
        transmitter at ``window.mid_time``:

            u = (R_tx,mid - R_rx,mid) / |R_tx,mid - R_rx,mid|

        This direction is used consistently by ``candidate_position`` and
        ``geolocation_from_distance``.

        Args:
            window: Signal processing window with Tx/Rx state vectors.

        Returns:
            Unit ``Vector3`` in ECEF coordinates.

        Raises:
            TypeError: If ``window`` is not a ``SignalWindow``.
            LosGeometryError: If midpoint states are missing or degenerate.
        """
        rx_mid_m, tx_mid_m = self._midpoint_positions(window)
        los_unit, _ = self._los_unit_and_range(rx_mid_m, tx_mid_m)
        return los_unit

    def candidate_position(self, window: SignalWindow, distance_km: float) -> Vector3:
        """Compute ECEF candidate position along the midpoint LOS.

        The candidate position is measured from the midpoint receiver position
        toward the midpoint transmitter position by ``distance_km``:

            R_cand = R_rx,mid + 1000 * distance_km * u_rx_to_tx

        This method is used by BP-plane construction to select the point where
        IGRF-13 magnetic-field direction is evaluated.

        Args:
            window: Signal processing window with Tx/Rx state vectors.
            distance_km: Distance from receiver toward transmitter in km.

        Returns:
            Candidate ECEF position as ``Vector3`` in meters.

        Raises:
            TypeError: If ``window`` is not a ``SignalWindow``.
            ValueError: If ``distance_km`` is negative or non-finite.
            LosGeometryError: If geometry is missing, degenerate, or the
                candidate would lie beyond the transmitter.
        """
        distance_m: float = self._validate_distance_km(distance_km) * _M_PER_KM
        rx_mid_m, tx_mid_m = self._midpoint_positions(window)
        los_unit, los_range_m = self._los_unit_and_range(rx_mid_m, tx_mid_m)

        self._validate_distance_within_link(distance_m, los_range_m)

        rx_array: NDArray[np.float64] = rx_mid_m.to_array()
        los_array: NDArray[np.float64] = los_unit.to_array()
        candidate_array: NDArray[np.float64] = rx_array + distance_m * los_array
        candidate_position_m: Vector3 = self._array_to_vector(candidate_array)

        self._validate_candidate_direction(
            rx_m=rx_mid_m,
            tx_m=tx_mid_m,
            candidate_m=candidate_position_m,
            distance_m=distance_m,
        )

        return candidate_position_m

    def tangent_height(self, window: SignalWindow) -> float:
        """Estimate minimum LOS tangent-point height over the window.

        For each available Tx/Rx state pair, the method minimizes geodetic
        altitude along the straight segment

            P(s) = R_rx + s * (R_tx - R_rx), 0 <= s <= 1

        and returns the minimum over the full processing window.

        If state vectors are unavailable but ``window.tangent_height_km`` is a
        finite precomputed value, that value is returned. Otherwise the method
        fails explicitly.

        Args:
            window: Signal processing window.

        Returns:
            Minimum tangent-point height in kilometers above the WGS84
            reference ellipsoid.

        Raises:
            TypeError: If ``window`` is not a ``SignalWindow``.
            LosGeometryError: If no tangent height can be computed.
        """
        self._validate_window(window)

        if not window.rx_states or not window.tx_states:
            if math.isfinite(float(window.tangent_height_km)):
                return float(window.tangent_height_km)
            raise LosGeometryError(
                "Cannot compute tangent height: missing rx_states or tx_states "
                "and no finite precomputed window.tangent_height_km is available."
            )

        state_pairs: list[tuple[Vector3, Vector3]] = self._paired_positions(window)
        if not state_pairs:
            if math.isfinite(float(window.tangent_height_km)):
                return float(window.tangent_height_km)
            raise LosGeometryError("Cannot compute tangent height: no valid state pairs.")

        tangent_heights_km: list[float] = []
        for rx_position_m, tx_position_m in state_pairs:
            tangent_heights_km.append(
                self._minimum_altitude_on_segment_km(
                    rx_m=rx_position_m,
                    tx_m=tx_position_m,
                )
            )

        if not tangent_heights_km:
            raise LosGeometryError("Cannot compute tangent height: no valid altitudes.")

        tangent_height_km: float = float(np.nanmin(np.asarray(tangent_heights_km)))
        if not math.isfinite(tangent_height_km):
            raise LosGeometryError("Computed tangent height is non-finite.")

        return tangent_height_km

    def geolocation_from_distance(
        self,
        window: SignalWindow,
        distance_km: float,
    ) -> tuple[float, float, float]:
        """Convert accepted BP distance to geodetic coordinates.

        The same receiver-to-transmitter LOS convention as
        ``candidate_position`` is used. The resulting ECEF point is converted to
        WGS84 geodetic coordinates.

        Args:
            window: Signal processing window with Tx/Rx state vectors.
            distance_km: Accepted geolocation distance from receiver toward
                transmitter in km.

        Returns:
            Tuple ``(latitude_deg, longitude_deg, altitude_km)``.

        Raises:
            TypeError: If ``window`` is not a ``SignalWindow``.
            ValueError: If ``distance_km`` is invalid.
            LosGeometryError: If the geolocation ECEF point cannot be computed.
        """
        geolocation_m: Vector3 = self.candidate_position(window, distance_km)
        latitude_deg, longitude_deg, altitude_km = self.transformer.ecef_to_geodetic(
            geolocation_m
        )
        return float(latitude_deg), float(longitude_deg), float(altitude_km)

    def _midpoint_positions(self, window: SignalWindow) -> tuple[Vector3, Vector3]:
        """Return interpolated/nearest Tx/Rx midpoint ECEF positions.

        Args:
            window: Signal processing window.

        Returns:
            Tuple ``(rx_mid_m, tx_mid_m)``.

        Raises:
            TypeError: If ``window`` is not a ``SignalWindow``.
            LosGeometryError: If state vectors are missing or invalid.
        """
        self._validate_window(window)

        if not window.rx_states:
            raise LosGeometryError("Receiver states are required for LOS geometry.")
        if not window.tx_states:
            raise LosGeometryError("Transmitter states are required for LOS geometry.")

        rx_mid_m: Vector3 = self._position_at_time(
            states=window.rx_states,
            target_time=window.mid_time,
            label="receiver",
        )
        tx_mid_m: Vector3 = self._position_at_time(
            states=window.tx_states,
            target_time=window.mid_time,
            label="transmitter",
        )

        return rx_mid_m, tx_mid_m

    @staticmethod
    def _validate_window(window: SignalWindow) -> None:
        """Validate window type.

        Args:
            window: Object expected to be a ``SignalWindow``.

        Raises:
            TypeError: If ``window`` is not a ``SignalWindow``.
        """
        if not isinstance(window, SignalWindow):
            raise TypeError(
                f"window must be a SignalWindow, got {type(window).__name__}."
            )

    def _position_at_time(
        self,
        states: Sequence[StateVector],
        target_time: datetime,
        label: str,
    ) -> Vector3:
        """Return linearly interpolated or nearest state position at target time.

        If the state list brackets ``target_time``, ECEF coordinates are
        linearly interpolated. If bracketing is impossible, the state closest in
        time is selected.

        Args:
            states: State vectors.
            target_time: Desired timestamp.
            label: Human-readable state label for errors.

        Returns:
            ECEF position as ``Vector3`` in meters.

        Raises:
            LosGeometryError: If states are empty or contain invalid positions.
        """
        if not states:
            raise LosGeometryError(f"No {label} states are available.")

        target_seconds: float = self._datetime_seconds(target_time)
        sorted_states: list[StateVector] = sorted(
            states,
            key=lambda state: self._datetime_seconds(state.time),
        )

        state_times_s: NDArray[np.float64] = np.asarray(
            [self._datetime_seconds(state.time) for state in sorted_states],
            dtype=np.float64,
        )

        for state, state_time_s in zip(sorted_states, state_times_s, strict=True):
            if abs(float(state_time_s) - target_seconds) <= _STATE_TIME_MATCH_TOLERANCE_S:
                return self._validated_position(state.position_m, f"{label} position")

        if len(sorted_states) == 1:
            return self._validated_position(
                sorted_states[0].position_m,
                f"{label} position",
            )

        insertion_index: int = int(np.searchsorted(state_times_s, target_seconds))

        if 0 < insertion_index < len(sorted_states):
            lower_state: StateVector = sorted_states[insertion_index - 1]
            upper_state: StateVector = sorted_states[insertion_index]
            lower_time_s: float = float(state_times_s[insertion_index - 1])
            upper_time_s: float = float(state_times_s[insertion_index])

            time_span_s: float = upper_time_s - lower_time_s
            if time_span_s > _STATE_TIME_MATCH_TOLERANCE_S:
                fraction: float = (target_seconds - lower_time_s) / time_span_s
                return self._interpolate_position(
                    lower_position_m=lower_state.position_m,
                    upper_position_m=upper_state.position_m,
                    fraction=fraction,
                    label=label,
                )

        nearest_index: int = int(np.argmin(np.abs(state_times_s - target_seconds)))
        return self._validated_position(
            sorted_states[nearest_index].position_m,
            f"{label} position",
        )

    @staticmethod
    def _datetime_seconds(value: datetime) -> float:
        """Convert datetime to POSIX-like seconds, treating naive times as UTC.

        Args:
            value: Datetime value.

        Returns:
            Seconds since Unix epoch.

        Raises:
            TypeError: If ``value`` is not a datetime.
            LosGeometryError: If conversion is non-finite.
        """
        if not isinstance(value, datetime):
            raise TypeError(f"State time must be datetime, got {type(value).__name__}.")

        if value.tzinfo is None:
            normalized_time: datetime = value.replace(tzinfo=timezone.utc)
        else:
            normalized_time = value.astimezone(timezone.utc)

        seconds: float = float(normalized_time.timestamp())
        if not math.isfinite(seconds):
            raise LosGeometryError(f"Non-finite datetime timestamp: {value!r}.")
        return seconds

    def _interpolate_position(
        self,
        lower_position_m: Vector3,
        upper_position_m: Vector3,
        fraction: float,
        label: str,
    ) -> Vector3:
        """Linearly interpolate between two ECEF positions.

        Args:
            lower_position_m: Position at lower time.
            upper_position_m: Position at upper time.
            fraction: Interpolation fraction in ``[0, 1]``.
            label: Human-readable label for errors.

        Returns:
            Interpolated ECEF position.

        Raises:
            LosGeometryError: If interpolation inputs are invalid.
        """
        if not math.isfinite(float(fraction)):
            raise LosGeometryError(f"Invalid interpolation fraction for {label}.")

        lower_array: NDArray[np.float64] = self._validated_position(
            lower_position_m,
            f"{label} lower position",
        ).to_array()
        upper_array: NDArray[np.float64] = self._validated_position(
            upper_position_m,
            f"{label} upper position",
        ).to_array()

        interpolated_array: NDArray[np.float64] = (
            lower_array + float(fraction) * (upper_array - lower_array)
        )
        return self._array_to_vector(interpolated_array)

    @staticmethod
    def _validated_position(position_m: Vector3, label: str) -> Vector3:
        """Validate a finite ECEF position vector.

        Args:
            position_m: Position vector.
            label: Label for error messages.

        Returns:
            The original vector if valid.

        Raises:
            TypeError: If ``position_m`` is not ``Vector3``.
            LosGeometryError: If components are non-finite.
        """
        if not isinstance(position_m, Vector3):
            raise TypeError(f"{label} must be Vector3, got {type(position_m).__name__}.")

        array: NDArray[np.float64] = position_m.to_array()
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise LosGeometryError(f"{label} must contain finite ECEF coordinates.")

        return position_m

    @staticmethod
    def _array_to_vector(array: NDArray[np.float64]) -> Vector3:
        """Convert a finite length-3 array to ``Vector3``.

        Args:
            array: Array with shape ``(3,)``.

        Returns:
            Vector3.

        Raises:
            LosGeometryError: If the array is invalid.
        """
        vector_array: NDArray[np.float64] = np.asarray(array, dtype=np.float64)
        if vector_array.shape != (3,) or not np.all(np.isfinite(vector_array)):
            raise LosGeometryError("Expected a finite ECEF array with shape (3,).")
        return Vector3(
            x=float(vector_array[0]),
            y=float(vector_array[1]),
            z=float(vector_array[2]),
        )

    def _los_unit_and_range(self, rx_m: Vector3, tx_m: Vector3) -> tuple[Vector3, float]:
        """Compute receiver-to-transmitter LOS unit vector and range.

        Args:
            rx_m: Receiver ECEF position in meters.
            tx_m: Transmitter ECEF position in meters.

        Returns:
            Tuple ``(los_unit, los_range_m)``.

        Raises:
            LosGeometryError: If Tx/Rx geometry is degenerate.
        """
        rx_position: Vector3 = self._validated_position(rx_m, "receiver position")
        tx_position: Vector3 = self._validated_position(tx_m, "transmitter position")

        rx_array: NDArray[np.float64] = rx_position.to_array()
        tx_array: NDArray[np.float64] = tx_position.to_array()
        delta_array: NDArray[np.float64] = tx_array - rx_array

        los_range_m: float = float(np.linalg.norm(delta_array))
        if not math.isfinite(los_range_m) or los_range_m <= _MIN_LOS_RANGE_M:
            raise LosGeometryError(
                "Degenerate Tx/Rx midpoint geometry: transmitter and receiver "
                "positions are identical or too close."
            )

        los_unit_array: NDArray[np.float64] = delta_array / los_range_m
        los_unit: Vector3 = self._array_to_vector(los_unit_array).unit()
        return los_unit, los_range_m

    @staticmethod
    def _validate_distance_km(distance_km: float) -> float:
        """Validate an input LOS distance in kilometers.

        Args:
            distance_km: Distance in km.

        Returns:
            Validated distance.

        Raises:
            ValueError: If distance is negative or non-finite.
        """
        distance_value_km: float = float(distance_km)
        if not math.isfinite(distance_value_km):
            raise ValueError(f"distance_km must be finite, got {distance_km!r}.")
        if distance_value_km < 0.0:
            raise ValueError(f"distance_km must be non-negative, got {distance_km!r}.")
        return distance_value_km

    @staticmethod
    def _validate_distance_within_link(distance_m: float, los_range_m: float) -> None:
        """Validate that a LOS distance does not exceed Tx/Rx range.

        Args:
            distance_m: Distance from receiver in meters.
            los_range_m: Tx/Rx separation in meters.

        Raises:
            LosGeometryError: If the candidate lies beyond the transmitter.
        """
        if distance_m > los_range_m + _DISTANCE_TOLERANCE_M:
            raise LosGeometryError(
                "Candidate/geolocation distance exceeds midpoint Tx/Rx range: "
                f"distance={distance_m * _KM_PER_M:.3f} km, "
                f"range={los_range_m * _KM_PER_M:.3f} km."
            )

    @staticmethod
    def _validate_candidate_direction(
        rx_m: Vector3,
        tx_m: Vector3,
        candidate_m: Vector3,
        distance_m: float,
    ) -> None:
        """Validate candidate movement from receiver toward transmitter.

        Args:
            rx_m: Receiver ECEF position.
            tx_m: Transmitter ECEF position.
            candidate_m: Candidate ECEF position.
            distance_m: Requested receiver-to-candidate distance in meters.

        Raises:
            LosGeometryError: If the candidate does not lie in the expected LOS
                direction or at the expected distance.
        """
        rx_array: NDArray[np.float64] = rx_m.to_array()
        tx_array: NDArray[np.float64] = tx_m.to_array()
        candidate_array: NDArray[np.float64] = candidate_m.to_array()

        rx_to_tx: NDArray[np.float64] = tx_array - rx_array
        rx_to_candidate: NDArray[np.float64] = candidate_array - rx_array

        candidate_distance_m: float = float(np.linalg.norm(rx_to_candidate))
        if abs(candidate_distance_m - distance_m) > max(
            _DISTANCE_TOLERANCE_M,
            1.0e-10 * max(1.0, distance_m),
        ):
            raise LosGeometryError(
                "Candidate distance from receiver does not match requested "
                "LOS distance."
            )

        if distance_m > _DISTANCE_TOLERANCE_M:
            direction_dot: float = float(np.dot(rx_to_candidate, rx_to_tx))
            if direction_dot <= 0.0:
                raise LosGeometryError(
                    "Candidate position is not in the receiver-to-transmitter "
                    "LOS direction. Check LOS sign convention."
                )

            tx_range_m: float = float(np.linalg.norm(rx_to_tx))
            candidate_to_tx_m: float = float(np.linalg.norm(tx_array - candidate_array))
            if candidate_to_tx_m > tx_range_m + _DISTANCE_TOLERANCE_M:
                raise LosGeometryError(
                    "Candidate moved farther from the transmitter than the "
                    "receiver, indicating an invalid LOS sign convention."
                )

    def _paired_positions(self, window: SignalWindow) -> list[tuple[Vector3, Vector3]]:
        """Build Tx/Rx ECEF state pairs for tangent-height estimation.

        If Rx and Tx state lists have equal length, they are paired by index.
        Otherwise, the denser state timeline is used as reference and the other
        spacecraft position is interpolated or nearest-matched to each reference
        time.

        Args:
            window: Signal processing window.

        Returns:
            List of ``(rx_position_m, tx_position_m)`` pairs.
        """
        rx_states: list[StateVector] = list(window.rx_states)
        tx_states: list[StateVector] = list(window.tx_states)

        if len(rx_states) == len(tx_states):
            return [
                (
                    self._validated_position(rx_state.position_m, "receiver position"),
                    self._validated_position(tx_state.position_m, "transmitter position"),
                )
                for rx_state, tx_state in zip(rx_states, tx_states, strict=True)
            ]

        pairs: list[tuple[Vector3, Vector3]] = []

        if len(rx_states) >= len(tx_states):
            for rx_state in rx_states:
                rx_position_m: Vector3 = self._validated_position(
                    rx_state.position_m,
                    "receiver position",
                )
                tx_position_m: Vector3 = self._position_at_time(
                    states=tx_states,
                    target_time=rx_state.time,
                    label="transmitter",
                )
                pairs.append((rx_position_m, tx_position_m))
        else:
            for tx_state in tx_states:
                tx_position_m = self._validated_position(
                    tx_state.position_m,
                    "transmitter position",
                )
                rx_position_m = self._position_at_time(
                    states=rx_states,
                    target_time=tx_state.time,
                    label="receiver",
                )
                pairs.append((rx_position_m, tx_position_m))

        return pairs

    def _minimum_altitude_on_segment_km(self, rx_m: Vector3, tx_m: Vector3) -> float:
        """Minimize geodetic altitude along a Tx/Rx straight-line segment.

        Args:
            rx_m: Receiver ECEF position in meters.
            tx_m: Transmitter ECEF position in meters.

        Returns:
            Minimum altitude in kilometers.

        Raises:
            LosGeometryError: If altitude evaluation fails.
        """
        rx_position: Vector3 = self._validated_position(rx_m, "receiver position")
        tx_position: Vector3 = self._validated_position(tx_m, "transmitter position")

        rx_array: NDArray[np.float64] = rx_position.to_array()
        tx_array: NDArray[np.float64] = tx_position.to_array()
        delta_array: NDArray[np.float64] = tx_array - rx_array
        segment_length_m: float = float(np.linalg.norm(delta_array))

        if not math.isfinite(segment_length_m):
            raise LosGeometryError("Non-finite Tx/Rx segment length.")

        if segment_length_m <= _MIN_LOS_RANGE_M:
            return self._altitude_km(rx_position)

        candidate_s_values: list[float] = [0.0, 1.0]
        center_closest_s: float = -float(np.dot(rx_array, delta_array)) / float(
            np.dot(delta_array, delta_array)
        )
        if math.isfinite(center_closest_s):
            candidate_s_values.append(float(np.clip(center_closest_s, 0.0, 1.0)))

        candidate_altitudes_km: list[float] = [
            self._altitude_at_segment_fraction_km(rx_array, delta_array, s_value)
            for s_value in candidate_s_values
        ]

        optimization_result = minimize_scalar(
            lambda s_value: self._altitude_at_segment_fraction_km(
                rx_array,
                delta_array,
                float(s_value),
            ),
            bounds=(0.0, 1.0),
            method="bounded",
            options={
                "xatol": _TANGENT_OPT_XATOL,
                "maxiter": _TANGENT_OPT_MAXITER,
            },
        )

        if optimization_result.success and math.isfinite(float(optimization_result.fun)):
            candidate_altitudes_km.append(float(optimization_result.fun))

        minimum_altitude_km: float = float(np.nanmin(np.asarray(candidate_altitudes_km)))
        if not math.isfinite(minimum_altitude_km):
            raise LosGeometryError("Minimum altitude along LOS segment is non-finite.")

        return minimum_altitude_km

    def _altitude_at_segment_fraction_km(
        self,
        rx_array_m: NDArray[np.float64],
        delta_array_m: NDArray[np.float64],
        fraction: float,
    ) -> float:
        """Evaluate geodetic altitude at a segment fraction.

        Args:
            rx_array_m: Receiver ECEF array.
            delta_array_m: Tx minus Rx ECEF array.
            fraction: Segment coordinate in ``[0, 1]``.

        Returns:
            Altitude in kilometers.

        Raises:
            LosGeometryError: If conversion fails or altitude is non-finite.
        """
        bounded_fraction: float = float(np.clip(float(fraction), 0.0, 1.0))
        position_array_m: NDArray[np.float64] = (
            np.asarray(rx_array_m, dtype=np.float64)
            + bounded_fraction * np.asarray(delta_array_m, dtype=np.float64)
        )
        position_m: Vector3 = self._array_to_vector(position_array_m)
        return self._altitude_km(position_m)

    def _altitude_km(self, position_m: Vector3) -> float:
        """Return geodetic altitude of an ECEF position in kilometers.

        Args:
            position_m: ECEF position.

        Returns:
            Altitude in kilometers.

        Raises:
            LosGeometryError: If coordinate conversion fails or altitude is
                non-finite.
        """
        try:
            _, _, altitude_km = self.transformer.ecef_to_geodetic(position_m)
        except Exception as exc:  # pyproj may raise different exception types.
            raise LosGeometryError(
                "Failed to convert ECEF position to geodetic altitude."
            ) from exc

        altitude_value_km: float = float(altitude_km)
        if not math.isfinite(altitude_value_km):
            raise LosGeometryError("Geodetic altitude is non-finite.")

        return altitude_value_km


__all__ = ["LosGeometry", "LosGeometryError"]
