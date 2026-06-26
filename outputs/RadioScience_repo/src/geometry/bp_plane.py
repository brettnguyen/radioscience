"""Back-propagation plane geometry for COSMIC-2 RO geolocation.

This module implements the magnetic-field-defined two-dimensional
back-propagation (BP) plane used by the paper's COSMIC-2 scintillation
geolocation algorithm.

For each 10-second ``SignalWindow`` and candidate magnetic-field distance
``L_mf``:

1. The midpoint receiver-to-transmitter line of sight defines the local
   x-axis.
2. The candidate point at ``L_mf`` along that LOS defines the phase-screen
   origin.
3. IGRF-13 magnetic-field direction at the candidate point is projected onto
   the phase-screen plane perpendicular to the LOS.
4. The projected magnetic-field direction defines the field-aligned
   irregularity direction, y-axis.
5. The BP plane contains x and z, where z = x cross y.
6. The scan angle alpha is estimated from the instantaneous LOS intersections
   with the candidate phase screen over the high-rate window.

The module does not apply final scientific QC thresholds such as
``cos(alpha) > 0.1``. It computes diagnostics used by downstream modules.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.core.types import BpPlaneGeometry, SignalWindow, StateVector, Vector3
from src.geometry.los import LosGeometry
from src.geometry.magnetic_field import MagneticFieldModel


_M_PER_KM: float = 1000.0
_VECTOR_NORM_EPS: float = 1.0e-12
_SCREEN_INTERSECTION_DENOM_EPS_M: float = 1.0e-9
_SCAN_SPEED_EPS_MPS: float = 1.0e-9
_TIME_SPAN_EPS_S: float = 1.0e-9
_ORTHONORMALITY_ATOL: float = 1.0e-9


class BpPlaneGeometryError(ValueError):
    """Raised when BP-plane geometry cannot be constructed reliably."""


class BpPlaneBuilder:
    """Construct magnetic-field-defined BP-plane geometry.

    Args:
        los_geometry: LOS geometry provider. This is the single source of truth
            for receiver-to-transmitter LOS sign convention.
        magnetic_model: IGRF-13 magnetic-field provider. Only field direction is
            used by this class.

    The public interface follows the project design:
        * ``build(window, l_mf_km)``
        * ``project_state(state, plane)``
        * ``compute_scan_angle(window, plane)``
    """

    def __init__(
        self,
        los_geometry: LosGeometry,
        magnetic_model: MagneticFieldModel,
    ) -> None:
        """Initialize the BP-plane builder.

        Args:
            los_geometry: LOS geometry utility.
            magnetic_model: Magnetic-field model wrapper.

        Raises:
            TypeError: If dependencies have invalid types.
        """
        if not isinstance(los_geometry, LosGeometry):
            raise TypeError(
                "los_geometry must be a LosGeometry, got "
                f"{type(los_geometry).__name__}."
            )
        if not isinstance(magnetic_model, MagneticFieldModel):
            raise TypeError(
                "magnetic_model must be a MagneticFieldModel, got "
                f"{type(magnetic_model).__name__}."
            )

        self.los_geometry: LosGeometry = los_geometry
        self.magnetic_model: MagneticFieldModel = magnetic_model

    def build(self, window: SignalWindow, l_mf_km: float) -> BpPlaneGeometry:
        """Build a BP-plane geometry for one magnetic-field candidate distance.

        Args:
            window: High-rate COSMIC-2 signal window with Tx/Rx states.
            l_mf_km: Candidate distance from receiver toward transmitter in km
                where the magnetic field is evaluated.

        Returns:
            Fully specified ``BpPlaneGeometry``.

        Raises:
            TypeError: If ``window`` is not a ``SignalWindow``.
            ValueError: If ``l_mf_km`` is invalid.
            BpPlaneGeometryError: If midpoint LOS or magnetic-field evaluation
                fails in a non-recoverable way.
        """
        self._validate_window(window)
        candidate_distance_km: float = self._validate_distance_km(l_mf_km)

        x_axis: Vector3 = self.los_geometry.midpoint_los(window).unit()
        candidate_position_m: Vector3 = self.los_geometry.candidate_position(
            window=window,
            distance_km=candidate_distance_km,
        )

        magnetic_unit: Vector3 = self.magnetic_model.field_unit(
            time=window.mid_time,
            position_m=candidate_position_m,
        ).unit()

        x_array: NDArray[np.float64] = x_axis.to_array()
        magnetic_array: NDArray[np.float64] = magnetic_unit.to_array()

        magnetic_projection_array: NDArray[np.float64] = (
            magnetic_array - float(np.dot(magnetic_array, x_array)) * x_array
        )
        magnetic_projection_norm: float = self._norm(magnetic_projection_array)

        if magnetic_projection_norm <= _VECTOR_NORM_EPS:
            y_axis, z_axis = self._fallback_axes_perpendicular_to_x(x_axis)
            beta_rad: float = math.pi / 2.0
            alpha_rad: float = math.pi / 2.0
            cos_alpha: float = 0.0

            return BpPlaneGeometry(
                l_mf_km=candidate_distance_km,
                los_unit=x_axis,
                x_axis=x_axis,
                y_axis=y_axis,
                z_axis=z_axis,
                magnetic_unit=magnetic_unit,
                cos_alpha=cos_alpha,
                alpha_rad=alpha_rad,
                beta_rad=beta_rad,
                candidate_position_m=candidate_position_m,
            )

        y_array: NDArray[np.float64] = (
            magnetic_projection_array / magnetic_projection_norm
        )
        y_axis: Vector3 = self._array_to_unit_vector(y_array, "projected magnetic field")

        z_array: NDArray[np.float64] = np.cross(x_axis.to_array(), y_axis.to_array())
        z_axis: Vector3 = self._array_to_unit_vector(z_array, "BP z-axis")

        self._validate_axes(x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)

        cos_beta: float = self._clip_unit_interval(
            abs(float(np.dot(magnetic_unit.to_array(), y_axis.to_array())))
        )
        beta_rad: float = math.acos(cos_beta)

        provisional_plane: BpPlaneGeometry = BpPlaneGeometry(
            l_mf_km=candidate_distance_km,
            los_unit=x_axis,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            magnetic_unit=magnetic_unit,
            cos_alpha=math.nan,
            alpha_rad=math.nan,
            beta_rad=beta_rad,
            candidate_position_m=candidate_position_m,
        )

        scan_components: _ScanAngleComponents = self._compute_scan_angle_components(
            window=window,
            plane=provisional_plane,
        )

        return BpPlaneGeometry(
            l_mf_km=candidate_distance_km,
            los_unit=x_axis,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            magnetic_unit=magnetic_unit,
            cos_alpha=scan_components.cos_alpha,
            alpha_rad=scan_components.alpha_rad,
            beta_rad=beta_rad,
            candidate_position_m=candidate_position_m,
        )

    def project_state(
        self,
        state: StateVector,
        plane: BpPlaneGeometry,
    ) -> tuple[float, float]:
        """Project a state position into the local BP x-z coordinates.

        The origin is ``plane.candidate_position_m``, which corresponds to
        ``plane.l_mf_km``. Downstream modules that need coordinates relative to
        an inner-loop screen distance ``L`` should translate the origin along
        ``plane.los_unit`` by ``(L - plane.l_mf_km)``.

        Args:
            state: Spacecraft/transmitter state vector.
            plane: BP-plane geometry.

        Returns:
            Tuple ``(x_m, z_m)`` in meters.

        Raises:
            TypeError: If inputs have invalid types.
            BpPlaneGeometryError: If vector components are invalid.
        """
        if not isinstance(state, StateVector):
            raise TypeError(
                f"state must be a StateVector, got {type(state).__name__}."
            )
        if not isinstance(plane, BpPlaneGeometry):
            raise TypeError(
                f"plane must be a BpPlaneGeometry, got {type(plane).__name__}."
            )

        position_array: NDArray[np.float64] = self._validated_vector_array(
            state.position_m,
            "state.position_m",
        )
        origin_array: NDArray[np.float64] = self._validated_vector_array(
            plane.candidate_position_m,
            "plane.candidate_position_m",
        )
        x_axis_array: NDArray[np.float64] = self._validated_vector_array(
            plane.x_axis,
            "plane.x_axis",
        )
        z_axis_array: NDArray[np.float64] = self._validated_vector_array(
            plane.z_axis,
            "plane.z_axis",
        )

        displacement_array: NDArray[np.float64] = position_array - origin_array
        x_m: float = float(np.dot(displacement_array, x_axis_array))
        z_m: float = float(np.dot(displacement_array, z_axis_array))

        if not math.isfinite(x_m) or not math.isfinite(z_m):
            raise BpPlaneGeometryError("Projected state coordinates are non-finite.")

        return x_m, z_m

    def compute_scan_angle(
        self,
        window: SignalWindow,
        plane: BpPlaneGeometry,
    ) -> float:
        """Compute scan angle alpha for a window and BP plane.

        The scan angle is the angle between the LOS scan trajectory on the
        candidate phase screen and the BP-plane transverse z-axis. This method
        returns ``alpha_rad``. The associated ``cos(alpha)`` is stored in the
        ``BpPlaneGeometry`` returned by ``build()`` and is computed internally by
        the same algorithm.

        Args:
            window: High-rate COSMIC-2 signal window.
            plane: BP-plane geometry.

        Returns:
            Scan angle in radians. If the scan direction cannot be estimated
            reliably, returns ``pi/2`` so that ``cos(alpha)`` is effectively
            zero for downstream QC.

        Raises:
            TypeError: If inputs have invalid types.
        """
        return self._compute_scan_angle_components(window=window, plane=plane).alpha_rad

    def _compute_scan_angle_components(
        self,
        window: SignalWindow,
        plane: BpPlaneGeometry,
    ) -> "_ScanAngleComponents":
        """Compute both alpha and cos(alpha) from screen-intersection motion."""
        self._validate_window(window)
        if not isinstance(plane, BpPlaneGeometry):
            raise TypeError(
                f"plane must be a BpPlaneGeometry, got {type(plane).__name__}."
            )

        intersections: _ScreenIntersections = self._screen_intersections(
            window=window,
            plane=plane,
        )

        if intersections.time_s.size < 2:
            return _ScanAngleComponents(alpha_rad=math.pi / 2.0, cos_alpha=0.0)

        time_centered_s: NDArray[np.float64] = (
            intersections.time_s - float(np.mean(intersections.time_s))
        )
        if float(np.ptp(time_centered_s)) <= _TIME_SPAN_EPS_S:
            return _ScanAngleComponents(alpha_rad=math.pi / 2.0, cos_alpha=0.0)

        try:
            y_slope_mps: float = self._linear_slope(
                time_s=time_centered_s,
                value_m=intersections.y_m,
            )
            z_slope_mps: float = self._linear_slope(
                time_s=time_centered_s,
                value_m=intersections.z_m,
            )
        except BpPlaneGeometryError:
            return _ScanAngleComponents(alpha_rad=math.pi / 2.0, cos_alpha=0.0)

        scan_speed_mps: float = math.hypot(y_slope_mps, z_slope_mps)
        if (
            not math.isfinite(scan_speed_mps)
            or scan_speed_mps <= _SCAN_SPEED_EPS_MPS
        ):
            return _ScanAngleComponents(alpha_rad=math.pi / 2.0, cos_alpha=0.0)

        cos_alpha: float = self._clip_unit_interval(abs(z_slope_mps) / scan_speed_mps)
        alpha_rad: float = math.acos(cos_alpha)

        if not math.isfinite(alpha_rad):
            return _ScanAngleComponents(alpha_rad=math.pi / 2.0, cos_alpha=0.0)

        return _ScanAngleComponents(alpha_rad=alpha_rad, cos_alpha=cos_alpha)

    def _screen_intersections(
        self,
        window: SignalWindow,
        plane: BpPlaneGeometry,
    ) -> "_ScreenIntersections":
        """Compute instantaneous LOS intersections with a candidate screen.

        The phase screen is defined by:

            (r - O) dot x_axis = 0

        where ``O = plane.candidate_position_m``.
        """
        if not window.rx_states or not window.tx_states:
            return _ScreenIntersections(
                time_s=np.empty(0, dtype=np.float64),
                y_m=np.empty(0, dtype=np.float64),
                z_m=np.empty(0, dtype=np.float64),
            )

        origin_array: NDArray[np.float64] = self._validated_vector_array(
            plane.candidate_position_m,
            "plane.candidate_position_m",
        )
        x_axis_array: NDArray[np.float64] = self._validated_unit_axis(
            plane.x_axis,
            "plane.x_axis",
        )
        y_axis_array: NDArray[np.float64] = self._validated_unit_axis(
            plane.y_axis,
            "plane.y_axis",
        )
        z_axis_array: NDArray[np.float64] = self._validated_unit_axis(
            plane.z_axis,
            "plane.z_axis",
        )

        state_pairs: list[tuple[StateVector, StateVector]] = self._paired_states(window)
        if not state_pairs:
            return _ScreenIntersections(
                time_s=np.empty(0, dtype=np.float64),
                y_m=np.empty(0, dtype=np.float64),
                z_m=np.empty(0, dtype=np.float64),
            )

        time_values_s: list[float] = []
        y_values_m: list[float] = []
        z_values_m: list[float] = []

        for sample_index, (rx_state, tx_state) in enumerate(state_pairs):
            rx_array: NDArray[np.float64] = self._validated_vector_array(
                rx_state.position_m,
                "receiver state position",
            )
            tx_array: NDArray[np.float64] = self._validated_vector_array(
                tx_state.position_m,
                "transmitter state position",
            )

            los_delta_array: NDArray[np.float64] = tx_array - rx_array
            denominator_m: float = float(np.dot(los_delta_array, x_axis_array))

            if (
                not math.isfinite(denominator_m)
                or abs(denominator_m) <= _SCREEN_INTERSECTION_DENOM_EPS_M
            ):
                continue

            numerator_m: float = -float(np.dot(rx_array - origin_array, x_axis_array))
            u_parameter: float = numerator_m / denominator_m

            if not math.isfinite(u_parameter):
                continue

            screen_point_array: NDArray[np.float64] = (
                rx_array + u_parameter * los_delta_array
            )
            displacement_array: NDArray[np.float64] = screen_point_array - origin_array

            y_m: float = float(np.dot(displacement_array, y_axis_array))
            z_m: float = float(np.dot(displacement_array, z_axis_array))
            if not math.isfinite(y_m) or not math.isfinite(z_m):
                continue

            time_s: float = self._sample_time_seconds(
                window=window,
                state=rx_state,
                sample_index=sample_index,
            )
            if not math.isfinite(time_s):
                continue

            time_values_s.append(time_s)
            y_values_m.append(y_m)
            z_values_m.append(z_m)

        return _ScreenIntersections(
            time_s=np.asarray(time_values_s, dtype=np.float64),
            y_m=np.asarray(y_values_m, dtype=np.float64),
            z_m=np.asarray(z_values_m, dtype=np.float64),
        )

    @staticmethod
    def _paired_states(window: SignalWindow) -> list[tuple[StateVector, StateVector]]:
        """Return Tx/Rx state pairs by index.

        ``SignalWindow`` enforces state-list length consistency when lists are
        provided. This helper still uses the minimum length defensively.
        """
        pair_count: int = min(len(window.rx_states), len(window.tx_states))
        if pair_count <= 0:
            return []
        return [
            (window.rx_states[index], window.tx_states[index])
            for index in range(pair_count)
        ]

    @staticmethod
    def _sample_time_seconds(
        window: SignalWindow,
        state: StateVector,
        sample_index: int,
    ) -> float:
        """Return sample time in seconds relative to ``window.mid_time``."""
        if isinstance(state.time, datetime) and isinstance(window.mid_time, datetime):
            state_time: datetime = state.time
            mid_time: datetime = window.mid_time

            if state_time.tzinfo is None:
                state_time = state_time.replace(tzinfo=timezone.utc)
            else:
                state_time = state_time.astimezone(timezone.utc)

            if mid_time.tzinfo is None:
                mid_time = mid_time.replace(tzinfo=timezone.utc)
            else:
                mid_time = mid_time.astimezone(timezone.utc)

            return float((state_time - mid_time).total_seconds())

        if 0 <= sample_index < len(window.times):
            time_values: NDArray[np.float64] = np.asarray(window.times, dtype=np.float64)
            return float(time_values[sample_index] - np.mean(time_values))

        return float(sample_index)

    @staticmethod
    def _linear_slope(
        time_s: NDArray[np.float64],
        value_m: NDArray[np.float64],
    ) -> float:
        """Estimate linear slope by least squares."""
        time_array: NDArray[np.float64] = np.asarray(time_s, dtype=np.float64)
        value_array: NDArray[np.float64] = np.asarray(value_m, dtype=np.float64)

        if time_array.ndim != 1 or value_array.ndim != 1:
            raise BpPlaneGeometryError("Linear fit inputs must be one-dimensional.")
        if time_array.size != value_array.size:
            raise BpPlaneGeometryError("Linear fit inputs must have equal lengths.")
        if time_array.size < 2:
            raise BpPlaneGeometryError("At least two samples are required for slope.")
        if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(value_array)):
            raise BpPlaneGeometryError("Linear fit inputs must be finite.")

        design_matrix: NDArray[np.float64] = np.column_stack(
            [time_array, np.ones_like(time_array)]
        )
        try:
            coefficients, _, _, _ = np.linalg.lstsq(
                design_matrix,
                value_array,
                rcond=None,
            )
        except np.linalg.LinAlgError as exc:
            raise BpPlaneGeometryError("Linear scan-direction fit failed.") from exc

        slope: float = float(coefficients[0])
        if not math.isfinite(slope):
            raise BpPlaneGeometryError("Linear scan-direction slope is non-finite.")
        return slope

    @staticmethod
    def _fallback_axes_perpendicular_to_x(x_axis: Vector3) -> tuple[Vector3, Vector3]:
        """Construct deterministic axes when magnetic projection is degenerate."""
        x_array: NDArray[np.float64] = x_axis.unit().to_array()

        reference_candidates: tuple[NDArray[np.float64], ...] = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )

        best_reference: NDArray[np.float64] = reference_candidates[0]
        smallest_alignment: float = abs(float(np.dot(x_array, best_reference)))

        for reference in reference_candidates[1:]:
            alignment: float = abs(float(np.dot(x_array, reference)))
            if alignment < smallest_alignment:
                best_reference = reference
                smallest_alignment = alignment

        y_raw: NDArray[np.float64] = (
            best_reference - float(np.dot(best_reference, x_array)) * x_array
        )
        y_norm: float = float(np.linalg.norm(y_raw))
        if y_norm <= _VECTOR_NORM_EPS:
            raise BpPlaneGeometryError(
                "Could not construct fallback y-axis perpendicular to LOS."
            )

        y_array: NDArray[np.float64] = y_raw / y_norm
        z_array: NDArray[np.float64] = np.cross(x_array, y_array)

        y_axis: Vector3 = BpPlaneBuilder._array_to_unit_vector_static(y_array)
        z_axis: Vector3 = BpPlaneBuilder._array_to_unit_vector_static(z_array)
        return y_axis, z_axis

    @staticmethod
    def _array_to_unit_vector_static(array: NDArray[np.float64]) -> Vector3:
        """Convert array to a unit Vector3 without an instance."""
        vector_array: NDArray[np.float64] = np.asarray(array, dtype=np.float64)
        norm: float = float(np.linalg.norm(vector_array))
        if vector_array.shape != (3,) or not np.all(np.isfinite(vector_array)):
            raise BpPlaneGeometryError("Expected finite array with shape (3,).")
        if norm <= _VECTOR_NORM_EPS:
            raise BpPlaneGeometryError("Cannot normalize near-zero vector.")
        unit_array: NDArray[np.float64] = vector_array / norm
        return Vector3(
            x=float(unit_array[0]),
            y=float(unit_array[1]),
            z=float(unit_array[2]),
        )

    def _array_to_unit_vector(
        self,
        array: NDArray[np.float64],
        label: str,
    ) -> Vector3:
        """Convert a finite length-3 array to a unit ``Vector3``."""
        try:
            return self._array_to_unit_vector_static(array)
        except BpPlaneGeometryError as exc:
            raise BpPlaneGeometryError(f"Invalid {label}: {exc}") from exc

    @staticmethod
    def _validated_vector_array(vector: Vector3, label: str) -> NDArray[np.float64]:
        """Return a finite length-3 array from a ``Vector3``."""
        if not isinstance(vector, Vector3):
            raise TypeError(f"{label} must be Vector3, got {type(vector).__name__}.")
        array: NDArray[np.float64] = vector.to_array()
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise BpPlaneGeometryError(f"{label} must contain finite components.")
        return array

    @classmethod
    def _validated_unit_axis(cls, vector: Vector3, label: str) -> NDArray[np.float64]:
        """Return a validated unit-axis array."""
        array: NDArray[np.float64] = cls._validated_vector_array(vector, label)
        norm: float = float(np.linalg.norm(array))
        if not math.isfinite(norm) or norm <= _VECTOR_NORM_EPS:
            raise BpPlaneGeometryError(f"{label} is zero or non-finite.")
        return array / norm

    @staticmethod
    def _validate_axes(
        x_axis: Vector3,
        y_axis: Vector3,
        z_axis: Vector3,
    ) -> None:
        """Validate approximate orthonormality of BP axes."""
        x_array: NDArray[np.float64] = x_axis.to_array()
        y_array: NDArray[np.float64] = y_axis.to_array()
        z_array: NDArray[np.float64] = z_axis.to_array()

        for axis_name, axis_array in {
            "x_axis": x_array,
            "y_axis": y_array,
            "z_axis": z_array,
        }.items():
            norm: float = float(np.linalg.norm(axis_array))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_ORTHONORMALITY_ATOL):
                raise BpPlaneGeometryError(f"{axis_name} is not unit length.")

        dot_xy: float = abs(float(np.dot(x_array, y_array)))
        dot_xz: float = abs(float(np.dot(x_array, z_array)))
        dot_yz: float = abs(float(np.dot(y_array, z_array)))

        if dot_xy > _ORTHONORMALITY_ATOL:
            raise BpPlaneGeometryError("x_axis and y_axis are not orthogonal.")
        if dot_xz > _ORTHONORMALITY_ATOL:
            raise BpPlaneGeometryError("x_axis and z_axis are not orthogonal.")
        if dot_yz > _ORTHONORMALITY_ATOL:
            raise BpPlaneGeometryError("y_axis and z_axis are not orthogonal.")

    @staticmethod
    def _validate_window(window: SignalWindow) -> None:
        """Validate ``SignalWindow`` type."""
        if not isinstance(window, SignalWindow):
            raise TypeError(
                f"window must be a SignalWindow, got {type(window).__name__}."
            )

    @staticmethod
    def _validate_distance_km(distance_km: float) -> float:
        """Validate candidate distance in kilometers."""
        distance_value_km: float = float(distance_km)
        if not math.isfinite(distance_value_km):
            raise ValueError(f"l_mf_km must be finite, got {distance_km!r}.")
        if distance_value_km < 0.0:
            raise ValueError(f"l_mf_km must be non-negative, got {distance_km!r}.")
        return distance_value_km

    @staticmethod
    def _norm(array: NDArray[np.float64]) -> float:
        """Return Euclidean norm for a finite vector array."""
        vector_array: NDArray[np.float64] = np.asarray(array, dtype=np.float64)
        if vector_array.shape != (3,) or not np.all(np.isfinite(vector_array)):
            raise BpPlaneGeometryError("Vector array must be finite with shape (3,).")
        return float(np.linalg.norm(vector_array))

    @staticmethod
    def _clip_unit_interval(value: float) -> float:
        """Clip a scalar to the closed interval [0, 1]."""
        scalar_value: float = float(value)
        if not math.isfinite(scalar_value):
            return 0.0
        return float(np.clip(scalar_value, 0.0, 1.0))


class _ScanAngleComponents:
    """Internal container for alpha and cos(alpha)."""

    def __init__(self, alpha_rad: float, cos_alpha: float) -> None:
        """Initialize scan-angle components."""
        self.alpha_rad: float = float(alpha_rad)
        self.cos_alpha: float = float(cos_alpha)


class _ScreenIntersections:
    """Internal container for phase-screen intersection coordinates."""

    def __init__(
        self,
        time_s: NDArray[np.float64],
        y_m: NDArray[np.float64],
        z_m: NDArray[np.float64],
    ) -> None:
        """Initialize screen-intersection arrays."""
        self.time_s: NDArray[np.float64] = np.asarray(time_s, dtype=np.float64)
        self.y_m: NDArray[np.float64] = np.asarray(y_m, dtype=np.float64)
        self.z_m: NDArray[np.float64] = np.asarray(z_m, dtype=np.float64)

        if self.time_s.ndim != 1 or self.y_m.ndim != 1 or self.z_m.ndim != 1:
            raise BpPlaneGeometryError(
                "Screen-intersection arrays must be one-dimensional."
            )
        if not (
            self.time_s.size == self.y_m.size == self.z_m.size
        ):
            raise BpPlaneGeometryError(
                "Screen-intersection arrays must have equal lengths."
            )


__all__ = ["BpPlaneBuilder", "BpPlaneGeometryError"]
