"""Quality-control utilities for COSMIC-2 BP geolocation.

This module implements the paper-faithful quality-control and curve-analysis
logic for the COSMIC-2 back-propagation geolocation method.

Implemented responsibilities:
    * Pre-BP filtering by phase scintillation and tangent-point height.
    * Sliding polynomial smoothing of V(L).
    * Global local-minimum detection for smoothed V(L).
    * Q metric computation from monotonic V(L) rises around L0.
    * D(L_mf) zero-crossing detection with two-sample persistence.
    * Final geolocation QC by single-valued status, Q, and cos(alpha).

This module deliberately does not perform propagation, geometry construction,
magnetic-field evaluation, stationary-transmitter correction, geodetic
conversion, plotting, or file I/O.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config import AppConfig
from src.core import constants
from src.core.types import DCurve, GeolocationResult, ScintillationMetrics


_NUMERICAL_EPS: float = float(np.finfo(np.float64).eps)
_MIN_ARRAY_LENGTH_FOR_LOCAL_MINIMUM: int = 3


class QualityController:
    """Quality-control and curve-analysis logic for BP geolocation.

    Args:
        config: Validated application configuration. Thresholds and method
            settings are derived from ``config.yaml`` through ``AppConfig``.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize QC thresholds and method settings.

        Args:
            config: Application configuration.

        Raises:
            TypeError: If ``config`` is not an ``AppConfig``.
            ValueError: If a required threshold or grid setting is invalid.
        """
        if not isinstance(config, AppConfig):
            raise TypeError(
                f"config must be an AppConfig, got {type(config).__name__}."
            )

        self.config: AppConfig = config

        self.sigma_phi_threshold_rad: float = self._positive_float(
            getattr(config, "sigma_phi_threshold_rad"),
            "sigma_phi_threshold_rad",
        )
        self.tangent_height_min_km: float = self._nonnegative_float(
            getattr(config, "tangent_height_min_km"),
            "tangent_height_min_km",
        )
        self.smoothing_window_km: float = self._positive_float(
            getattr(config, "smoothing_window_km"),
            "smoothing_window_km",
        )
        self.smoothing_poly_order: int = self._nonnegative_int(
            getattr(config, "smoothing_poly_order"),
            "smoothing_poly_order",
        )
        self.bp_step_km: float = self._positive_float(
            getattr(config, "bp_step_km"),
            "bp_step_km",
        )
        self.q_threshold: float = self._positive_float(
            getattr(config, "q_threshold"),
            "q_threshold",
        )
        self.cos_alpha_threshold: float = self._bounded_float(
            getattr(config, "cos_alpha_threshold"),
            "cos_alpha_threshold",
            lower=0.0,
            upper=1.0,
        )

        self.min_l1_samples: int = self._nonnegative_int(
            getattr(config, "min_l1_samples", constants.DEFAULT_MIN_L1_SAMPLES),
            "min_l1_samples",
        )
        self.min_l2_samples: int = self._nonnegative_int(
            getattr(config, "min_l2_samples", constants.DEFAULT_MIN_L2_SAMPLES),
            "min_l2_samples",
        )
        self.sample_spacing_km: float = self._positive_float(
            getattr(
                config,
                "minimum_detection_sample_spacing_km",
                getattr(
                    config,
                    "sample_spacing_km",
                    getattr(config, "bp_step_km", constants.DEFAULT_BP_STEP_KM),
                ),
            ),
            "sample_spacing_km",
        )

        self.zero_crossing_persistence_samples: int = self._nonnegative_int(
            getattr(
                config,
                "zero_crossing_persistence_samples",
                constants.DEFAULT_ZERO_CROSSING_PERSISTENCE_SAMPLES,
            ),
            "zero_crossing_persistence_samples",
        )

        self.require_single_valued_geolocation: bool = bool(
            getattr(
                config,
                "require_single_valued_geolocation",
                constants.DEFAULT_REQUIRE_SINGLE_VALUED,
            )
        )
        self.discard_multivalued_geolocations: bool = bool(
            getattr(
                config,
                "discard_multivalued_geolocations",
                constants.DEFAULT_DISCARD_MULTIVALUED,
            )
        )

    def passes_pre_bp(
        self,
        metrics: ScintillationMetrics,
        tangent_height_km: float,
    ) -> tuple[bool, str]:
        """Apply paper pre-BP QC: sigma_phi and tangent height.

        Criteria are strict:
            sigma_phi_rad > config.sigma_phi_threshold_rad
            tangent_height_km > config.tangent_height_min_km

        Args:
            metrics: Scintillation metrics for a 10-second window.
            tangent_height_km: LOS tangent-point height in kilometers.

        Returns:
            Tuple ``(passed, rejection_reason)``.
        """
        if not isinstance(metrics, ScintillationMetrics):
            return False, "invalid_pre_bp_metrics"

        sigma_phi_rad: float = float(metrics.sigma_phi_rad)
        tangent_height_value_km: float = float(tangent_height_km)

        if not math.isfinite(sigma_phi_rad) or not math.isfinite(
            tangent_height_value_km
        ):
            return False, "invalid_pre_bp_metrics"

        if sigma_phi_rad <= self.sigma_phi_threshold_rad:
            return False, "sigma_phi_below_threshold"

        if tangent_height_value_km <= self.tangent_height_min_km:
            return False, "tangent_height_below_threshold"

        return True, "passed"

    def smooth_v_curve(
        self,
        distances_km: NDArray[Any],
        v_raw: NDArray[Any],
    ) -> NDArray[np.float64]:
        """Smooth V(L) using sliding polynomial regression.

        For each distance sample, neighboring finite samples within the
        configured smoothing window are fit by a polynomial of configured order,
        then evaluated at the center distance.

        Args:
            distances_km: BP distance grid in kilometers.
            v_raw: Raw V(L) values.

        Returns:
            Smoothed V(L) array with the same shape as ``v_raw``.

        Raises:
            ValueError: If arrays are not one-dimensional or have unequal length.
        """
        distance_array_km: NDArray[np.float64] = self._as_1d_float_array(
            distances_km,
            "distances_km",
        )
        v_array: NDArray[np.float64] = self._as_1d_float_array(v_raw, "v_raw")
        self._require_same_length(
            distance_array_km,
            "distances_km",
            v_array,
            "v_raw",
        )

        if distance_array_km.size == 0:
            return np.empty(0, dtype=np.float64)

        if not np.all(np.isfinite(distance_array_km)):
            raise ValueError("distances_km must contain only finite values.")

        v_smooth: NDArray[np.float64] = np.full_like(v_array, np.nan, dtype=np.float64)
        half_window_km: float = 0.5 * self.smoothing_window_km

        for index, center_distance_km in enumerate(distance_array_km):
            finite_window_mask: NDArray[np.bool_] = (
                np.isfinite(v_array)
                & (np.abs(distance_array_km - center_distance_km) <= half_window_km)
            )

            window_distances_km: NDArray[np.float64] = distance_array_km[
                finite_window_mask
            ]
            window_values: NDArray[np.float64] = v_array[finite_window_mask]

            if window_values.size < self.smoothing_poly_order + 1:
                v_smooth[index] = v_array[index]
                continue

            unique_distance_count: int = int(np.unique(window_distances_km).size)
            if unique_distance_count < self.smoothing_poly_order + 1:
                v_smooth[index] = v_array[index]
                continue

            centered_distances_km: NDArray[np.float64] = (
                window_distances_km - center_distance_km
            )

            try:
                coefficients: NDArray[np.float64] = np.polyfit(
                    centered_distances_km,
                    window_values,
                    deg=self.smoothing_poly_order,
                )
                smoothed_value: float = float(np.polyval(coefficients, 0.0))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                smoothed_value = float(v_array[index])

            v_smooth[index] = smoothed_value

        return v_smooth

    def find_global_local_minimum(
        self,
        distances_km: NDArray[Any],
        v_smooth: NDArray[Any],
    ) -> tuple[float, float, bool]:
        """Find the global local minimum of a smoothed V(L) curve.

        Endpoints are never accepted as valid local minima.

        Args:
            distances_km: BP distance grid in kilometers.
            v_smooth: Smoothed V(L) values.

        Returns:
            Tuple ``(l0_km, v0, has_valid_minimum)``.
        """
        distance_array_km: NDArray[np.float64] = self._as_1d_float_array(
            distances_km,
            "distances_km",
        )
        v_array: NDArray[np.float64] = self._as_1d_float_array(
            v_smooth,
            "v_smooth",
        )
        self._require_same_length(
            distance_array_km,
            "distances_km",
            v_array,
            "v_smooth",
        )

        if distance_array_km.size < _MIN_ARRAY_LENGTH_FOR_LOCAL_MINIMUM:
            return math.nan, math.nan, False

        if not np.all(np.isfinite(distance_array_km)):
            return math.nan, math.nan, False

        candidate_indices: list[int] = []
        for index in range(1, v_array.size - 1):
            left_value: float = float(v_array[index - 1])
            center_value: float = float(v_array[index])
            right_value: float = float(v_array[index + 1])

            if not (
                math.isfinite(left_value)
                and math.isfinite(center_value)
                and math.isfinite(right_value)
            ):
                continue

            if center_value <= left_value and center_value <= right_value:
                candidate_indices.append(index)

        if not candidate_indices:
            return math.nan, math.nan, False

        candidate_array: NDArray[np.int64] = np.asarray(candidate_indices, dtype=np.int64)
        candidate_values: NDArray[np.float64] = v_array[candidate_array]
        finite_candidate_mask: NDArray[np.bool_] = np.isfinite(candidate_values)

        if not np.any(finite_candidate_mask):
            return math.nan, math.nan, False

        finite_candidate_array: NDArray[np.int64] = candidate_array[finite_candidate_mask]
        finite_candidate_values: NDArray[np.float64] = candidate_values[
            finite_candidate_mask
        ]

        minimum_value: float = float(np.min(finite_candidate_values))
        value_tolerance: float = self._numeric_tolerance(minimum_value)

        global_min_indices: NDArray[np.int64] = finite_candidate_array[
            np.abs(finite_candidate_values - minimum_value) <= value_tolerance
        ]

        selected_index: int = self._select_representative_minimum_index(
            global_min_indices,
            v_array.size,
        )

        l0_km: float = float(distance_array_km[selected_index])
        v0: float = float(v_array[selected_index])

        if not math.isfinite(l0_km) or not math.isfinite(v0):
            return math.nan, math.nan, False

        return l0_km, v0, True

    def compute_q(
        self,
        distances_km: NDArray[Any],
        v_smooth: NDArray[Any],
        l0_km: float,
    ) -> tuple[float, float, float, float, float]:
        """Compute L1, L2, V1, V2, and Q for a V(L) minimum.

        Args:
            distances_km: BP distance grid in kilometers.
            v_smooth: Smoothed V(L) values.
            l0_km: Distance of the local minimum.

        Returns:
            Tuple ``(l1_km, l2_km, v1, v2, q)``. If the Q metric is invalid
            under paper QC criteria, ``q`` is returned as NaN while diagnostic
            span/value outputs are preserved when available.
        """
        distance_array_km: NDArray[np.float64] = self._as_1d_float_array(
            distances_km,
            "distances_km",
        )
        v_array: NDArray[np.float64] = self._as_1d_float_array(
            v_smooth,
            "v_smooth",
        )
        self._require_same_length(
            distance_array_km,
            "distances_km",
            v_array,
            "v_smooth",
        )

        if distance_array_km.size < _MIN_ARRAY_LENGTH_FOR_LOCAL_MINIMUM:
            return math.nan, math.nan, math.nan, math.nan, math.nan
        if not math.isfinite(float(l0_km)):
            return math.nan, math.nan, math.nan, math.nan, math.nan
        if not np.all(np.isfinite(distance_array_km)):
            return math.nan, math.nan, math.nan, math.nan, math.nan

        i0: int | None = self._locate_distance_index(distance_array_km, float(l0_km))
        if i0 is None:
            return math.nan, math.nan, math.nan, math.nan, math.nan
        if i0 <= 0 or i0 >= distance_array_km.size - 1:
            return math.nan, math.nan, math.nan, math.nan, math.nan

        v0: float = float(v_array[i0])
        if not math.isfinite(v0):
            return math.nan, math.nan, math.nan, math.nan, math.nan

        left_index: int = self._monotonic_endpoint_index(
            v_array=v_array,
            start_index=i0,
            direction=-1,
        )
        right_index: int = self._monotonic_endpoint_index(
            v_array=v_array,
            start_index=i0,
            direction=1,
        )

        l1_km: float = abs(float(distance_array_km[i0] - distance_array_km[left_index]))
        l2_km: float = abs(float(distance_array_km[right_index] - distance_array_km[i0]))
        v1: float = float(v_array[left_index])
        v2: float = float(v_array[right_index])

        q: float = math.nan
        minimum_l1_km: float = self.min_l1_samples * self.sample_spacing_km
        minimum_l2_km: float = self.min_l2_samples * self.sample_spacing_km

        if (
            math.isfinite(l1_km)
            and math.isfinite(l2_km)
            and math.isfinite(v1)
            and math.isfinite(v2)
            and math.isfinite(v0)
            and v0 > 0.0
            and l1_km >= minimum_l1_km
            and l2_km >= minimum_l2_km
        ):
            q_value: float = min(v1, v2) / v0
            if math.isfinite(q_value):
                q = float(q_value)

        return float(l1_km), float(l2_km), v1, v2, q

    def find_zero_crossings(self, d_curve: DCurve) -> list[float]:
        """Find valid D(L_mf)=0 geolocation crossings.

        The paper's persistence criterion is enforced: each side of a crossing
        must contain at least the configured number of contiguous finite samples
        with opposite signs.

        Args:
            d_curve: D-curve diagnostic object.

        Returns:
            List of valid zero-crossing distances in kilometers.
        """
        if not isinstance(d_curve, DCurve):
            raise TypeError(
                f"d_curve must be a DCurve, got {type(d_curve).__name__}."
            )

        l_mf_km: NDArray[np.float64] = self._as_1d_float_array(
            d_curve.l_mf_km,
            "d_curve.l_mf_km",
        )
        d_km: NDArray[np.float64] = self._as_1d_float_array(
            d_curve.d_km,
            "d_curve.d_km",
        )
        self._require_same_length(l_mf_km, "l_mf_km", d_km, "d_km")

        if l_mf_km.size < 2:
            d_curve.zero_crossings_km = []
            d_curve.is_multivalued = False
            return []

        if not np.all(np.isfinite(l_mf_km)):
            d_curve.zero_crossings_km = []
            d_curve.is_multivalued = False
            return []

        zero_tolerance: float = self._zero_tolerance(d_km)
        signs: NDArray[np.int64] = self._sign_array(d_km, zero_tolerance)

        crossings_km: list[float] = []
        index: int = 0
        while index < d_km.size:
            sign_value: int = int(signs[index])

            if sign_value == 0:
                run_start: int = index
                while index + 1 < d_km.size and int(signs[index + 1]) == 0:
                    index += 1
                run_end: int = index

                crossing_km: float | None = self._zero_run_crossing(
                    l_mf_km=l_mf_km,
                    signs=signs,
                    run_start=run_start,
                    run_end=run_end,
                )
                if crossing_km is not None:
                    crossings_km.append(crossing_km)

                index += 1
                continue

            if sign_value in {-1, 1} and index + 1 < d_km.size:
                next_sign_value: int = int(signs[index + 1])
                if (
                    next_sign_value in {-1, 1}
                    and sign_value * next_sign_value < 0
                    and self._has_persistent_sign_left(signs, index, sign_value)
                    and self._has_persistent_sign_right(
                        signs,
                        index + 1,
                        next_sign_value,
                    )
                ):
                    crossing_km = self._linear_zero_crossing(
                        l0=float(l_mf_km[index]),
                        d0=float(d_km[index]),
                        l1=float(l_mf_km[index + 1]),
                        d1=float(d_km[index + 1]),
                    )
                    if crossing_km is not None:
                        crossings_km.append(crossing_km)

            index += 1

        crossings_km = self._deduplicate_crossings(crossings_km)

        d_curve.zero_crossings_km = crossings_km
        d_curve.is_multivalued = len(crossings_km) > 1

        return crossings_km

    def passes_final(self, result: GeolocationResult) -> tuple[bool, str]:
        """Apply final paper QC for an accepted geolocation candidate.

        Criteria are strict:
            single-valued geolocation
            Q > config.q_threshold
            cos(alpha) > config.cos_alpha_threshold

        Args:
            result: Candidate geolocation result.

        Returns:
            Tuple ``(passed, rejection_reason)``.
        """
        if not isinstance(result, GeolocationResult):
            return False, "invalid_final_result"

        if (
            self.require_single_valued_geolocation
            and self.discard_multivalued_geolocations
            and bool(result.is_multivalued)
        ):
            return False, "multivalued_geolocation"

        distance_km: float = float(result.distance_km)
        q_value: float = float(result.q)
        cos_alpha: float = float(result.cos_alpha)

        if not (
            math.isfinite(distance_km)
            and math.isfinite(q_value)
            and math.isfinite(cos_alpha)
        ):
            return False, "invalid_final_result"

        if q_value <= self.q_threshold:
            return False, "q_below_threshold"

        if cos_alpha <= self.cos_alpha_threshold:
            return False, "cos_alpha_below_threshold"

        return True, "passed"

    @staticmethod
    def _as_1d_float_array(values: NDArray[Any] | Any, name: str) -> NDArray[np.float64]:
        """Convert input to a one-dimensional float64 array."""
        try:
            array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be convertible to a float array.") from exc

        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {array.ndim}D.")

        return array.astype(np.float64, copy=False)

    @staticmethod
    def _require_same_length(
        first: NDArray[Any],
        first_name: str,
        second: NDArray[Any],
        second_name: str,
    ) -> None:
        """Validate equal one-dimensional array lengths."""
        if len(first) != len(second):
            raise ValueError(
                f"{first_name} and {second_name} must have equal length, got "
                f"{len(first)} and {len(second)}."
            )

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        """Validate a finite positive scalar."""
        scalar: float = float(value)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}.")
        return scalar

    @staticmethod
    def _nonnegative_float(value: Any, name: str) -> float:
        """Validate a finite nonnegative scalar."""
        scalar: float = float(value)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}.")
        return scalar

    @staticmethod
    def _bounded_float(value: Any, name: str, lower: float, upper: float) -> float:
        """Validate a finite scalar in a closed interval."""
        scalar: float = float(value)
        if not math.isfinite(scalar) or scalar < lower or scalar > upper:
            raise ValueError(
                f"{name} must be finite and in [{lower}, {upper}], got {value!r}."
            )
        return scalar

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        """Validate a nonnegative integer."""
        integer_value: int = int(value)
        if integer_value < 0:
            raise ValueError(f"{name} must be >= 0, got {value!r}.")
        return integer_value

    @staticmethod
    def _numeric_tolerance(value: float) -> float:
        """Return a scale-aware numerical tolerance."""
        return 64.0 * _NUMERICAL_EPS * max(1.0, abs(float(value)))

    @staticmethod
    def _select_representative_minimum_index(
        indices: NDArray[np.int64],
        curve_length: int,
    ) -> int:
        """Select a deterministic representative for tied flat minima.

        Consecutive tied indices are grouped, and the center of the group
        closest to the curve center is selected. This handles flat-bottom local
        minima deterministically without changing the paper's global-local
        minimum definition.
        """
        if indices.size == 1:
            return int(indices[0])

        sorted_indices: NDArray[np.int64] = np.sort(indices.astype(np.int64))
        groups: list[NDArray[np.int64]] = []
        start: int = 0

        for position in range(1, sorted_indices.size):
            if int(sorted_indices[position]) != int(sorted_indices[position - 1]) + 1:
                groups.append(sorted_indices[start:position])
                start = position
        groups.append(sorted_indices[start:])

        curve_center: float = 0.5 * float(curve_length - 1)
        best_group: NDArray[np.int64] = groups[0]
        best_distance_to_center: float = math.inf

        for group in groups:
            group_center: float = 0.5 * (float(group[0]) + float(group[-1]))
            distance_to_center: float = abs(group_center - curve_center)
            if distance_to_center < best_distance_to_center:
                best_distance_to_center = distance_to_center
                best_group = group

        representative_position: int = int(best_group.size // 2)
        return int(best_group[representative_position])

    def _locate_distance_index(
        self,
        distances_km: NDArray[np.float64],
        l0_km: float,
    ) -> int | None:
        """Locate the distance-grid index corresponding to l0."""
        differences: NDArray[np.float64] = np.abs(distances_km - float(l0_km))
        if differences.size == 0 or not np.any(np.isfinite(differences)):
            return None

        index: int = int(np.nanargmin(differences))
        tolerance_km: float = max(
            1.0e-9,
            1.0e-6 * max(self.bp_step_km, self.sample_spacing_km, 1.0),
        )
        if float(differences[index]) > tolerance_km:
            return None

        return index

    def _monotonic_endpoint_index(
        self,
        v_array: NDArray[np.float64],
        start_index: int,
        direction: int,
    ) -> int:
        """Find endpoint where monotonic rise away from minimum ends."""
        if direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1.")

        current_index: int = int(start_index)
        current_value: float = float(v_array[current_index])
        next_index: int = current_index + direction

        while 0 <= next_index < v_array.size:
            next_value: float = float(v_array[next_index])
            if not math.isfinite(next_value):
                break

            tolerance: float = self._numeric_tolerance(
                max(abs(current_value), abs(next_value), 1.0)
            )
            if next_value + tolerance < current_value:
                break

            current_index = next_index
            current_value = next_value
            next_index = current_index + direction

        return current_index

    def _sign_array(
        self,
        values: NDArray[np.float64],
        zero_tolerance: float,
    ) -> NDArray[np.int64]:
        """Convert values to sign codes: -1, 0, +1, or 99 invalid."""
        signs: NDArray[np.int64] = np.full(values.shape, 99, dtype=np.int64)
        finite_mask: NDArray[np.bool_] = np.isfinite(values)

        signs[finite_mask & (values > zero_tolerance)] = 1
        signs[finite_mask & (values < -zero_tolerance)] = -1
        signs[finite_mask & (np.abs(values) <= zero_tolerance)] = 0

        return signs

    @staticmethod
    def _zero_tolerance(values: NDArray[np.float64]) -> float:
        """Return numerical zero tolerance for a D-curve."""
        finite_values: NDArray[np.float64] = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 0.0
        scale: float = float(np.max(np.abs(finite_values)))
        return 64.0 * _NUMERICAL_EPS * max(1.0, scale)

    def _has_persistent_sign_left(
        self,
        signs: NDArray[np.int64],
        end_index: int,
        required_sign: int,
    ) -> bool:
        """Check contiguous sign persistence ending at an index."""
        required_count: int = self.zero_crossing_persistence_samples
        if required_count <= 0:
            return True

        if end_index - required_count + 1 < 0:
            return False

        for index in range(end_index, end_index - required_count, -1):
            if int(signs[index]) != int(required_sign):
                return False

        return True

    def _has_persistent_sign_right(
        self,
        signs: NDArray[np.int64],
        start_index: int,
        required_sign: int,
    ) -> bool:
        """Check contiguous sign persistence starting at an index."""
        required_count: int = self.zero_crossing_persistence_samples
        if required_count <= 0:
            return True

        if start_index + required_count > signs.size:
            return False

        for index in range(start_index, start_index + required_count):
            if int(signs[index]) != int(required_sign):
                return False

        return True

    def _zero_run_crossing(
        self,
        l_mf_km: NDArray[np.float64],
        signs: NDArray[np.int64],
        run_start: int,
        run_end: int,
    ) -> float | None:
        """Evaluate an exact-zero sample or plateau as one crossing."""
        left_index: int = run_start - 1
        right_index: int = run_end + 1

        if left_index < 0 or right_index >= signs.size:
            return None

        left_sign: int = int(signs[left_index])
        right_sign: int = int(signs[right_index])

        if left_sign not in {-1, 1} or right_sign not in {-1, 1}:
            return None
        if left_sign * right_sign >= 0:
            return None

        if not self._has_persistent_sign_left(signs, left_index, left_sign):
            return None
        if not self._has_persistent_sign_right(signs, right_index, right_sign):
            return None

        crossing_km: float = 0.5 * (
            float(l_mf_km[run_start]) + float(l_mf_km[run_end])
        )
        return crossing_km if math.isfinite(crossing_km) else None

    @staticmethod
    def _linear_zero_crossing(
        l0: float,
        d0: float,
        l1: float,
        d1: float,
    ) -> float | None:
        """Linearly interpolate a D=0 crossing between two samples."""
        denominator: float = d1 - d0
        if not (
            math.isfinite(l0)
            and math.isfinite(l1)
            and math.isfinite(d0)
            and math.isfinite(d1)
            and math.isfinite(denominator)
        ):
            return None
        if denominator == 0.0:
            return None

        crossing_km: float = l0 - d0 * (l1 - l0) / denominator
        if not math.isfinite(crossing_km):
            return None

        lower_bound: float = min(l0, l1) - QualityController._numeric_tolerance(
            max(abs(l0), abs(l1), 1.0)
        )
        upper_bound: float = max(l0, l1) + QualityController._numeric_tolerance(
            max(abs(l0), abs(l1), 1.0)
        )
        if crossing_km < lower_bound or crossing_km > upper_bound:
            return None

        return float(crossing_km)

    def _deduplicate_crossings(self, crossings_km: list[float]) -> list[float]:
        """Sort and merge numerically duplicate crossing distances."""
        finite_crossings: list[float] = [
            float(value) for value in crossings_km if math.isfinite(float(value))
        ]
        if not finite_crossings:
            return []

        finite_crossings.sort()
        deduplicated: list[float] = [finite_crossings[0]]
        tolerance_km: float = max(1.0e-9, 1.0e-6 * self.bp_step_km)

        for crossing_km in finite_crossings[1:]:
            if abs(crossing_km - deduplicated[-1]) <= tolerance_km:
                deduplicated[-1] = 0.5 * (deduplicated[-1] + crossing_km)
            else:
                deduplicated.append(crossing_km)

        return deduplicated


__all__ = ["QualityController"]
