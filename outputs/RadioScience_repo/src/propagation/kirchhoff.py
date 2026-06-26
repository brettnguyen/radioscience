"""Scalar Kirchhoff diffraction-integral propagation for diagnostics.

This module implements the optional ``KirchhoffPropagator`` interface from the
project design. It is a secondary validation path for arbitrary receiver/source
trajectories and is not the configured operational propagation method. The
paper's large-scale reproduction path uses FFT plane-wave propagation, while
this class supports selected diagnostic comparisons, especially for geometry
corrections that retain an arbitrary receiver trajectory.

The parsed paper text omits the exact Kirchhoff equation. The implementation
therefore uses a conservative two-dimensional scalar high-frequency
Kirchhoff/Rayleigh-Sommerfeld-style approximation:

    u(r_i) ~= C * sum_j u(r'_j)
                    * cos(phi_ij)
                    * exp(-i k rho_ij)
                    / sqrt(rho_ij)
                    * ds_j

where:
    * ``r'_j`` are source trajectory samples,
    * ``r_i`` are target points,
    * ``rho_ij = |r_i - r'_j|``,
    * ``ds_j`` are source arc-length integration weights,
    * ``cos(phi_ij)`` is an estimated obliquity factor for 2D curves,
    * ``C = sqrt(k / (2*pi*i))`` is a global asymptotic prefactor,
    * ``k = 2*pi/lambda``.

For back propagation of a field convention ``u = A * exp(i*phase)`` with
forward propagation accumulating ``exp(+i k L)``, this diagnostic method uses
``exp(-i k rho)`` to remove propagation phase.

Important limitations:
    * The physically meaningful obliquity estimate is implemented only for
      2D projected BP-plane trajectories. For 3D curve inputs, a unique
      surface normal is not defined by the public interface, so the obliquity
      factor defaults to one.
    * The direct integral is O(MN), where N is the number of source samples and
      M is the number of target points. It is suitable for selected diagnostics,
      not full nested 60 x 60 COSMIC-2 processing loops.
    * Numerical noise increases when targets are close to the source trajectory,
      consistent with the paper's warning about the Fresnel zone becoming
      comparable to sample spacing. Distances are floored by a data-dependent
      lower bound to prevent singularities.

The public API intentionally contains only:
    * ``KirchhoffPropagator(wavelength_m)``
    * ``backpropagate_arbitrary_trajectory(signal, source_points_m,
      target_points_m)``
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core import constants


_MIN_SOURCE_SAMPLES: int = 2
_SUPPORTED_COORDINATE_DIMS: frozenset[int] = frozenset({2, 3})
_DISTANCE_FLOOR_FRACTION_OF_MIN_SPACING: float = 0.5
_DISTANCE_FLOOR_FRACTION_OF_WAVELENGTH: float = 0.25
_ABSOLUTE_DISTANCE_FLOOR_M: float = 1.0e-9
_TRAJECTORY_NORM_EPS_M: float = 1.0e-12
_NORMAL_NORM_EPS: float = 1.0e-15
_DEFAULT_MAX_KERNEL_BYTES: int = 256 * 1024 * 1024


class KirchhoffPropagator:
    """Scalar diffraction-integral back propagator for arbitrary trajectories.

    Args:
        wavelength_m: Positive carrier wavelength in meters.

    The method implemented here is an approximate diagnostic counterpart to the
    paper's arbitrary-trajectory Kirchhoff discussion. It should be used for
    tests and selected event diagnostics, while the configured main method
    remains FFT plane-wave propagation.
    """

    def __init__(self, wavelength_m: float) -> None:
        """Initialize the Kirchhoff propagator.

        Args:
            wavelength_m: Positive electromagnetic wavelength in meters.

        Raises:
            ValueError: If ``wavelength_m`` is not finite and positive.
        """
        self.wavelength_m: float = self._validate_positive_scalar(
            wavelength_m,
            "wavelength_m",
        )
        self.wave_number_rad_per_m: float = constants.wave_number_rad_per_m(
            self.wavelength_m
        )

        # sqrt(k / (2*pi*i)) = sqrt(k/(2*pi)) * exp(-i*pi/4).
        # This global factor does not affect normalized amplitude variance V,
        # but retaining it makes the integral dimensionally closer to the
        # asymptotic 2D Green-function form.
        self._prefactor: complex = complex(
            np.sqrt(self.wave_number_rad_per_m / (constants.TWO_PI * 1j))
        )

    def backpropagate_arbitrary_trajectory(
        self,
        signal: NDArray[Any],
        source_points_m: NDArray[Any],
        target_points_m: NDArray[Any],
    ) -> NDArray[np.complex128]:
        """Back propagate a complex field from an arbitrary source trajectory.

        Args:
            signal: Complex source-trajectory field samples with shape ``(N,)``.
            source_points_m: Source trajectory coordinates in meters with shape
                ``(N, 2)`` or ``(N, 3)``.
            target_points_m: Target coordinates in meters with shape ``(M, 2)``
                or ``(M, 3)``. A single target supplied as shape ``(D,)`` is
                accepted and treated as ``(1, D)``.

        Returns:
            Complex reconstructed field at target points with shape ``(M,)``.

        Raises:
            ValueError: If arrays are invalid, dimensions do not match, fewer
                than two source samples are provided, or source arc-length
                weights cannot be computed.
        """
        signal_array: NDArray[np.complex128] = self._validate_signal(signal)
        source_array_m: NDArray[np.float64] = self._validate_source_points(
            source_points_m=source_points_m,
            expected_length=signal_array.size,
        )
        target_array_m: NDArray[np.float64] = self._validate_target_points(
            target_points_m=target_points_m,
            expected_dimension=source_array_m.shape[1],
        )

        if target_array_m.shape[0] == 0:
            return np.empty(0, dtype=np.complex128)

        arc_weights_m: NDArray[np.float64] = self._arc_length_weights_m(source_array_m)
        distance_floor_m: float = self._distance_floor_m(
            source_points_m=source_array_m,
            arc_weights_m=arc_weights_m,
        )

        source_normals: NDArray[np.float64] | None = None
        if source_array_m.shape[1] == 2:
            source_normals = self._estimate_2d_normals(source_array_m)

        weighted_signal: NDArray[np.complex128] = (
            signal_array * arc_weights_m.astype(np.complex128, copy=False)
        )

        output_field: NDArray[np.complex128] = np.empty(
            target_array_m.shape[0],
            dtype=np.complex128,
        )

        chunk_size: int = self._target_chunk_size(
            source_count=source_array_m.shape[0],
            coordinate_dimension=source_array_m.shape[1],
        )

        for start_index in range(0, target_array_m.shape[0], chunk_size):
            end_index: int = min(start_index + chunk_size, target_array_m.shape[0])
            output_field[start_index:end_index] = self._evaluate_target_chunk(
                source_signal_weighted=weighted_signal,
                source_points_m=source_array_m,
                target_points_m=target_array_m[start_index:end_index],
                source_normals=source_normals,
                distance_floor_m=distance_floor_m,
            )

        return output_field

    def _evaluate_target_chunk(
        self,
        source_signal_weighted: NDArray[np.complex128],
        source_points_m: NDArray[np.float64],
        target_points_m: NDArray[np.float64],
        source_normals: NDArray[np.float64] | None,
        distance_floor_m: float,
    ) -> NDArray[np.complex128]:
        """Evaluate the direct Kirchhoff sum for a chunk of target points.

        Args:
            source_signal_weighted: Source samples multiplied by arc weights.
            source_points_m: Source coordinates with shape ``(N, D)``.
            target_points_m: Target coordinates with shape ``(C, D)``.
            source_normals: Optional 2D unit normals with shape ``(N, 2)``.
            distance_floor_m: Lower bound applied to all source-target
                distances.

        Returns:
            Complex field for the target chunk with shape ``(C,)``.
        """
        displacement_m: NDArray[np.float64] = (
            target_points_m[:, np.newaxis, :] - source_points_m[np.newaxis, :, :]
        )
        distances_m: NDArray[np.float64] = np.linalg.norm(displacement_m, axis=2)
        safe_distances_m: NDArray[np.float64] = np.maximum(
            distances_m,
            distance_floor_m,
        )

        if source_normals is None:
            obliquity: NDArray[np.float64] = np.ones_like(
                safe_distances_m,
                dtype=np.float64,
            )
        else:
            normal_projection: NDArray[np.float64] = np.einsum(
                "cnd,nd->cn",
                displacement_m,
                source_normals,
                optimize=True,
            )
            obliquity = np.abs(normal_projection) / safe_distances_m
            obliquity = np.clip(obliquity, 0.0, 1.0)

        phase_argument_rad: NDArray[np.float64] = (
            -self.wave_number_rad_per_m * safe_distances_m
        )
        kernel: NDArray[np.complex128] = (
            obliquity
            * np.exp(1j * phase_argument_rad)
            / np.sqrt(safe_distances_m)
        ).astype(np.complex128, copy=False)

        chunk_field: NDArray[np.complex128] = self._prefactor * np.sum(
            kernel * source_signal_weighted[np.newaxis, :],
            axis=1,
        )

        return chunk_field.astype(np.complex128, copy=False)

    @staticmethod
    def _validate_signal(signal: NDArray[Any]) -> NDArray[np.complex128]:
        """Validate source complex signal samples.

        Args:
            signal: Array-like complex signal.

        Returns:
            One-dimensional complex128 array.

        Raises:
            ValueError: If the signal is invalid.
        """
        try:
            signal_array: NDArray[np.complex128] = np.asarray(
                signal,
                dtype=np.complex128,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("signal must be convertible to a complex array.") from exc

        if signal_array.ndim != 1:
            raise ValueError(f"signal must be one-dimensional, got {signal_array.ndim}D.")
        if signal_array.size < _MIN_SOURCE_SAMPLES:
            raise ValueError(
                f"signal must contain at least {_MIN_SOURCE_SAMPLES} samples, "
                f"got {signal_array.size}."
            )
        if not np.all(np.isfinite(signal_array.real)) or not np.all(
            np.isfinite(signal_array.imag)
        ):
            raise ValueError("signal contains NaN or Inf values.")

        return signal_array.astype(np.complex128, copy=False)

    @staticmethod
    def _validate_source_points(
        source_points_m: NDArray[Any],
        expected_length: int,
    ) -> NDArray[np.float64]:
        """Validate source trajectory coordinates.

        Args:
            source_points_m: Source coordinates.
            expected_length: Required number of rows matching signal samples.

        Returns:
            Float64 array with shape ``(N, 2)`` or ``(N, 3)``.

        Raises:
            ValueError: If validation fails.
        """
        try:
            source_array_m: NDArray[np.float64] = np.asarray(
                source_points_m,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "source_points_m must be convertible to a float array."
            ) from exc

        if source_array_m.ndim != 2:
            raise ValueError(
                "source_points_m must have shape (N, 2) or (N, 3), "
                f"got {source_array_m.ndim}D."
            )
        if source_array_m.shape[0] != int(expected_length):
            raise ValueError(
                "source_points_m first dimension must equal signal length, got "
                f"{source_array_m.shape[0]} and {int(expected_length)}."
            )
        if source_array_m.shape[0] < _MIN_SOURCE_SAMPLES:
            raise ValueError(
                f"source_points_m must contain at least {_MIN_SOURCE_SAMPLES} "
                f"points, got {source_array_m.shape[0]}."
            )
        if source_array_m.shape[1] not in _SUPPORTED_COORDINATE_DIMS:
            raise ValueError(
                "source_points_m coordinate dimension must be 2 or 3, got "
                f"{source_array_m.shape[1]}."
            )
        if not np.all(np.isfinite(source_array_m)):
            raise ValueError("source_points_m contains NaN or Inf values.")

        return source_array_m.astype(np.float64, copy=False)

    @staticmethod
    def _validate_target_points(
        target_points_m: NDArray[Any],
        expected_dimension: int,
    ) -> NDArray[np.float64]:
        """Validate target coordinates.

        Args:
            target_points_m: Target coordinates.
            expected_dimension: Coordinate dimension required to match sources.

        Returns:
            Float64 target array with shape ``(M, D)``.

        Raises:
            ValueError: If validation fails.
        """
        try:
            target_array_m: NDArray[np.float64] = np.asarray(
                target_points_m,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "target_points_m must be convertible to a float array."
            ) from exc

        if target_array_m.ndim == 1:
            if target_array_m.size != int(expected_dimension):
                raise ValueError(
                    "Single target point dimension must match source coordinate "
                    f"dimension {int(expected_dimension)}, got {target_array_m.size}."
                )
            target_array_m = target_array_m.reshape(1, int(expected_dimension))

        if target_array_m.ndim != 2:
            raise ValueError(
                "target_points_m must have shape (M, 2) or (M, 3), "
                f"got {target_array_m.ndim}D."
            )
        if target_array_m.shape[1] != int(expected_dimension):
            raise ValueError(
                "target_points_m coordinate dimension must match "
                f"source_points_m dimension {int(expected_dimension)}, got "
                f"{target_array_m.shape[1]}."
            )
        if target_array_m.shape[1] not in _SUPPORTED_COORDINATE_DIMS:
            raise ValueError(
                "target_points_m coordinate dimension must be 2 or 3, got "
                f"{target_array_m.shape[1]}."
            )
        if not np.all(np.isfinite(target_array_m)):
            raise ValueError("target_points_m contains NaN or Inf values.")

        return target_array_m.astype(np.float64, copy=False)

    @staticmethod
    def _arc_length_weights_m(
        source_points_m: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute trapezoidal arc-length integration weights.

        For ordered trajectory samples:
            * first weight is half the first segment length,
            * interior weights are half the sum of neighboring segment lengths,
            * last weight is half the last segment length.

        Args:
            source_points_m: Source trajectory coordinates with shape ``(N, D)``.

        Returns:
            Arc-length weights in meters with shape ``(N,)``.

        Raises:
            ValueError: If segment lengths are invalid or degenerate.
        """
        segment_vectors_m: NDArray[np.float64] = np.diff(source_points_m, axis=0)
        segment_lengths_m: NDArray[np.float64] = np.linalg.norm(
            segment_vectors_m,
            axis=1,
        )

        if segment_lengths_m.size < 1:
            raise ValueError("At least one source trajectory segment is required.")
        if not np.all(np.isfinite(segment_lengths_m)):
            raise ValueError("Source trajectory segment lengths are non-finite.")
        if np.any(segment_lengths_m <= _TRAJECTORY_NORM_EPS_M):
            raise ValueError(
                "Source trajectory contains duplicate or near-duplicate adjacent "
                "points; Kirchhoff arc-length weights are undefined."
            )

        weights_m: NDArray[np.float64] = np.empty(
            source_points_m.shape[0],
            dtype=np.float64,
        )
        weights_m[0] = 0.5 * segment_lengths_m[0]
        weights_m[-1] = 0.5 * segment_lengths_m[-1]

        if weights_m.size > 2:
            weights_m[1:-1] = 0.5 * (
                segment_lengths_m[:-1] + segment_lengths_m[1:]
            )

        if not np.all(np.isfinite(weights_m)) or np.any(weights_m <= 0.0):
            raise ValueError("Computed arc-length weights are invalid.")

        return weights_m

    def _distance_floor_m(
        self,
        source_points_m: NDArray[np.float64],
        arc_weights_m: NDArray[np.float64],
    ) -> float:
        """Compute a safe lower bound for source-target distances.

        Args:
            source_points_m: Source trajectory coordinates.
            arc_weights_m: Arc-length integration weights.

        Returns:
            Positive distance floor in meters.
        """
        segment_lengths_m: NDArray[np.float64] = np.linalg.norm(
            np.diff(source_points_m, axis=0),
            axis=1,
        )
        positive_segment_lengths_m: NDArray[np.float64] = segment_lengths_m[
            np.isfinite(segment_lengths_m) & (segment_lengths_m > 0.0)
        ]
        positive_weights_m: NDArray[np.float64] = arc_weights_m[
            np.isfinite(arc_weights_m) & (arc_weights_m > 0.0)
        ]

        spacing_candidates: list[float] = [
            _ABSOLUTE_DISTANCE_FLOOR_M,
            _DISTANCE_FLOOR_FRACTION_OF_WAVELENGTH * self.wavelength_m,
        ]

        if positive_segment_lengths_m.size > 0:
            spacing_candidates.append(
                _DISTANCE_FLOOR_FRACTION_OF_MIN_SPACING
                * float(np.min(positive_segment_lengths_m))
            )

        if positive_weights_m.size > 0:
            spacing_candidates.append(
                _DISTANCE_FLOOR_FRACTION_OF_MIN_SPACING
                * float(np.min(positive_weights_m))
            )

        distance_floor_m: float = max(spacing_candidates)
        if not math.isfinite(distance_floor_m) or distance_floor_m <= 0.0:
            return _ABSOLUTE_DISTANCE_FLOOR_M

        return distance_floor_m

    @staticmethod
    def _estimate_2d_normals(
        source_points_m: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Estimate local 2D unit normals from a source trajectory.

        The public interface does not provide source normals. For a 2D curve,
        this method estimates tangents from neighboring samples and rotates
        them by 90 degrees. The downstream obliquity calculation uses the
        absolute normal projection, so the arbitrary sign of the normal does not
        affect the result.

        Args:
            source_points_m: Source trajectory coordinates with shape ``(N, 2)``.

        Returns:
            Unit normal vectors with shape ``(N, 2)``.

        Raises:
            ValueError: If tangents/normals cannot be estimated.
        """
        if source_points_m.ndim != 2 or source_points_m.shape[1] != 2:
            raise ValueError("2D normal estimation requires shape (N, 2).")

        source_count: int = source_points_m.shape[0]
        tangents: NDArray[np.float64] = np.empty_like(source_points_m, dtype=np.float64)

        tangents[0] = source_points_m[1] - source_points_m[0]
        tangents[-1] = source_points_m[-1] - source_points_m[-2]

        if source_count > 2:
            tangents[1:-1] = source_points_m[2:] - source_points_m[:-2]

            tangent_norms: NDArray[np.float64] = np.linalg.norm(tangents[1:-1], axis=1)
            bad_interior_indices: NDArray[np.int64] = (
                np.flatnonzero(tangent_norms <= _TRAJECTORY_NORM_EPS_M) + 1
            )

            for index in bad_interior_indices:
                left_vector: NDArray[np.float64] = (
                    source_points_m[index] - source_points_m[index - 1]
                )
                right_vector: NDArray[np.float64] = (
                    source_points_m[index + 1] - source_points_m[index]
                )
                left_norm: float = float(np.linalg.norm(left_vector))
                right_norm: float = float(np.linalg.norm(right_vector))

                if left_norm > _TRAJECTORY_NORM_EPS_M and right_norm > _TRAJECTORY_NORM_EPS_M:
                    tangents[index] = left_vector / left_norm + right_vector / right_norm
                elif left_norm > _TRAJECTORY_NORM_EPS_M:
                    tangents[index] = left_vector
                elif right_norm > _TRAJECTORY_NORM_EPS_M:
                    tangents[index] = right_vector
                else:
                    raise ValueError(
                        "Cannot estimate tangent for source trajectory normal."
                    )

        tangent_norms_all: NDArray[np.float64] = np.linalg.norm(tangents, axis=1)
        if not np.all(np.isfinite(tangent_norms_all)) or np.any(
            tangent_norms_all <= _TRAJECTORY_NORM_EPS_M
        ):
            raise ValueError(
                "Cannot estimate finite nonzero tangents for source trajectory."
            )

        unit_tangents: NDArray[np.float64] = tangents / tangent_norms_all[:, np.newaxis]

        normals: NDArray[np.float64] = np.empty_like(unit_tangents, dtype=np.float64)
        normals[:, 0] = -unit_tangents[:, 1]
        normals[:, 1] = unit_tangents[:, 0]

        normal_norms: NDArray[np.float64] = np.linalg.norm(normals, axis=1)
        if not np.all(np.isfinite(normal_norms)) or np.any(
            normal_norms <= _NORMAL_NORM_EPS
        ):
            raise ValueError("Estimated source trajectory normals are invalid.")

        normals = normals / normal_norms[:, np.newaxis]
        return normals.astype(np.float64, copy=False)

    @staticmethod
    def _target_chunk_size(
        source_count: int,
        coordinate_dimension: int,
    ) -> int:
        """Choose a target chunk size for bounded memory use.

        Args:
            source_count: Number of source trajectory samples.
            coordinate_dimension: Coordinate dimension, 2 or 3.

        Returns:
            Positive target chunk size.
        """
        # Approximate per target-source pair memory:
        # displacement D float64 values + distance + obliquity + kernel complex.
        bytes_per_pair: int = (
            int(coordinate_dimension) * np.dtype(np.float64).itemsize
            + 2 * np.dtype(np.float64).itemsize
            + np.dtype(np.complex128).itemsize
        )
        bytes_per_target: int = max(1, int(source_count) * bytes_per_pair)

        chunk_size: int = max(1, _DEFAULT_MAX_KERNEL_BYTES // bytes_per_target)
        return int(chunk_size)

    @staticmethod
    def _validate_finite_scalar(value: float, name: str) -> float:
        """Validate a finite scalar.

        Args:
            value: Scalar value.
            name: Name used in error messages.

        Returns:
            Float scalar.

        Raises:
            ValueError: If conversion fails or value is not finite.
        """
        try:
            scalar_value: float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite scalar.") from exc

        if not math.isfinite(scalar_value):
            raise ValueError(f"{name} must be finite, got {value!r}.")

        return scalar_value

    @classmethod
    def _validate_positive_scalar(cls, value: float, name: str) -> float:
        """Validate a finite positive scalar.

        Args:
            value: Scalar value.
            name: Name used in error messages.

        Returns:
            Float scalar.

        Raises:
            ValueError: If value is not finite and strictly positive.
        """
        scalar_value: float = cls._validate_finite_scalar(value, name)
        if scalar_value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {scalar_value}.")
        return scalar_value


__all__ = ["KirchhoffPropagator"]
