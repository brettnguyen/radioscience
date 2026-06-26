"""Stationary-transmitter and wavefront-curvature correction for real BP geometry.

This module implements the real-observation geometry correction described in
Section 3 of the paper. For each candidate back-propagation screen distance
``L`` it transforms the observed COSMIC-2 high-rate complex signal into a
stationary, two-dimensional signal sampled on a uniform ``z`` grid suitable for
FFT plane-wave back propagation.

The correction follows the paper's Figure 14 geometry. For each high-rate
sample, transmitter and receiver positions are projected into the BP plane:

    transmitter: (x1, z1)
    receiver:    (x2, z2)

The candidate phase screen is placed at ``x = 0``. The original Tx-Rx ray
crosses the screen at:

    z_s = z1 - x1 * (z2 - z1) / (x2 - x1)

The GNSS transmitter is then held stationary at the midpoint coordinate
``(x10, 0)``, and the corrected receiver coordinate is:

    z2' = z_s * (1 - x2' / x10)

with the two options described in the paper:

    option 1: x2' = x2
    option 2: x2' = x20

where ``x20`` is the midpoint receiver x-coordinate. Option 2 is the configured
default in ``config.yaml`` and is the appropriate option for FFT propagation
after interpolation to a uniform z grid.

The phase/path correction implemented here is:

    phi' = phi + k * (L'_tr - L_tr)

where ``k = 2*pi/lambda``. Amplitude is not corrected. If enabled by
configuration defaults, the finite-wavefront-curvature correction is then added:

    phi'' = phi' + k * z^2 * tan(alpha)^2 / (2R)

where ``R`` is the midpoint three-dimensional Tx-Rx range.

Important phase-convention note:
    The paper's equation assumes the input phase is compatible with a path-based
    ``S - L_tr`` correction. The exact CDAAC phase convention and detrending
    choices are listed as unclear in ``config.yaml``. This module therefore
    does not detrend, unwrap, or otherwise reinterpret ``window.phase_rad``; it
    applies only the paper's geometric path correction to the phase supplied by
    upstream preprocessing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.config import AppConfig
from src.core import constants
from src.core.types import BpPlaneGeometry, CorrectedSignal, SignalWindow, StateVector, Vector3


_LOGGER = logging.getLogger(__name__)

_M_PER_KM: float = 1000.0
_MIN_SAMPLES: int = 2
_VECTOR_NORM_EPS: float = 1.0e-12
_DENOMINATOR_EPS_M: float = 1.0e-9
_DISTANCE_TOLERANCE_M: float = 1.0e-3
_X10_EPS_M: float = 1.0e-6
_DUPLICATE_Z_TOL_M: float = 1.0e-6
_UNIFORM_GRID_MIN_SPAN_M: float = 1.0e-9
_MAX_TAN_ALPHA: float = 1.0e8


class StationaryCorrectionError(ValueError):
    """Raised when stationary-transmitter correction cannot be computed."""


class _TxRxCorrection(NamedTuple):
    """Internal tuple returned by ``apply_tx_rx_correction``.

    The public design specifies that ``apply_tx_rx_correction`` returns a tuple
    but does not prescribe the tuple schema. This named tuple keeps the schema
    explicit while remaining tuple-compatible.
    """

    z2_prime_m: NDArray[np.float64]
    original_path_m: NDArray[np.float64]
    corrected_path_m: NDArray[np.float64]
    x2_prime_m: NDArray[np.float64]
    x10_m: float
    x20_m: float
    curvature_radius_m: float


class StationaryCorrector:
    """Apply real-geometry stationary-transmitter correction for BP processing.

    Args:
        config: Application configuration. The provided public ``AppConfig``
            exposes the paper-backed BP defaults. This class also supports
            future flat config attributes when present, while defaulting to the
            values in ``config.yaml`` and ``src.core.constants``.

    Public methods follow the project design:
        * ``correct(window, plane, screen_distance_km, option)``
        * ``apply_tx_rx_correction(window, plane, screen_distance_km, option)``
        * ``apply_phase_correction(...)``
        * ``apply_curvature_correction(...)``
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the stationary correction module.

        Args:
            config: Validated application configuration.

        Raises:
            TypeError: If ``config`` is not an ``AppConfig``.
            ValueError: If configured options are invalid.
        """
        if not isinstance(config, AppConfig):
            raise TypeError(
                f"config must be an AppConfig, got {type(config).__name__}."
            )

        self.config: AppConfig = config
        self.default_option: int = self._configured_int(
            names=(
                "stationary_transmitter_correction_option",
                "stationary_correction_option",
                "default_stationary_correction_option",
            ),
            default=constants.DEFAULT_STATIONARY_CORRECTION_OPTION,
        )
        self.default_option = self._validate_option(self.default_option)

        self.curvature_correction_enabled: bool = self._configured_bool(
            names=(
                "wavefront_curvature_correction_enabled",
                "curvature_correction_enabled",
            ),
            default=constants.DEFAULT_WAVEFRONT_CURVATURE_CORRECTION_ENABLED,
        )

    def correct(
        self,
        window: SignalWindow,
        plane: BpPlaneGeometry,
        screen_distance_km: float,
        option: int | None = None,
    ) -> CorrectedSignal:
        """Apply geometry, phase, curvature, and z-grid corrections.

        Args:
            window: Preprocessed high-rate signal window. ``phase_rad`` is used
                as supplied; it is not detrended or unwrapped here.
            plane: BP plane geometry for the current ``L_mf``.
            screen_distance_km: Inner-loop candidate screen distance ``L`` in
                kilometers from receiver toward transmitter.
            option: Stationary-transmitter correction option. ``1`` preserves
                sample-wise receiver ``x2`` and is intended for Kirchhoff-style
                diagnostics. ``2`` uses constant ``x20`` and is the configured
                FFT-compatible default.

        Returns:
            ``CorrectedSignal`` containing the corrected complex field on a
            strictly increasing, uniformly sampled ``z_m`` grid.

        Raises:
            StationaryCorrectionError: If geometry or interpolation is invalid.
            ValueError: If inputs are invalid.
        """
        self._validate_window(window)
        self._validate_plane(plane)
        correction_option: int = self._validate_option(
            self.default_option if option is None else option
        )
        screen_distance_value_km: float = self._validate_screen_distance_km(
            screen_distance_km
        )

        wavelength_m: float = self._resolve_wavelength_m(window)
        amplitude: NDArray[np.float64] = self._select_amplitude(window)
        phase_rad: NDArray[np.float64] = self._as_1d_float_array(
            window.phase_rad,
            "window.phase_rad",
        )

        self._require_same_length(amplitude, "amplitude", phase_rad, "phase_rad")

        tx_rx_correction: _TxRxCorrection = self.apply_tx_rx_correction(
            window=window,
            plane=plane,
            screen_distance_km=screen_distance_value_km,
            option=correction_option,
        )

        self._require_same_length(
            phase_rad,
            "phase_rad",
            tx_rx_correction.z2_prime_m,
            "z2_prime_m",
        )

        phase_corrected_rad: NDArray[np.float64] = self.apply_phase_correction(
            phase_rad=phase_rad,
            original_path_m=tx_rx_correction.original_path_m,
            corrected_path_m=tx_rx_correction.corrected_path_m,
            wavelength_m=wavelength_m,
        )

        if self.curvature_correction_enabled:
            phase_corrected_rad = self.apply_curvature_correction(
                phase_rad=phase_corrected_rad,
                z_m=tx_rx_correction.z2_prime_m,
                alpha_rad=plane.alpha_rad,
                radius_m=tx_rx_correction.curvature_radius_m,
                wavelength_m=wavelength_m,
            )

        raw_complex_signal: NDArray[np.complex128] = (
            amplitude * np.exp(1j * phase_corrected_rad)
        ).astype(np.complex128, copy=False)

        z_grid_m, complex_grid = self._interpolate_complex_to_uniform_z(
            z_m=tx_rx_correction.z2_prime_m,
            complex_signal=raw_complex_signal,
        )

        return CorrectedSignal(
            screen_distance_km=screen_distance_value_km,
            z_m=z_grid_m,
            complex_signal=complex_grid,
            amplitude=np.abs(complex_grid).astype(np.float64, copy=False),
            phase_rad=np.angle(complex_grid).astype(np.float64, copy=False),
            curvature_radius_m=tx_rx_correction.curvature_radius_m,
        )

    def apply_tx_rx_correction(
        self,
        window: SignalWindow,
        plane: BpPlaneGeometry,
        screen_distance_km: float,
        option: int,
    ) -> tuple:
        """Apply the paper's Tx/Rx stationary geometry correction.

        Args:
            window: Signal window with Tx/Rx state vectors.
            plane: BP-plane geometry. ``plane.x_axis`` and ``plane.z_axis`` are
                used for projection.
            screen_distance_km: Candidate screen distance from receiver toward
                transmitter in kilometers.
            option: ``1`` for ``x2'=x2`` or ``2`` for ``x2'=x20``.

        Returns:
            Tuple-compatible ``_TxRxCorrection`` containing corrected z samples,
            original and corrected 2D path lengths, corrected x samples, midpoint
            x coordinates, and curvature radius.

        Raises:
            StationaryCorrectionError: If the required geometry is missing,
                degenerate, or nonphysical.
            ValueError: If scalar inputs are invalid.
        """
        self._validate_window(window)
        self._validate_plane(plane)
        correction_option: int = self._validate_option(option)
        screen_distance_value_km: float = self._validate_screen_distance_km(
            screen_distance_km
        )

        rx_positions_m, tx_positions_m = self._state_position_arrays(window)
        sample_count: int = rx_positions_m.shape[0]

        if sample_count < _MIN_SAMPLES:
            raise StationaryCorrectionError(
                f"At least {_MIN_SAMPLES} Tx/Rx samples are required for "
                "stationary correction."
            )

        rx_mid_m, tx_mid_m = self._midpoint_positions(window=window)
        rx_mid_array_m: NDArray[np.float64] = rx_mid_m.to_array()
        tx_mid_array_m: NDArray[np.float64] = tx_mid_m.to_array()

        x_axis: NDArray[np.float64] = self._unit_axis_array(plane.x_axis, "plane.x_axis")
        z_axis: NDArray[np.float64] = self._unit_axis_array(plane.z_axis, "plane.z_axis")

        midpoint_delta_m: NDArray[np.float64] = tx_mid_array_m - rx_mid_array_m
        midpoint_range_m: float = float(np.linalg.norm(midpoint_delta_m))
        if not math.isfinite(midpoint_range_m) or midpoint_range_m <= _VECTOR_NORM_EPS:
            raise StationaryCorrectionError(
                "Midpoint Tx/Rx range is zero, near-zero, or non-finite."
            )

        screen_distance_m: float = screen_distance_value_km * _M_PER_KM
        if screen_distance_m > midpoint_range_m + _DISTANCE_TOLERANCE_M:
            raise StationaryCorrectionError(
                "Candidate screen distance exceeds midpoint Tx/Rx range: "
                f"screen_distance={screen_distance_value_km:.3f} km, "
                f"range={midpoint_range_m / _M_PER_KM:.3f} km."
            )

        direction_projection: float = float(np.dot(midpoint_delta_m, x_axis))
        if not math.isfinite(direction_projection) or abs(direction_projection) <= _VECTOR_NORM_EPS:
            raise StationaryCorrectionError(
                "BP x-axis is orthogonal to midpoint receiver-to-transmitter "
                "direction; cannot place screen along LOS."
            )

        sign_to_transmitter: float = 1.0 if direction_projection > 0.0 else -1.0
        screen_origin_m: NDArray[np.float64] = (
            rx_mid_array_m + sign_to_transmitter * screen_distance_m * x_axis
        )

        self._validate_screen_between_rx_tx(
            rx_mid_m=rx_mid_array_m,
            tx_mid_m=tx_mid_array_m,
            screen_origin_m=screen_origin_m,
            requested_distance_m=screen_distance_m,
        )

        tx_displacement_m: NDArray[np.float64] = tx_positions_m - screen_origin_m
        rx_displacement_m: NDArray[np.float64] = rx_positions_m - screen_origin_m

        x1_m: NDArray[np.float64] = tx_displacement_m @ x_axis
        z1_m: NDArray[np.float64] = tx_displacement_m @ z_axis
        x2_m: NDArray[np.float64] = rx_displacement_m @ x_axis
        z2_m: NDArray[np.float64] = rx_displacement_m @ z_axis

        x10_m: float = float(np.dot(tx_mid_array_m - screen_origin_m, x_axis))
        x20_m: float = float(np.dot(rx_mid_array_m - screen_origin_m, x_axis))

        if not math.isfinite(x10_m) or abs(x10_m) <= _X10_EPS_M:
            raise StationaryCorrectionError(
                "Stationary transmitter x10 is zero, near-zero, or non-finite. "
                "The candidate screen is too close to the transmitter in the "
                "projected BP geometry."
            )
        if not math.isfinite(x20_m):
            raise StationaryCorrectionError("Midpoint receiver x20 is non-finite.")

        denominator_m: NDArray[np.float64] = x2_m - x1_m
        valid_denominator_mask: NDArray[np.bool_] = (
            np.isfinite(denominator_m) & (np.abs(denominator_m) > _DENOMINATOR_EPS_M)
        )
        if not np.all(valid_denominator_mask):
            raise StationaryCorrectionError(
                "Cannot compute screen crossing height because one or more "
                "samples have x2 - x1 approximately zero or non-finite."
            )

        z_s_m: NDArray[np.float64] = (
            z1_m - x1_m * (z2_m - z1_m) / denominator_m
        ).astype(np.float64, copy=False)

        if not np.all(np.isfinite(z_s_m)):
            raise StationaryCorrectionError(
                "Computed phase-screen crossing heights z_s contain non-finite values."
            )

        if correction_option == 1:
            x2_prime_m: NDArray[np.float64] = x2_m.astype(np.float64, copy=True)
        else:
            x2_prime_m = np.full(sample_count, x20_m, dtype=np.float64)

        z2_prime_m: NDArray[np.float64] = (
            z_s_m * (1.0 - x2_prime_m / x10_m)
        ).astype(np.float64, copy=False)

        if not np.all(np.isfinite(z2_prime_m)):
            raise StationaryCorrectionError(
                "Corrected receiver coordinates z2_prime contain non-finite values."
            )

        original_path_m: NDArray[np.float64] = np.sqrt(
            (x2_m - x1_m) * (x2_m - x1_m)
            + (z2_m - z1_m) * (z2_m - z1_m)
        ).astype(np.float64, copy=False)

        corrected_path_m: NDArray[np.float64] = np.sqrt(
            (x2_prime_m - x10_m) * (x2_prime_m - x10_m)
            + z2_prime_m * z2_prime_m
        ).astype(np.float64, copy=False)

        if not np.all(np.isfinite(original_path_m)) or np.any(original_path_m <= 0.0):
            raise StationaryCorrectionError(
                "Original projected Tx/Rx path lengths are invalid."
            )
        if not np.all(np.isfinite(corrected_path_m)) or np.any(corrected_path_m <= 0.0):
            raise StationaryCorrectionError(
                "Corrected projected Tx/Rx path lengths are invalid."
            )

        return _TxRxCorrection(
            z2_prime_m=z2_prime_m,
            original_path_m=original_path_m,
            corrected_path_m=corrected_path_m,
            x2_prime_m=x2_prime_m,
            x10_m=x10_m,
            x20_m=x20_m,
            curvature_radius_m=midpoint_range_m,
        )

    def apply_phase_correction(
        self,
        phase_rad: NDArray[Any],
        original_path_m: NDArray[Any],
        corrected_path_m: NDArray[Any],
        wavelength_m: float,
    ) -> NDArray[np.float64]:
        """Apply the paper's path-length phase correction.

        The implemented correction is:

            phase_corrected = phase + k * (corrected_path - original_path)

        where ``k = 2*pi/wavelength``.

        Args:
            phase_rad: Input phase in radians.
            original_path_m: Original projected path length ``L_tr`` in meters.
            corrected_path_m: Corrected projected path length ``L'_tr`` in meters.
            wavelength_m: Carrier wavelength in meters.

        Returns:
            Corrected phase in radians.

        Raises:
            ValueError: If arrays are invalid, lengths differ, or wavelength is
                not positive.
        """
        phase_array: NDArray[np.float64] = self._as_1d_float_array(
            phase_rad,
            "phase_rad",
        )
        original_path_array_m: NDArray[np.float64] = self._as_1d_float_array(
            original_path_m,
            "original_path_m",
        )
        corrected_path_array_m: NDArray[np.float64] = self._as_1d_float_array(
            corrected_path_m,
            "corrected_path_m",
        )
        wavelength_value_m: float = self._validate_positive_scalar(
            wavelength_m,
            "wavelength_m",
        )

        self._require_same_length(
            phase_array,
            "phase_rad",
            original_path_array_m,
            "original_path_m",
        )
        self._require_same_length(
            phase_array,
            "phase_rad",
            corrected_path_array_m,
            "corrected_path_m",
        )

        if not np.all(np.isfinite(phase_array)):
            raise ValueError("phase_rad contains non-finite values.")
        if not np.all(np.isfinite(original_path_array_m)):
            raise ValueError("original_path_m contains non-finite values.")
        if not np.all(np.isfinite(corrected_path_array_m)):
            raise ValueError("corrected_path_m contains non-finite values.")

        wave_number_rad_per_m: float = constants.wave_number_rad_per_m(
            wavelength_value_m
        )
        corrected_phase_rad: NDArray[np.float64] = (
            phase_array
            + wave_number_rad_per_m
            * (corrected_path_array_m - original_path_array_m)
        ).astype(np.float64, copy=False)

        if not np.all(np.isfinite(corrected_phase_rad)):
            raise ValueError("Phase correction produced non-finite values.")

        return corrected_phase_rad

    def apply_curvature_correction(
        self,
        phase_rad: NDArray[Any],
        z_m: NDArray[Any],
        alpha_rad: float,
        radius_m: float,
        wavelength_m: float,
    ) -> NDArray[np.float64]:
        """Apply the finite-wavefront-curvature correction.

        The paper's path correction is:

            Delta S = z^2 * tan(alpha)^2 / (2R)

        and this method applies the corresponding phase term:

            Delta phi = k * Delta S

        Args:
            phase_rad: Input phase in radians.
            z_m: Corrected receiver transverse coordinate in meters.
            alpha_rad: Scan angle in radians.
            radius_m: Wavefront curvature radius, taken as midpoint Tx/Rx range.
            wavelength_m: Carrier wavelength in meters.

        Returns:
            Phase after curvature correction in radians.

        Raises:
            ValueError: If inputs are invalid or the correction would be
                numerically unstable.
        """
        phase_array: NDArray[np.float64] = self._as_1d_float_array(
            phase_rad,
            "phase_rad",
        )
        z_array_m: NDArray[np.float64] = self._as_1d_float_array(z_m, "z_m")
        self._require_same_length(phase_array, "phase_rad", z_array_m, "z_m")

        if not np.all(np.isfinite(phase_array)):
            raise ValueError("phase_rad contains non-finite values.")
        if not np.all(np.isfinite(z_array_m)):
            raise ValueError("z_m contains non-finite values.")

        alpha_value_rad: float = self._validate_finite_scalar(alpha_rad, "alpha_rad")
        radius_value_m: float = self._validate_positive_scalar(radius_m, "radius_m")
        wavelength_value_m: float = self._validate_positive_scalar(
            wavelength_m,
            "wavelength_m",
        )

        tan_alpha: float = math.tan(alpha_value_rad)
        if not math.isfinite(tan_alpha):
            raise ValueError(
                "tan(alpha_rad) is non-finite; curvature correction cannot be applied."
            )
        if abs(tan_alpha) > _MAX_TAN_ALPHA:
            raise ValueError(
                "tan(alpha_rad) is too large for stable curvature correction. "
                "Such near-90-degree scan-angle cases should be rejected by "
                "downstream cos(alpha) QC."
            )

        wave_number_rad_per_m: float = constants.wave_number_rad_per_m(
            wavelength_value_m
        )
        path_correction_m: NDArray[np.float64] = (
            z_array_m
            * z_array_m
            * tan_alpha
            * tan_alpha
            / (2.0 * radius_value_m)
        ).astype(np.float64, copy=False)

        corrected_phase_rad: NDArray[np.float64] = (
            phase_array + wave_number_rad_per_m * path_correction_m
        ).astype(np.float64, copy=False)

        if not np.all(np.isfinite(corrected_phase_rad)):
            raise ValueError("Curvature correction produced non-finite phase values.")

        return corrected_phase_rad

    def _resolve_wavelength_m(self, window: SignalWindow) -> float:
        """Resolve carrier wavelength from ``SignalWindow`` metadata.

        Args:
            window: Signal window containing ``signal_name`` and
                ``constellation``.

        Returns:
            Carrier wavelength in meters.

        Raises:
            StationaryCorrectionError: If the wavelength cannot be resolved.
        """
        signal_name: str = str(window.signal_name or constants.DEFAULT_SIGNAL).strip()
        constellation: str = str(window.constellation or "").strip()

        try:
            return constants.signal_wavelength_m(
                signal_name=signal_name,
                constellation=constellation or None,
            )
        except ValueError as exc:
            constellation_key: str = constellation.upper()
            signal_key: str = signal_name.upper()

            # The public SignalWindow type does not include GLONASS FDMA
            # channel number. Use the configured/base GLONASS carrier constants
            # as a deterministic fallback rather than silently using GPS.
            if constellation_key in {"GLONASS", "GLO", "R"}:
                if "L2" in signal_key:
                    _LOGGER.warning(
                        "GLONASS L2 wavelength requested without FDMA channel "
                        "metadata; using GLONASS L2 base frequency."
                    )
                    return constants.frequency_to_wavelength_m(
                        constants.GLONASS_L2_BASE_FREQUENCY_HZ
                    )
                if "L1" in signal_key or not signal_key:
                    _LOGGER.warning(
                        "GLONASS L1 wavelength requested without FDMA channel "
                        "metadata; using GLONASS L1 base frequency."
                    )
                    return constants.frequency_to_wavelength_m(
                        constants.GLONASS_L1_BASE_FREQUENCY_HZ
                    )

            raise StationaryCorrectionError(
                "Could not resolve carrier wavelength from SignalWindow metadata "
                f"signal_name={window.signal_name!r}, "
                f"constellation={window.constellation!r}."
            ) from exc

    @staticmethod
    def _select_amplitude(window: SignalWindow) -> NDArray[np.float64]:
        """Select amplitude array, preserving the paper's no-amplitude-correction rule."""
        amplitude_array: NDArray[np.float64] = StationaryCorrector._as_1d_float_array(
            window.amplitude,
            "window.amplitude",
        )
        if np.any(np.isfinite(amplitude_array) & (amplitude_array > 0.0)):
            return amplitude_array.copy()

        snr_array: NDArray[np.float64] = StationaryCorrector._as_1d_float_array(
            window.snr_vv,
            "window.snr_vv",
        )
        if np.any(np.isfinite(snr_array) & (snr_array > 0.0)):
            return snr_array.copy()

        raise StationaryCorrectionError(
            "Neither window.amplitude nor window.snr_vv contains valid positive samples."
        )

    @staticmethod
    def _state_position_arrays(
        window: SignalWindow,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return receiver and transmitter position arrays with shape ``(N, 3)``."""
        sample_count: int = len(window.phase_rad)
        if sample_count < _MIN_SAMPLES:
            raise StationaryCorrectionError(
                f"Signal window must contain at least {_MIN_SAMPLES} samples."
            )

        if len(window.rx_states) != sample_count:
            raise StationaryCorrectionError(
                "Receiver state count must match signal sample count for "
                "stationary correction: "
                f"rx_states={len(window.rx_states)}, samples={sample_count}."
            )
        if len(window.tx_states) != sample_count:
            raise StationaryCorrectionError(
                "Transmitter state count must match signal sample count for "
                "stationary correction: "
                f"tx_states={len(window.tx_states)}, samples={sample_count}."
            )

        rx_positions_m: NDArray[np.float64] = np.asarray(
            [state.position_m.to_array() for state in window.rx_states],
            dtype=np.float64,
        )
        tx_positions_m: NDArray[np.float64] = np.asarray(
            [state.position_m.to_array() for state in window.tx_states],
            dtype=np.float64,
        )

        if rx_positions_m.shape != (sample_count, 3):
            raise StationaryCorrectionError(
                f"Receiver position array has invalid shape {rx_positions_m.shape}."
            )
        if tx_positions_m.shape != (sample_count, 3):
            raise StationaryCorrectionError(
                f"Transmitter position array has invalid shape {tx_positions_m.shape}."
            )
        if not np.all(np.isfinite(rx_positions_m)):
            raise StationaryCorrectionError("Receiver positions contain non-finite values.")
        if not np.all(np.isfinite(tx_positions_m)):
            raise StationaryCorrectionError("Transmitter positions contain non-finite values.")

        return rx_positions_m, tx_positions_m

    def _midpoint_positions(self, window: SignalWindow) -> tuple[Vector3, Vector3]:
        """Return linearly interpolated or nearest midpoint Tx/Rx positions."""
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

    def _position_at_time(
        self,
        states: list[StateVector],
        target_time: datetime,
        label: str,
    ) -> Vector3:
        """Return interpolated state position at a requested time."""
        if not states:
            raise StationaryCorrectionError(f"No {label} states are available.")

        target_seconds: float = self._datetime_seconds(target_time)
        sorted_states: list[StateVector] = sorted(
            states,
            key=lambda state: self._datetime_seconds(state.time),
        )
        state_times_s: NDArray[np.float64] = np.asarray(
            [self._datetime_seconds(state.time) for state in sorted_states],
            dtype=np.float64,
        )

        if state_times_s.size == 1:
            return sorted_states[0].position_m

        insertion_index: int = int(np.searchsorted(state_times_s, target_seconds))

        if 0 < insertion_index < state_times_s.size:
            lower_state: StateVector = sorted_states[insertion_index - 1]
            upper_state: StateVector = sorted_states[insertion_index]
            lower_time_s: float = float(state_times_s[insertion_index - 1])
            upper_time_s: float = float(state_times_s[insertion_index])
            span_s: float = upper_time_s - lower_time_s

            if span_s > 0.0:
                fraction: float = (target_seconds - lower_time_s) / span_s
                lower_position: NDArray[np.float64] = lower_state.position_m.to_array()
                upper_position: NDArray[np.float64] = upper_state.position_m.to_array()
                position: NDArray[np.float64] = (
                    lower_position + fraction * (upper_position - lower_position)
                )
                return self._array_to_vector(position, f"{label} midpoint position")

        nearest_index: int = int(np.argmin(np.abs(state_times_s - target_seconds)))
        return sorted_states[nearest_index].position_m

    @staticmethod
    def _datetime_seconds(value: datetime) -> float:
        """Convert datetime to POSIX seconds, treating naive datetimes as UTC."""
        if not isinstance(value, datetime):
            raise TypeError(f"State time must be datetime, got {type(value).__name__}.")

        if value.tzinfo is None:
            normalized: datetime = value.replace(tzinfo=timezone.utc)
        else:
            normalized = value.astimezone(timezone.utc)

        seconds: float = float(normalized.timestamp())
        if not math.isfinite(seconds):
            raise StationaryCorrectionError(f"Non-finite datetime timestamp: {value!r}.")
        return seconds

    @staticmethod
    def _validate_screen_between_rx_tx(
        rx_mid_m: NDArray[np.float64],
        tx_mid_m: NDArray[np.float64],
        screen_origin_m: NDArray[np.float64],
        requested_distance_m: float,
    ) -> None:
        """Ensure the screen origin lies between midpoint receiver and transmitter."""
        rx_to_tx_m: NDArray[np.float64] = tx_mid_m - rx_mid_m
        rx_to_screen_m: NDArray[np.float64] = screen_origin_m - rx_mid_m

        link_range_m: float = float(np.linalg.norm(rx_to_tx_m))
        screen_distance_m: float = float(np.linalg.norm(rx_to_screen_m))

        if abs(screen_distance_m - requested_distance_m) > max(
            _DISTANCE_TOLERANCE_M,
            1.0e-10 * max(1.0, requested_distance_m),
        ):
            raise StationaryCorrectionError(
                "Screen origin distance from receiver does not match requested "
                "screen_distance_km."
            )

        if requested_distance_m > _DISTANCE_TOLERANCE_M:
            direction_dot: float = float(np.dot(rx_to_screen_m, rx_to_tx_m))
            if direction_dot <= 0.0:
                raise StationaryCorrectionError(
                    "Screen origin is not in the receiver-to-transmitter direction. "
                    "Check BP x-axis sign convention."
                )

        screen_to_tx_m: float = float(np.linalg.norm(tx_mid_m - screen_origin_m))
        if screen_to_tx_m > link_range_m + _DISTANCE_TOLERANCE_M:
            raise StationaryCorrectionError(
                "Screen origin is farther from transmitter than the receiver, "
                "indicating invalid LOS placement."
            )

    def _interpolate_complex_to_uniform_z(
        self,
        z_m: NDArray[Any],
        complex_signal: NDArray[Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Interpolate a complex signal onto a strictly uniform z grid.

        Real and imaginary parts are interpolated separately to avoid direct
        interpolation of wrapped phase.

        Args:
            z_m: Nonuniform corrected z coordinates in meters.
            complex_signal: Complex signal samples at ``z_m``.

        Returns:
            Tuple ``(uniform_z_m, interpolated_complex_signal)``.

        Raises:
            StationaryCorrectionError: If insufficient valid, unique, or
                spatially extended samples exist.
        """
        z_array_m: NDArray[np.float64] = self._as_1d_float_array(z_m, "z_m")
        complex_array: NDArray[np.complex128] = np.asarray(
            complex_signal,
            dtype=np.complex128,
        )
        if complex_array.ndim != 1:
            raise StationaryCorrectionError(
                f"complex_signal must be one-dimensional, got {complex_array.ndim}D."
            )
        self._require_same_length(z_array_m, "z_m", complex_array, "complex_signal")

        valid_mask: NDArray[np.bool_] = (
            np.isfinite(z_array_m)
            & np.isfinite(complex_array.real)
            & np.isfinite(complex_array.imag)
        )
        if int(np.count_nonzero(valid_mask)) < _MIN_SAMPLES:
            raise StationaryCorrectionError(
                "Too few finite corrected complex samples for z-grid interpolation."
            )

        z_valid_m: NDArray[np.float64] = z_array_m[valid_mask]
        complex_valid: NDArray[np.complex128] = complex_array[valid_mask]

        sort_order: NDArray[np.int64] = np.argsort(z_valid_m)
        z_sorted_m: NDArray[np.float64] = z_valid_m[sort_order]
        complex_sorted: NDArray[np.complex128] = complex_valid[sort_order]

        z_unique_m, complex_unique = self._average_duplicate_z_samples(
            z_sorted_m=z_sorted_m,
            complex_sorted=complex_sorted,
        )

        if z_unique_m.size < _MIN_SAMPLES:
            raise StationaryCorrectionError(
                "Too few unique z samples after duplicate handling."
            )

        z_span_m: float = float(z_unique_m[-1] - z_unique_m[0])
        if not math.isfinite(z_span_m) or z_span_m <= _UNIFORM_GRID_MIN_SPAN_M:
            raise StationaryCorrectionError(
                "Corrected z samples have zero or non-finite spatial span."
            )

        z_differences_m: NDArray[np.float64] = np.diff(z_unique_m)
        positive_differences_m: NDArray[np.float64] = z_differences_m[
            np.isfinite(z_differences_m) & (z_differences_m > 0.0)
        ]
        if positive_differences_m.size == 0:
            raise StationaryCorrectionError("No positive z spacing is available.")

        median_spacing_m: float = float(np.median(positive_differences_m))
        if not math.isfinite(median_spacing_m) or median_spacing_m <= 0.0:
            raise StationaryCorrectionError("Median corrected z spacing is invalid.")

        grid_count: int = max(
            _MIN_SAMPLES,
            int(round(z_span_m / median_spacing_m)) + 1,
        )
        uniform_z_m: NDArray[np.float64] = np.linspace(
            float(z_unique_m[0]),
            float(z_unique_m[-1]),
            num=grid_count,
            dtype=np.float64,
        )

        real_grid: NDArray[np.float64] = np.interp(
            uniform_z_m,
            z_unique_m,
            complex_unique.real,
        ).astype(np.float64, copy=False)
        imag_grid: NDArray[np.float64] = np.interp(
            uniform_z_m,
            z_unique_m,
            complex_unique.imag,
        ).astype(np.float64, copy=False)

        complex_grid: NDArray[np.complex128] = (
            real_grid + 1j * imag_grid
        ).astype(np.complex128, copy=False)

        if not np.all(np.isfinite(uniform_z_m)):
            raise StationaryCorrectionError("Uniform z grid contains non-finite values.")
        if not np.all(np.isfinite(complex_grid.real)) or not np.all(
            np.isfinite(complex_grid.imag)
        ):
            raise StationaryCorrectionError(
                "Interpolated complex signal contains non-finite values."
            )

        if uniform_z_m.size >= 3:
            spacings_m: NDArray[np.float64] = np.diff(uniform_z_m)
            if not np.allclose(
                spacings_m,
                float(np.median(spacings_m)),
                rtol=1.0e-9,
                atol=1.0e-9,
            ):
                raise StationaryCorrectionError(
                    "Internal error: generated z grid is not uniformly spaced."
                )

        return uniform_z_m, complex_grid

    @staticmethod
    def _average_duplicate_z_samples(
        z_sorted_m: NDArray[np.float64],
        complex_sorted: NDArray[np.complex128],
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Average complex samples with duplicate or near-duplicate z values."""
        if z_sorted_m.size != complex_sorted.size:
            raise StationaryCorrectionError(
                "z_sorted_m and complex_sorted must have equal length."
            )

        z_groups: list[float] = []
        complex_groups: list[complex] = []

        group_z_values: list[float] = [float(z_sorted_m[0])]
        group_complex_values: list[complex] = [complex(complex_sorted[0])]

        z_scale_m: float = max(1.0, float(np.max(np.abs(z_sorted_m))))
        duplicate_tolerance_m: float = max(
            _DUPLICATE_Z_TOL_M,
            np.finfo(np.float64).eps * z_scale_m * 16.0,
        )

        for index in range(1, z_sorted_m.size):
            z_value_m: float = float(z_sorted_m[index])
            if abs(z_value_m - group_z_values[-1]) <= duplicate_tolerance_m:
                group_z_values.append(z_value_m)
                group_complex_values.append(complex(complex_sorted[index]))
            else:
                z_groups.append(float(np.mean(group_z_values)))
                complex_groups.append(complex(np.mean(group_complex_values)))
                group_z_values = [z_value_m]
                group_complex_values = [complex(complex_sorted[index])]

        z_groups.append(float(np.mean(group_z_values)))
        complex_groups.append(complex(np.mean(group_complex_values)))

        z_unique_m: NDArray[np.float64] = np.asarray(z_groups, dtype=np.float64)
        complex_unique: NDArray[np.complex128] = np.asarray(
            complex_groups,
            dtype=np.complex128,
        )

        if z_unique_m.size >= 2 and np.any(np.diff(z_unique_m) <= 0.0):
            raise StationaryCorrectionError(
                "Duplicate handling failed to produce strictly increasing z samples."
            )

        return z_unique_m, complex_unique

    @staticmethod
    def _validate_window(window: SignalWindow) -> None:
        """Validate a ``SignalWindow`` input."""
        if not isinstance(window, SignalWindow):
            raise TypeError(
                f"window must be a SignalWindow, got {type(window).__name__}."
            )

    @staticmethod
    def _validate_plane(plane: BpPlaneGeometry) -> None:
        """Validate a ``BpPlaneGeometry`` input."""
        if not isinstance(plane, BpPlaneGeometry):
            raise TypeError(
                f"plane must be a BpPlaneGeometry, got {type(plane).__name__}."
            )

    @staticmethod
    def _validate_option(option: int) -> int:
        """Validate stationary-transmitter correction option."""
        try:
            option_value: int = int(option)
        except (TypeError, ValueError) as exc:
            raise ValueError("Stationary correction option must be 1 or 2.") from exc

        if option_value not in {1, 2}:
            raise ValueError(
                f"Stationary correction option must be 1 or 2, got {option!r}."
            )
        return option_value

    @staticmethod
    def _validate_screen_distance_km(screen_distance_km: float) -> float:
        """Validate screen distance in kilometers."""
        distance_km: float = StationaryCorrector._validate_positive_scalar(
            screen_distance_km,
            "screen_distance_km",
        )
        return distance_km

    @staticmethod
    def _unit_axis_array(vector: Vector3, name: str) -> NDArray[np.float64]:
        """Return a finite unit-axis array from ``Vector3``."""
        if not isinstance(vector, Vector3):
            raise TypeError(f"{name} must be Vector3, got {type(vector).__name__}.")

        array: NDArray[np.float64] = vector.to_array()
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise StationaryCorrectionError(f"{name} must contain finite components.")

        norm: float = float(np.linalg.norm(array))
        if not math.isfinite(norm) or norm <= _VECTOR_NORM_EPS:
            raise StationaryCorrectionError(f"{name} is zero, near-zero, or non-finite.")

        return (array / norm).astype(np.float64, copy=False)

    @staticmethod
    def _array_to_vector(array: NDArray[Any], name: str) -> Vector3:
        """Convert a finite length-3 array to ``Vector3``."""
        vector_array: NDArray[np.float64] = np.asarray(array, dtype=np.float64)
        if vector_array.shape != (3,) or not np.all(np.isfinite(vector_array)):
            raise StationaryCorrectionError(f"{name} must be a finite length-3 array.")
        return Vector3(
            x=float(vector_array[0]),
            y=float(vector_array[1]),
            z=float(vector_array[2]),
        )

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
        """Validate that two one-dimensional arrays have equal length."""
        if len(first) != len(second):
            raise ValueError(
                f"{first_name} and {second_name} must have equal length, got "
                f"{len(first)} and {len(second)}."
            )

    @staticmethod
    def _validate_finite_scalar(value: float, name: str) -> float:
        """Validate a finite scalar."""
        try:
            scalar: float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite scalar.") from exc

        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        return scalar

    @classmethod
    def _validate_positive_scalar(cls, value: float, name: str) -> float:
        """Validate a finite, strictly positive scalar."""
        scalar: float = cls._validate_finite_scalar(value, name)
        if scalar <= 0.0:
            raise ValueError(f"{name} must be > 0, got {scalar}.")
        return scalar

    def _configured_int(self, names: tuple[str, ...], default: int) -> int:
        """Read first available integer-like flat config attribute."""
        for name in names:
            if hasattr(self.config, name):
                value: Any = getattr(self.config, name)
                if value is not None:
                    return int(value)
        return int(default)

    def _configured_bool(self, names: tuple[str, ...], default: bool) -> bool:
        """Read first available boolean-like flat config attribute."""
        for name in names:
            if hasattr(self.config, name):
                value: Any = getattr(self.config, name)
                if value is not None:
                    return bool(value)
        return bool(default)


__all__ = ["StationaryCorrector", "StationaryCorrectionError"]
