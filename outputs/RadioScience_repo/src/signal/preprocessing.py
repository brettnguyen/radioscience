"""Signal-domain preprocessing for COSMIC-2 BP geolocation.

This module implements the ``SignalPreprocessor`` class used before
scintillation-index calculation and back propagation. It performs only
signal-domain operations inside each high-rate processing window:

* defensive phase connection via ``numpy.unwrap``;
* slow phase-trend removal;
* amplitude/SNR normalization without suppressing scintillation;
* construction of complex RO signal samples when requested.

The paper states that COSMIC-2 high-rate SNR and connected phase are used for
back propagation in 10-second intervals, but it does not specify the exact phase
detrending method. The provided configuration records this ambiguity and uses a
polynomial detrending default with polynomial order 1. This implementation
therefore defaults to linear polynomial detrending while allowing future
``AppConfig`` fields to override the method/order without changing this module's
public API.

Important limitation:
    Generic data-bit demodulation for GLONASS open-loop observations cannot be
    performed here because this class receives only phase, amplitude, and time
    arrays. If CDAAC files do not already provide connected/demodulated phase,
    the data-loader layer must perform that mission/product-specific operation
    before constructing ``SignalWindow`` objects.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from src.config import AppConfig
from src.core import constants
from src.core.types import SignalWindow


_DEFAULT_HIGH_PASS_ORDER: int = 4
_MIN_HIGH_PASS_SAMPLES_PER_SECTION: int = 3


class SignalPreprocessor:
    """Prepare high-rate phase and amplitude for scintillation BP processing.

    Args:
        config: Application configuration. The current public ``AppConfig``
            exposes the paper-backed defaults needed by this class indirectly
            through project constants. If future config versions expose
            ``phase_detrending_method``, ``phase_detrending_poly_order``, or
            high-pass settings, this class will use them automatically.

    Public methods follow the project design exactly:
        * ``detrend_phase(phase_rad, times)``
        * ``normalize_amplitude(amplitude)``
        * ``make_complex_signal(amplitude, phase_rad)``
        * ``prepare_window(window)``
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the signal preprocessor.

        Args:
            config: Validated application configuration.

        Raises:
            TypeError: If ``config`` is not an ``AppConfig``.
            ValueError: If configured detrending parameters are invalid.
        """
        if not isinstance(config, AppConfig):
            raise TypeError(
                f"config must be an AppConfig, got {type(config).__name__}."
            )

        self.config: AppConfig = config
        self.detrending_method: str = self._configured_string(
            names=(
                "phase_detrending_method",
                "detrending_method",
            ),
            default=constants.DEFAULT_PHASE_DETRENDING_METHOD,
        ).strip().lower()

        self.polynomial_order: int = self._configured_int(
            names=(
                "phase_detrending_poly_order",
                "phase_detrending_polynomial_order",
                "detrending_polynomial_order",
            ),
            default=constants.DEFAULT_PHASE_DETRENDING_POLY_ORDER,
        )
        if self.polynomial_order < 0:
            raise ValueError("Phase detrending polynomial order must be >= 0.")

        self.high_pass_cutoff_hz: float | None = self._configured_optional_float(
            names=(
                "phase_high_pass_cutoff_hz",
                "high_pass_cutoff_hz",
            )
        )
        if (
            self.high_pass_cutoff_hz is not None
            and self.high_pass_cutoff_hz <= 0.0
        ):
            raise ValueError("High-pass cutoff frequency must be positive.")

    def detrend_phase(
        self,
        phase_rad: NDArray[Any],
        times: NDArray[Any],
    ) -> NDArray[np.float64]:
        """Unwrap and detrend phase samples in radians.

        The default detrending is polynomial detrending over the window using a
        centered time coordinate. With the provided configuration this is linear
        detrending over each 10-second interval.

        Invalid phase samples are linearly interpolated before unwrapping and
        detrending when at least one valid sample exists. If all phase samples
        are invalid, an all-NaN array of the same length is returned so
        downstream scintillation QC can reject the window.

        Args:
            phase_rad: One-dimensional phase array in radians.
            times: One-dimensional sample-time array. Numeric values are treated
                as seconds in an arbitrary origin; datetime64 values are
                converted to seconds. Large epoch-like values are centered
                before fitting.

        Returns:
            One-dimensional detrended phase array in radians with the same
            length as ``phase_rad``.

        Raises:
            ValueError: If input arrays are not one-dimensional or have unequal
                lengths.
        """
        phase_array: NDArray[np.float64] = self._as_1d_float_array(
            phase_rad,
            "phase_rad",
        )
        time_seconds: NDArray[np.float64] = self._time_to_centered_seconds(
            times=times,
            expected_length=phase_array.size,
        )

        if phase_array.size == 0:
            return phase_array.copy()

        valid_phase_mask: NDArray[np.bool_] = np.isfinite(phase_array)
        if not np.any(valid_phase_mask):
            return np.full_like(phase_array, np.nan, dtype=np.float64)

        filled_phase: NDArray[np.float64] = self._fill_invalid_linear(
            values=phase_array,
            valid_mask=valid_phase_mask,
        )
        if not np.all(np.isfinite(filled_phase)):
            return filled_phase

        unwrapped_phase: NDArray[np.float64] = np.unwrap(filled_phase)

        method: str = self.detrending_method.replace("-", "_").replace(" ", "_")
        if method in {"none", "no", "disabled", "off"}:
            return unwrapped_phase.astype(np.float64, copy=False)

        if method in {"high_pass", "highpass", "high_pass_filter"}:
            high_passed_phase: NDArray[np.float64] | None = self._maybe_high_pass_phase(
                phase_rad=unwrapped_phase,
                time_seconds=time_seconds,
            )
            if high_passed_phase is not None:
                return high_passed_phase

            # The paper/configuration do not provide a default high-pass cutoff.
            # If a future configuration requests high-pass without a cutoff, use
            # the paper-backed polynomial fallback rather than inventing a
            # cutoff frequency.

        # Treat "linear", "quadratic", "polynomial", and the configuration's
        # "unspecified_in_paper" marker as polynomial detrending.
        if method == "linear":
            polynomial_order: int = 1
        elif method == "quadratic":
            polynomial_order = 2
        else:
            polynomial_order = self.polynomial_order

        return self._polynomial_detrend(
            phase_rad=unwrapped_phase,
            time_seconds=time_seconds,
            polynomial_order=polynomial_order,
        )

    def normalize_amplitude(
        self,
        amplitude: NDArray[Any],
    ) -> NDArray[np.float64]:
        """Normalize amplitude/SNR while preserving relative scintillation.

        The paper states that amplitude is represented by SNR scaled to a 1-Hz
        band and that high-rate SNR is used for BP. Multiplying all amplitudes
        by a constant does not affect the normalized BP variance
        ``V=<A^2>/<A>^2-1``. Therefore this method divides by the finite,
        positive mean amplitude in the window.

        Non-finite and nonpositive samples are treated as invalid and are
        linearly interpolated from valid positive samples when possible. If no
        valid positive samples exist, an all-NaN array is returned so downstream
        metrics/QC can reject the window.

        Args:
            amplitude: One-dimensional field-amplitude or SNR-like array.

        Returns:
            One-dimensional normalized amplitude array with mean approximately
            one over valid samples.

        Raises:
            ValueError: If ``amplitude`` is not one-dimensional.
        """
        amplitude_array: NDArray[np.float64] = self._as_1d_float_array(
            amplitude,
            "amplitude",
        )

        if amplitude_array.size == 0:
            return amplitude_array.copy()

        valid_mask: NDArray[np.bool_] = (
            np.isfinite(amplitude_array) & (amplitude_array > 0.0)
        )
        if not np.any(valid_mask):
            return np.full_like(amplitude_array, np.nan, dtype=np.float64)

        filled_amplitude: NDArray[np.float64] = self._fill_invalid_linear(
            values=amplitude_array,
            valid_mask=valid_mask,
        )

        positive_finite_mask: NDArray[np.bool_] = (
            np.isfinite(filled_amplitude) & (filled_amplitude > 0.0)
        )
        if not np.any(positive_finite_mask):
            return np.full_like(amplitude_array, np.nan, dtype=np.float64)

        mean_amplitude: float = float(np.mean(filled_amplitude[positive_finite_mask]))
        if not math.isfinite(mean_amplitude) or mean_amplitude <= 0.0:
            return np.full_like(amplitude_array, np.nan, dtype=np.float64)

        normalized_amplitude: NDArray[np.float64] = (
            filled_amplitude / mean_amplitude
        ).astype(np.float64, copy=False)

        return normalized_amplitude

    def make_complex_signal(
        self,
        amplitude: NDArray[Any],
        phase_rad: NDArray[Any],
    ) -> NDArray[np.complex128]:
        """Construct complex RO signal samples ``u = A * exp(i*phi)``.

        Args:
            amplitude: One-dimensional amplitude array.
            phase_rad: One-dimensional phase array in radians.

        Returns:
            Complex signal array with dtype ``complex128``.

        Raises:
            ValueError: If arrays are not one-dimensional or have unequal
                lengths.
        """
        amplitude_array: NDArray[np.float64] = self._as_1d_float_array(
            amplitude,
            "amplitude",
        )
        phase_array: NDArray[np.float64] = self._as_1d_float_array(
            phase_rad,
            "phase_rad",
        )
        self._require_same_length(
            first=amplitude_array,
            first_name="amplitude",
            second=phase_array,
            second_name="phase_rad",
        )

        if amplitude_array.size == 0:
            return np.empty(0, dtype=np.complex128)

        valid_amplitude_mask: NDArray[np.bool_] = (
            np.isfinite(amplitude_array) & (amplitude_array >= 0.0)
        )
        if np.any(valid_amplitude_mask):
            safe_amplitude: NDArray[np.float64] = self._fill_invalid_linear(
                values=amplitude_array,
                valid_mask=valid_amplitude_mask,
            )
            safe_amplitude = np.where(
                np.isfinite(safe_amplitude) & (safe_amplitude >= 0.0),
                safe_amplitude,
                0.0,
            )
        else:
            safe_amplitude = np.zeros_like(amplitude_array, dtype=np.float64)

        valid_phase_mask: NDArray[np.bool_] = np.isfinite(phase_array)
        if np.any(valid_phase_mask):
            safe_phase: NDArray[np.float64] = self._fill_invalid_linear(
                values=phase_array,
                valid_mask=valid_phase_mask,
            )
            safe_phase = np.where(np.isfinite(safe_phase), safe_phase, 0.0)
        else:
            safe_phase = np.zeros_like(phase_array, dtype=np.float64)

        complex_signal: NDArray[np.complex128] = (
            safe_amplitude * np.exp(1j * safe_phase)
        ).astype(np.complex128, copy=False)

        return complex_signal

    def prepare_window(self, window: SignalWindow) -> SignalWindow:
        """Return a preprocessed copy of a signal window.

        The input ``SignalWindow`` is not mutated. The returned copy contains:

        * detrended phase in ``phase_rad``;
        * normalized amplitude in ``amplitude``;
        * original ``snr_vv`` preserved for diagnostics;
        * all geometry, metadata, and orbit states preserved.

        Amplitude source selection follows the paper's statement that SNR is the
        amplitude representation: use ``window.amplitude`` if it contains any
        valid positive values, otherwise fall back to ``window.snr_vv``.

        Args:
            window: Signal window to preprocess.

        Returns:
            Copied and preprocessed ``SignalWindow``.

        Raises:
            TypeError: If ``window`` is not a ``SignalWindow``.
            ValueError: If core sample arrays have inconsistent lengths.
        """
        if not isinstance(window, SignalWindow):
            raise TypeError(
                f"window must be a SignalWindow, got {type(window).__name__}."
            )

        prepared_window: SignalWindow = window.copy()

        time_array: NDArray[np.float64] = self._as_1d_float_array(
            prepared_window.times,
            "window.times",
        )
        phase_array: NDArray[np.float64] = self._as_1d_float_array(
            prepared_window.phase_rad,
            "window.phase_rad",
        )
        amplitude_array: NDArray[np.float64] = self._as_1d_float_array(
            prepared_window.amplitude,
            "window.amplitude",
        )
        snr_array: NDArray[np.float64] = self._as_1d_float_array(
            prepared_window.snr_vv,
            "window.snr_vv",
        )

        self._require_same_length(
            first=time_array,
            first_name="window.times",
            second=phase_array,
            second_name="window.phase_rad",
        )
        self._require_same_length(
            first=time_array,
            first_name="window.times",
            second=amplitude_array,
            second_name="window.amplitude",
        )
        self._require_same_length(
            first=time_array,
            first_name="window.times",
            second=snr_array,
            second_name="window.snr_vv",
        )

        selected_amplitude: NDArray[np.float64] = self._select_amplitude_source(
            amplitude=amplitude_array,
            snr_vv=snr_array,
        )

        prepared_window.phase_rad = self.detrend_phase(
            phase_rad=phase_array,
            times=time_array,
        )
        prepared_window.amplitude = self.normalize_amplitude(selected_amplitude)

        # Preserve the original SNR diagnostic array. ``SignalWindow.copy()``
        # already copied it, but assign the validated copy to make the intended
        # preservation explicit.
        prepared_window.snr_vv = snr_array.copy()
        prepared_window.times = time_array.copy()

        return prepared_window

    @staticmethod
    def _as_1d_float_array(values: NDArray[Any] | Any, name: str) -> NDArray[np.float64]:
        """Convert input to a one-dimensional float array.

        Args:
            values: Array-like input.
            name: Field name for error messages.

        Returns:
            Float64 one-dimensional array.

        Raises:
            ValueError: If conversion fails or the result is not 1D.
        """
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
        """Validate two arrays have the same length."""
        if len(first) != len(second):
            raise ValueError(
                f"{first_name} and {second_name} must have equal length, got "
                f"{len(first)} and {len(second)}."
            )

    @staticmethod
    def _configured_string(names: tuple[str, ...], default: str) -> str:
        """Placeholder for class-level default access.

        This static method is intentionally not used because it cannot access
        ``self.config``. It remains private and unused only to keep helper
        naming clear in static analysis contexts.
        """
        del names
        return str(default)

    def _configured_int(self, names: tuple[str, ...], default: int) -> int:
        """Read the first available integer-like config attribute.

        Args:
            names: Candidate attribute names on ``self.config``.
            default: Default value when no attribute exists.

        Returns:
            Integer configuration value.
        """
        for name in names:
            if hasattr(self.config, name):
                value: Any = getattr(self.config, name)
                if value is not None:
                    return int(value)
        return int(default)

    def _configured_optional_float(self, names: tuple[str, ...]) -> float | None:
        """Read the first available optional float-like config attribute."""
        for name in names:
            if hasattr(self.config, name):
                value: Any = getattr(self.config, name)
                if value is not None:
                    return float(value)
        return None

    def _configured_string(self, names: tuple[str, ...], default: str) -> str:
        """Read the first available string-like config attribute.

        Args:
            names: Candidate attribute names on ``self.config``.
            default: Default value when no attribute exists.

        Returns:
            String configuration value.
        """
        for name in names:
            if hasattr(self.config, name):
                value: Any = getattr(self.config, name)
                if value is not None:
                    return str(value)
        return str(default)

    @staticmethod
    def _time_to_centered_seconds(
        times: NDArray[Any] | Any,
        expected_length: int,
    ) -> NDArray[np.float64]:
        """Convert sample times to a centered seconds coordinate.

        Args:
            times: Numeric or datetime64 sample times.
            expected_length: Required output length.

        Returns:
            Centered time coordinate in seconds. If supplied times are invalid
            or degenerate, centered sample indices are returned.

        Raises:
            ValueError: If ``times`` is not one-dimensional or has incorrect
                length.
        """
        raw_times: NDArray[Any] = np.asarray(times)

        if raw_times.ndim != 1:
            raise ValueError(f"times must be one-dimensional, got {raw_times.ndim}D.")
        if raw_times.size != expected_length:
            raise ValueError(
                f"times and phase_rad must have equal length, got "
                f"{raw_times.size} and {expected_length}."
            )

        if expected_length == 0:
            return np.empty(0, dtype=np.float64)

        if np.issubdtype(raw_times.dtype, np.datetime64):
            time_values: NDArray[np.float64] = (
                raw_times.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
                * 1.0e-9
            )
        else:
            try:
                time_values = raw_times.astype(np.float64, copy=False)
            except (TypeError, ValueError):
                time_values = np.arange(expected_length, dtype=np.float64)

        if (
            time_values.ndim != 1
            or time_values.size != expected_length
            or not np.all(np.isfinite(time_values))
        ):
            time_values = np.arange(expected_length, dtype=np.float64)

        if expected_length == 1:
            return np.zeros(1, dtype=np.float64)

        centered_time: NDArray[np.float64] = (
            time_values - float(np.mean(time_values))
        ).astype(np.float64, copy=False)

        if not np.all(np.isfinite(centered_time)) or float(np.ptp(centered_time)) <= 0.0:
            sample_index: NDArray[np.float64] = np.arange(expected_length, dtype=np.float64)
            centered_time = sample_index - float(np.mean(sample_index))

        return centered_time

    @staticmethod
    def _fill_invalid_linear(
        values: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Fill invalid samples by linear interpolation over sample index.

        Args:
            values: One-dimensional value array.
            valid_mask: Boolean mask identifying samples suitable for use as
                interpolation support.

        Returns:
            Filled array. If no valid samples exist, all values are NaN. If one
            valid sample exists, the entire array is filled with that sample.
        """
        value_array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
        mask_array: NDArray[np.bool_] = np.asarray(valid_mask, dtype=np.bool_)

        if value_array.ndim != 1 or mask_array.ndim != 1:
            raise ValueError("values and valid_mask must be one-dimensional.")
        if value_array.size != mask_array.size:
            raise ValueError("values and valid_mask must have equal lengths.")

        if value_array.size == 0:
            return value_array.copy()

        valid_indices: NDArray[np.int64] = np.flatnonzero(mask_array)
        if valid_indices.size == 0:
            return np.full_like(value_array, np.nan, dtype=np.float64)

        if valid_indices.size == 1:
            single_value: float = float(value_array[valid_indices[0]])
            return np.full_like(value_array, single_value, dtype=np.float64)

        all_indices: NDArray[np.int64] = np.arange(value_array.size, dtype=np.int64)
        filled_values: NDArray[np.float64] = np.interp(
            all_indices.astype(np.float64),
            valid_indices.astype(np.float64),
            value_array[valid_indices].astype(np.float64),
        ).astype(np.float64, copy=False)

        return filled_values

    def _polynomial_detrend(
        self,
        phase_rad: NDArray[np.float64],
        time_seconds: NDArray[np.float64],
        polynomial_order: int,
    ) -> NDArray[np.float64]:
        """Remove a polynomial trend from unwrapped phase.

        Args:
            phase_rad: Finite unwrapped phase in radians.
            time_seconds: Centered time coordinate in seconds.
            polynomial_order: Requested polynomial degree.

        Returns:
            Phase residual after subtracting the fitted trend.
        """
        phase_array: NDArray[np.float64] = np.asarray(phase_rad, dtype=np.float64)
        time_array: NDArray[np.float64] = np.asarray(time_seconds, dtype=np.float64)

        self._require_same_length(
            first=phase_array,
            first_name="phase_rad",
            second=time_array,
            second_name="time_seconds",
        )

        if phase_array.size == 0:
            return phase_array.copy()

        fit_mask: NDArray[np.bool_] = np.isfinite(phase_array) & np.isfinite(time_array)
        valid_count: int = int(np.count_nonzero(fit_mask))
        if valid_count == 0:
            return np.full_like(phase_array, np.nan, dtype=np.float64)

        if valid_count == 1:
            single_mean: float = float(np.mean(phase_array[fit_mask]))
            return (phase_array - single_mean).astype(np.float64, copy=False)

        unique_time_count: int = int(np.unique(time_array[fit_mask]).size)
        usable_order: int = min(
            int(max(0, polynomial_order)),
            valid_count - 1,
            unique_time_count - 1,
        )

        if usable_order <= 0:
            trend: NDArray[np.float64] = np.full_like(
                phase_array,
                float(np.mean(phase_array[fit_mask])),
                dtype=np.float64,
            )
            return (phase_array - trend).astype(np.float64, copy=False)

        try:
            coefficients: NDArray[np.float64] = np.polyfit(
                time_array[fit_mask],
                phase_array[fit_mask],
                deg=usable_order,
            )
            trend = np.polyval(coefficients, time_array).astype(np.float64, copy=False)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            trend = np.full_like(
                phase_array,
                float(np.mean(phase_array[fit_mask])),
                dtype=np.float64,
            )

        detrended_phase: NDArray[np.float64] = (
            phase_array - trend
        ).astype(np.float64, copy=False)

        return detrended_phase

    def _maybe_high_pass_phase(
        self,
        phase_rad: NDArray[np.float64],
        time_seconds: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Apply optional zero-phase high-pass filtering when fully configured.

        The paper does not specify a cutoff frequency, so this method returns
        ``None`` unless a positive cutoff has been supplied through a future
        configuration field.

        Args:
            phase_rad: Unwrapped phase in radians.
            time_seconds: Centered sample times in seconds.

        Returns:
            High-pass filtered phase, or ``None`` if filtering cannot be applied
            without inventing unspecified settings.
        """
        if self.high_pass_cutoff_hz is None:
            return None

        phase_array: NDArray[np.float64] = np.asarray(phase_rad, dtype=np.float64)
        time_array: NDArray[np.float64] = np.asarray(time_seconds, dtype=np.float64)

        if phase_array.size < 2 or time_array.size != phase_array.size:
            return None
        if not np.all(np.isfinite(phase_array)) or not np.all(np.isfinite(time_array)):
            return None

        sampling_rate_hz: float | None = self._estimate_sampling_rate_hz(time_array)
        if sampling_rate_hz is None:
            return None

        nyquist_hz: float = 0.5 * sampling_rate_hz
        cutoff_hz: float = float(self.high_pass_cutoff_hz)
        if cutoff_hz <= 0.0 or cutoff_hz >= nyquist_hz:
            return None

        normalized_cutoff: float = cutoff_hz / nyquist_hz
        try:
            sos: NDArray[np.float64] = scipy_signal.butter(
                N=_DEFAULT_HIGH_PASS_ORDER,
                Wn=normalized_cutoff,
                btype="highpass",
                output="sos",
            )
        except ValueError:
            return None

        minimum_samples: int = max(
            2,
            _MIN_HIGH_PASS_SAMPLES_PER_SECTION * int(sos.shape[0]),
        )
        if phase_array.size <= minimum_samples:
            return None

        try:
            filtered_phase: NDArray[np.float64] = scipy_signal.sosfiltfilt(
                sos,
                phase_array,
            ).astype(np.float64, copy=False)
        except ValueError:
            return None

        if not np.all(np.isfinite(filtered_phase)):
            return None

        return filtered_phase

    @staticmethod
    def _estimate_sampling_rate_hz(time_seconds: NDArray[np.float64]) -> float | None:
        """Estimate sampling frequency from centered sample times."""
        time_array: NDArray[np.float64] = np.asarray(time_seconds, dtype=np.float64)
        if time_array.ndim != 1 or time_array.size < 2:
            return None

        sorted_time: NDArray[np.float64] = np.sort(time_array)
        differences: NDArray[np.float64] = np.diff(sorted_time)
        positive_differences: NDArray[np.float64] = differences[
            np.isfinite(differences) & (differences > 0.0)
        ]

        if positive_differences.size == 0:
            return None

        median_dt_s: float = float(np.median(positive_differences))
        if not math.isfinite(median_dt_s) or median_dt_s <= 0.0:
            return None

        sampling_rate_hz: float = 1.0 / median_dt_s
        if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
            return None

        return sampling_rate_hz

    @staticmethod
    def _select_amplitude_source(
        amplitude: NDArray[np.float64],
        snr_vv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Select amplitude source for preprocessing.

        Args:
            amplitude: Window amplitude array.
            snr_vv: Window SNR array.

        Returns:
            Copy of the preferred amplitude source.
        """
        amplitude_mask: NDArray[np.bool_] = np.isfinite(amplitude) & (amplitude > 0.0)
        if amplitude.size > 0 and np.any(amplitude_mask):
            return amplitude.copy()

        snr_mask: NDArray[np.bool_] = np.isfinite(snr_vv) & (snr_vv > 0.0)
        if snr_vv.size > 0 and np.any(snr_mask):
            return snr_vv.copy()

        if amplitude.size > 0:
            return amplitude.copy()

        return snr_vv.copy()


__all__ = ["SignalPreprocessor"]
