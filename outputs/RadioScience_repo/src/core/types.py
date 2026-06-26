## src/core/types.py
"""Shared data contracts for COSMIC-2 back-propagation geolocation.

This module defines the lightweight in-memory types used across the
reproduction pipeline. It intentionally contains no data loading, geometry,
signal processing, propagation, quality-control, plotting, or file-writing
logic.

Conventions:
    * ECEF positions are in meters unless a field name says otherwise.
    * ECEF velocities are in meters per second.
    * Back-propagation and geolocation distances use kilometers for fields
      ending in ``_km``.
    * Propagation-grid coordinates use meters for fields ending in ``_m``.
    * Phase is always in radians for fields ending in ``_rad``.
    * SNR is represented in V/V for fields ending in ``_vv``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


_ZERO_NORM_EPS: float = 1.0e-15


def _default_datetime() -> datetime:
    """Return a deterministic UTC epoch default."""
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _empty_float_array() -> NDArray[np.float64]:
    """Return an empty one-dimensional float array."""
    return np.empty(0, dtype=np.float64)


def _empty_complex_array() -> NDArray[np.complex128]:
    """Return an empty one-dimensional complex array."""
    return np.empty(0, dtype=np.complex128)


def _as_1d_float_array(value: Any, field_name: str) -> NDArray[np.float64]:
    """Convert input to a one-dimensional float ndarray."""
    array: NDArray[np.float64] = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional, got {array.ndim}D.")
    return array


def _as_1d_complex_array(value: Any, field_name: str) -> NDArray[np.complex128]:
    """Convert input to a one-dimensional complex ndarray."""
    array: NDArray[np.complex128] = np.asarray(value, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional, got {array.ndim}D.")
    return array


def _validate_equal_lengths(lengths: dict[str, int]) -> None:
    """Validate that all named lengths are equal."""
    if not lengths:
        return

    unique_lengths: set[int] = set(lengths.values())
    if len(unique_lengths) <= 1:
        return

    length_text: str = ", ".join(f"{name}={length}" for name, length in lengths.items())
    raise ValueError(f"Expected equal lengths, got {length_text}.")


def _float_or_nan(value: Any) -> float:
    """Convert a value to float, preserving missing values as NaN when possible."""
    if value is None:
        return math.nan
    return float(value)


def _unit_x() -> "Vector3":
    """Return default Cartesian x unit vector."""
    return Vector3(1.0, 0.0, 0.0)


def _unit_y() -> "Vector3":
    """Return default Cartesian y unit vector."""
    return Vector3(0.0, 1.0, 0.0)


def _unit_z() -> "Vector3":
    """Return default Cartesian z unit vector."""
    return Vector3(0.0, 0.0, 1.0)


@dataclass(slots=True)
class Vector3:
    """Three-dimensional Cartesian vector.

    The unit is context dependent. For example, ``position_m`` fields are ECEF
    meters, ``velocity_mps`` fields are ECEF meters per second, and fields
    ending in ``_unit`` are dimensionless unit vectors.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self) -> None:
        """Normalize scalar field types."""
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)

    def to_array(self) -> NDArray[np.float64]:
        """Return vector components as a NumPy array.

        Returns:
            Array with shape ``(3,)`` containing ``[x, y, z]``.
        """
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def norm(self) -> float:
        """Return the Euclidean norm of the vector."""
        return float(np.linalg.norm(self.to_array()))

    def unit(self) -> "Vector3":
        """Return the unit vector in the same direction.

        Raises:
            ValueError: If the vector norm is zero, non-finite, or numerically
                too small for stable normalization.
        """
        vector_norm: float = self.norm()
        if not math.isfinite(vector_norm) or vector_norm <= _ZERO_NORM_EPS:
            raise ValueError(
                "Cannot normalize a zero, non-finite, or near-zero Vector3."
            )
        return Vector3(
            x=self.x / vector_norm,
            y=self.y / vector_norm,
            z=self.z / vector_norm,
        )


@dataclass(slots=True)
class StateVector:
    """Timestamped spacecraft or transmitter state vector.

    Attributes:
        time: Absolute UTC-like timestamp.
        position_m: ECEF position in meters.
        velocity_mps: ECEF velocity in meters per second.
    """

    time: datetime = field(default_factory=_default_datetime)
    position_m: Vector3 = field(default_factory=Vector3)
    velocity_mps: Vector3 = field(default_factory=Vector3)


@dataclass(slots=True)
class SignalWindow:
    """One high-rate COSMIC-2 scintillation processing interval.

    The ``times`` array is expected to contain seconds relative to
    ``start_time``. Arrays represent a single 10-second processing window in
    normal pipeline use, but this class does not enforce a duration.
    """

    event_id: str = ""
    leo_id: str = ""
    gnss_id: str = ""
    constellation: str = ""
    signal_name: str = ""
    antenna: str = ""
    start_time: datetime = field(default_factory=_default_datetime)
    mid_time: datetime = field(default_factory=_default_datetime)
    end_time: datetime = field(default_factory=_default_datetime)
    sampling_rate_hz: float = 0.0
    times: NDArray[np.float64] = field(default_factory=_empty_float_array)
    phase_rad: NDArray[np.float64] = field(default_factory=_empty_float_array)
    amplitude: NDArray[np.float64] = field(default_factory=_empty_float_array)
    snr_vv: NDArray[np.float64] = field(default_factory=_empty_float_array)
    rx_states: list[StateVector] = field(default_factory=list)
    tx_states: list[StateVector] = field(default_factory=list)
    tangent_height_km: float = math.nan

    def __post_init__(self) -> None:
        """Coerce array fields and perform lightweight consistency checks."""
        self.event_id = str(self.event_id)
        self.leo_id = str(self.leo_id)
        self.gnss_id = str(self.gnss_id)
        self.constellation = str(self.constellation)
        self.signal_name = str(self.signal_name)
        self.antenna = str(self.antenna)
        self.sampling_rate_hz = float(self.sampling_rate_hz)
        self.tangent_height_km = float(self.tangent_height_km)

        if self.sampling_rate_hz < 0.0:
            raise ValueError(
                f"sampling_rate_hz must be non-negative, got {self.sampling_rate_hz}."
            )

        self.times = _as_1d_float_array(self.times, "times")
        self.phase_rad = _as_1d_float_array(self.phase_rad, "phase_rad")
        self.amplitude = _as_1d_float_array(self.amplitude, "amplitude")
        self.snr_vv = _as_1d_float_array(self.snr_vv, "snr_vv")

        sample_lengths: dict[str, int] = {
            "times": len(self.times),
            "phase_rad": len(self.phase_rad),
            "amplitude": len(self.amplitude),
            "snr_vv": len(self.snr_vv),
        }
        _validate_equal_lengths(sample_lengths)

        sample_count: int = len(self.times)
        if self.rx_states and len(self.rx_states) != sample_count:
            raise ValueError(
                "rx_states length must match signal sample count when provided, "
                f"got rx_states={len(self.rx_states)}, samples={sample_count}."
            )
        if self.tx_states and len(self.tx_states) != sample_count:
            raise ValueError(
                "tx_states length must match signal sample count when provided, "
                f"got tx_states={len(self.tx_states)}, samples={sample_count}."
            )

    def copy(self) -> "SignalWindow":
        """Return an independent copy suitable for preprocessing.

        NumPy arrays are copied. State vectors are deep-copied to prevent later
        mutation of orbit states from affecting the original window.
        """
        return SignalWindow(
            event_id=self.event_id,
            leo_id=self.leo_id,
            gnss_id=self.gnss_id,
            constellation=self.constellation,
            signal_name=self.signal_name,
            antenna=self.antenna,
            start_time=self.start_time,
            mid_time=self.mid_time,
            end_time=self.end_time,
            sampling_rate_hz=self.sampling_rate_hz,
            times=self.times.copy(),
            phase_rad=self.phase_rad.copy(),
            amplitude=self.amplitude.copy(),
            snr_vv=self.snr_vv.copy(),
            rx_states=deepcopy(self.rx_states),
            tx_states=deepcopy(self.tx_states),
            tangent_height_km=self.tangent_height_km,
        )


@dataclass(slots=True)
class ScintillationMetrics:
    """Scintillation metrics for one processing window."""

    sigma_phi_rad: float = math.nan
    s4: float = math.nan
    mean_snr_vv: float = math.nan
    passes_threshold: bool = False

    def __post_init__(self) -> None:
        """Normalize scalar field types."""
        self.sigma_phi_rad = float(self.sigma_phi_rad)
        self.s4 = float(self.s4)
        self.mean_snr_vv = float(self.mean_snr_vv)
        self.passes_threshold = bool(self.passes_threshold)


@dataclass(slots=True)
class BpPlaneGeometry:
    """Local 2D back-propagation plane geometry.

    This geometry corresponds to one magnetic-field candidate distance
    ``l_mf_km`` and is used to compute a single BP ``V(L)`` curve.
    """

    l_mf_km: float = math.nan
    los_unit: Vector3 = field(default_factory=_unit_x)
    x_axis: Vector3 = field(default_factory=_unit_x)
    y_axis: Vector3 = field(default_factory=_unit_y)
    z_axis: Vector3 = field(default_factory=_unit_z)
    magnetic_unit: Vector3 = field(default_factory=_unit_z)
    cos_alpha: float = math.nan
    alpha_rad: float = math.nan
    beta_rad: float = math.nan
    candidate_position_m: Vector3 = field(default_factory=Vector3)

    def __post_init__(self) -> None:
        """Normalize scalar field types."""
        self.l_mf_km = float(self.l_mf_km)
        self.cos_alpha = float(self.cos_alpha)
        self.alpha_rad = float(self.alpha_rad)
        self.beta_rad = float(self.beta_rad)


@dataclass(slots=True)
class CorrectedSignal:
    """Corrected complex signal on a uniform transverse grid for FFT BP."""

    screen_distance_km: float = math.nan
    z_m: NDArray[np.float64] = field(default_factory=_empty_float_array)
    complex_signal: NDArray[np.complex128] = field(default_factory=_empty_complex_array)
    amplitude: NDArray[np.float64] = field(default_factory=_empty_float_array)
    phase_rad: NDArray[np.float64] = field(default_factory=_empty_float_array)
    curvature_radius_m: float = math.nan

    def __post_init__(self) -> None:
        """Coerce array fields and perform lightweight consistency checks."""
        self.screen_distance_km = float(self.screen_distance_km)
        self.curvature_radius_m = float(self.curvature_radius_m)

        self.z_m = _as_1d_float_array(self.z_m, "z_m")
        self.complex_signal = _as_1d_complex_array(
            self.complex_signal, "complex_signal"
        )
        self.amplitude = _as_1d_float_array(self.amplitude, "amplitude")
        self.phase_rad = _as_1d_float_array(self.phase_rad, "phase_rad")

        _validate_equal_lengths(
            {
                "z_m": len(self.z_m),
                "complex_signal": len(self.complex_signal),
                "amplitude": len(self.amplitude),
                "phase_rad": len(self.phase_rad),
            }
        )


@dataclass(slots=True)
class BpCurve:
    """Back-propagated amplitude-variance curve for one BP plane.

    The curve stores raw and smoothed values of

        V(L) = <A^2> / <A>^2 - 1

    over the candidate screen-distance grid for a fixed magnetic-field
    candidate distance ``l_mf_km``.
    """

    l_mf_km: float = math.nan
    distances_km: NDArray[np.float64] = field(default_factory=_empty_float_array)
    v_raw: NDArray[np.float64] = field(default_factory=_empty_float_array)
    v_smooth: NDArray[np.float64] = field(default_factory=_empty_float_array)
    l0_km: float = math.nan
    v0: float = math.nan
    l1_km: float = math.nan
    l2_km: float = math.nan
    v1: float = math.nan
    v2: float = math.nan
    q: float = math.nan
    has_valid_minimum: bool = False

    def __post_init__(self) -> None:
        """Coerce array fields and perform lightweight consistency checks."""
        self.l_mf_km = float(self.l_mf_km)
        self.l0_km = float(self.l0_km)
        self.v0 = float(self.v0)
        self.l1_km = float(self.l1_km)
        self.l2_km = float(self.l2_km)
        self.v1 = float(self.v1)
        self.v2 = float(self.v2)
        self.q = float(self.q)
        self.has_valid_minimum = bool(self.has_valid_minimum)

        self.distances_km = _as_1d_float_array(self.distances_km, "distances_km")
        self.v_raw = _as_1d_float_array(self.v_raw, "v_raw")
        self.v_smooth = _as_1d_float_array(self.v_smooth, "v_smooth")

        _validate_equal_lengths(
            {
                "distances_km": len(self.distances_km),
                "v_raw": len(self.v_raw),
                "v_smooth": len(self.v_smooth),
            }
        )


@dataclass(slots=True)
class DCurve:
    """Outer-loop geolocation curve D(L_mf) = L0 - L_mf."""

    l_mf_km: NDArray[np.float64] = field(default_factory=_empty_float_array)
    d_km: NDArray[np.float64] = field(default_factory=_empty_float_array)
    l0_km: NDArray[np.float64] = field(default_factory=_empty_float_array)
    q: NDArray[np.float64] = field(default_factory=_empty_float_array)
    cos_alpha: NDArray[np.float64] = field(default_factory=_empty_float_array)
    zero_crossings_km: list[float] = field(default_factory=list)
    is_multivalued: bool = False

    def __post_init__(self) -> None:
        """Coerce array fields and perform lightweight consistency checks."""
        self.l_mf_km = _as_1d_float_array(self.l_mf_km, "l_mf_km")
        self.d_km = _as_1d_float_array(self.d_km, "d_km")
        self.l0_km = _as_1d_float_array(self.l0_km, "l0_km")
        self.q = _as_1d_float_array(self.q, "q")
        self.cos_alpha = _as_1d_float_array(self.cos_alpha, "cos_alpha")
        self.zero_crossings_km = [float(value) for value in self.zero_crossings_km]
        self.is_multivalued = bool(self.is_multivalued)

        _validate_equal_lengths(
            {
                "l_mf_km": len(self.l_mf_km),
                "d_km": len(self.d_km),
                "l0_km": len(self.l0_km),
                "q": len(self.q),
                "cos_alpha": len(self.cos_alpha),
            }
        )


@dataclass(slots=True)
class GeolocationResult:
    """Final per-window geolocation or rejection record."""

    event_id: str = ""
    leo_id: str = ""
    gnss_id: str = ""
    signal_name: str = ""
    mid_time: datetime = field(default_factory=_default_datetime)
    accepted: bool = False
    rejection_reason: str = ""
    distance_km: float = math.nan
    latitude_deg: float = math.nan
    longitude_deg: float = math.nan
    altitude_km: float = math.nan
    local_time_hr: float = math.nan
    sigma_phi_rad: float = math.nan
    s4: float = math.nan
    mean_snr_vv: float = math.nan
    q: float = math.nan
    cos_alpha: float = math.nan
    d_slope: float = math.nan
    is_multivalued: bool = False

    def __post_init__(self) -> None:
        """Normalize scalar field types."""
        self.event_id = str(self.event_id)
        self.leo_id = str(self.leo_id)
        self.gnss_id = str(self.gnss_id)
        self.signal_name = str(self.signal_name)
        self.accepted = bool(self.accepted)
        self.rejection_reason = str(self.rejection_reason)
        self.distance_km = _float_or_nan(self.distance_km)
        self.latitude_deg = _float_or_nan(self.latitude_deg)
        self.longitude_deg = _float_or_nan(self.longitude_deg)
        self.altitude_km = _float_or_nan(self.altitude_km)
        self.local_time_hr = _float_or_nan(self.local_time_hr)
        self.sigma_phi_rad = _float_or_nan(self.sigma_phi_rad)
        self.s4 = _float_or_nan(self.s4)
        self.mean_snr_vv = _float_or_nan(self.mean_snr_vv)
        self.q = _float_or_nan(self.q)
        self.cos_alpha = _float_or_nan(self.cos_alpha)
        self.d_slope = _float_or_nan(self.d_slope)
        self.is_multivalued = bool(self.is_multivalued)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-Python dictionary suitable for tabular outputs."""
        return {
            "event_id": self.event_id,
            "leo_id": self.leo_id,
            "gnss_id": self.gnss_id,
            "signal_name": self.signal_name,
            "mid_time": self.mid_time,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "distance_km": self.distance_km,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "altitude_km": self.altitude_km,
            "local_time_hr": self.local_time_hr,
            "sigma_phi_rad": self.sigma_phi_rad,
            "s4": self.s4,
            "mean_snr_vv": self.mean_snr_vv,
            "q": self.q,
            "cos_alpha": self.cos_alpha,
            "d_slope": self.d_slope,
            "is_multivalued": self.is_multivalued,
        }


__all__ = [
    "Vector3",
    "StateVector",
    "SignalWindow",
    "ScintillationMetrics",
    "BpPlaneGeometry",
    "CorrectedSignal",
    "BpCurve",
    "DCurve",
    "GeolocationResult",
]
