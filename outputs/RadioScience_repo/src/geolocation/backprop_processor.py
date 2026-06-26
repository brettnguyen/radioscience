"""Core real-data back-propagation engine for COSMIC-2 geolocation.

This module implements the nested-loop back-propagation algorithm described in
Sections 3--4 of the paper:

1. Outer loop over magnetic-field candidate distances ``L_mf``.
2. Build an IGRF-defined 2D BP plane for each ``L_mf``.
3. Inner loop over BP screen distances ``L``.
4. Apply stationary-transmitter/wavefront-curvature correction.
5. Back propagate the corrected complex signal with the FFT plane-wave operator.
6. Compute the normalized BP amplitude variance

       V(L) = <A^2> / <A>^2 - 1

7. Smooth ``V(L)``, find its global local minimum ``L0``, compute ``Q``.
8. Form the decision curve

       D(L_mf) = L0 - L_mf

9. Find zero crossings of ``D`` as candidate geolocation distances.

This module intentionally does not perform pre-BP scintillation filtering,
final geodetic conversion, final acceptance/rejection, plotting, data loading,
or file writing. Those responsibilities belong to other pipeline modules.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config import AppConfig
from src.core import constants
from src.core.types import BpCurve, BpPlaneGeometry, DCurve, SignalWindow
from src.geometry.bp_plane import BpPlaneBuilder
from src.geolocation.qc import QualityController
from src.geolocation.stationary_correction import StationaryCorrector
from src.propagation.fft_propagator import FftPropagator


_LOGGER = logging.getLogger(__name__)

_M_PER_KM: float = 1000.0
_GRID_ENDPOINT_HALF_STEP_FACTOR: float = 0.5
_WAVELENGTH_RELATIVE_TOLERANCE: float = 1.0e-9
_WAVELENGTH_ABSOLUTE_TOLERANCE_M: float = 1.0e-12


class BackpropProcessor:
    """Compute BP ``V(L)`` curves and geolocation decision ``D(L_mf)`` curves.

    Args:
        config: Validated application configuration.
        plane_builder: Builder for IGRF-defined BP planes.
        corrector: Stationary-transmitter and curvature correction engine.
        propagator: FFT plane-wave propagation engine.
        qc: Quality-control and curve-analysis utility.

    Public methods follow the project design:
        * ``process_for_lmf(window, l_mf_km, wavelength_m)``
        * ``compute_v_curve(window, plane, wavelength_m)``
        * ``compute_d_curve(window, wavelength_m)``
    """

    def __init__(
        self,
        config: AppConfig,
        plane_builder: BpPlaneBuilder,
        corrector: StationaryCorrector,
        propagator: FftPropagator,
        qc: QualityController,
    ) -> None:
        """Initialize the real-data BP processor.

        No scientific computation is performed in the constructor.

        Args:
            config: Validated application configuration.
            plane_builder: BP-plane builder.
            corrector: Stationary-transmitter corrector.
            propagator: FFT plane-wave propagator.
            qc: Quality controller.

        Raises:
            TypeError: If any dependency has an unexpected type.
            ValueError: If required grid configuration values are invalid.
        """
        if not isinstance(config, AppConfig):
            raise TypeError(
                f"config must be an AppConfig, got {type(config).__name__}."
            )
        if not isinstance(plane_builder, BpPlaneBuilder):
            raise TypeError(
                "plane_builder must be a BpPlaneBuilder, got "
                f"{type(plane_builder).__name__}."
            )
        if not isinstance(corrector, StationaryCorrector):
            raise TypeError(
                "corrector must be a StationaryCorrector, got "
                f"{type(corrector).__name__}."
            )
        if not isinstance(propagator, FftPropagator):
            raise TypeError(
                "propagator must be a FftPropagator, got "
                f"{type(propagator).__name__}."
            )
        if not isinstance(qc, QualityController):
            raise TypeError(f"qc must be a QualityController, got {type(qc).__name__}.")

        self.config: AppConfig = config
        self.plane_builder: BpPlaneBuilder = plane_builder
        self.corrector: StationaryCorrector = corrector
        self.propagator: FftPropagator = propagator
        self.qc: QualityController = qc

        self.bp_min_distance_km: float = self._positive_finite_float(
            getattr(config, "bp_min_distance_km", constants.DEFAULT_BP_MIN_DISTANCE_KM),
            "bp_min_distance_km",
        )
        self.bp_max_distance_km: float = self._positive_finite_float(
            getattr(config, "bp_max_distance_km", constants.DEFAULT_BP_MAX_DISTANCE_KM),
            "bp_max_distance_km",
        )
        self.bp_step_km: float = self._positive_finite_float(
            getattr(config, "bp_step_km", constants.DEFAULT_BP_STEP_KM),
            "bp_step_km",
        )
        self.mf_step_km: float = self._positive_finite_float(
            getattr(config, "mf_step_km", constants.DEFAULT_MF_STEP_KM),
            "mf_step_km",
        )

        if self.bp_max_distance_km <= self.bp_min_distance_km:
            raise ValueError(
                "bp_max_distance_km must be greater than bp_min_distance_km."
            )

        self.stationary_correction_option: int = self._configured_stationary_option()

        self._bp_distances_km: NDArray[np.float64] = self._distance_grid_km(
            min_km=self.bp_min_distance_km,
            max_km=self.bp_max_distance_km,
            step_km=self.bp_step_km,
            name="BP distance grid",
        )
        self._mf_distances_km: NDArray[np.float64] = self._distance_grid_km(
            min_km=self.bp_min_distance_km,
            max_km=self.bp_max_distance_km,
            step_km=self.mf_step_km,
            name="magnetic-field candidate distance grid",
        )

    def process_for_lmf(
        self,
        window: SignalWindow,
        l_mf_km: float,
        wavelength_m: float,
    ) -> BpCurve:
        """Process one magnetic-field candidate distance ``L_mf``.

        This method builds the BP plane using IGRF geometry and delegates the
        full inner-loop ``V(L)`` calculation to ``compute_v_curve``.

        Args:
            window: Prepared 10-second high-rate signal window.
            l_mf_km: Magnetic-field candidate distance in kilometers.
            wavelength_m: Carrier wavelength in meters. This is validated and
                checked against the injected FFT propagator.

        Returns:
            ``BpCurve`` for the requested BP-plane orientation.

        Raises:
            TypeError: If ``window`` is not a ``SignalWindow``.
            ValueError: If ``l_mf_km`` or ``wavelength_m`` is invalid.
            Exception: Propagates geometry construction errors from
                ``BpPlaneBuilder``. ``compute_d_curve`` catches such errors when
                processing full outer-loop grids.
        """
        self._validate_window(window)
        l_mf_value_km: float = self._positive_finite_float(l_mf_km, "l_mf_km")
        self._validate_wavelength_consistency(wavelength_m)

        plane: BpPlaneGeometry = self.plane_builder.build(window, l_mf_value_km)
        return self.compute_v_curve(
            window=window,
            plane=plane,
            wavelength_m=wavelength_m,
        )

    def compute_v_curve(
        self,
        window: SignalWindow,
        plane: BpPlaneGeometry,
        wavelength_m: float,
    ) -> BpCurve:
        """Compute the full inner-loop BP amplitude-variance curve ``V(L)``.

        For a fixed BP plane orientation, this method evaluates all candidate
        screen distances from the configured paper grid, applies the
        stationary-transmitter correction, back propagates with the FFT
        plane-wave operator, and computes ``V(L)``.

        Args:
            window: Prepared 10-second high-rate signal window.
            plane: BP-plane geometry for one ``L_mf``.
            wavelength_m: Carrier wavelength in meters. The injected propagator
                should have been constructed with the same wavelength.

        Returns:
            ``BpCurve`` containing raw/smoothed ``V(L)``, minimum diagnostics,
            and ``Q``. Failed individual distances are represented by NaN
            values; invalid curves are marked with ``has_valid_minimum=False``.

        Raises:
            TypeError: If ``window`` or ``plane`` has invalid type.
            ValueError: If ``wavelength_m`` is invalid.
        """
        self._validate_window(window)
        self._validate_plane(plane)
        self._validate_wavelength_consistency(wavelength_m)

        distances_km: NDArray[np.float64] = self._bp_distances_km.copy()
        v_raw: NDArray[np.float64] = np.full(
            distances_km.shape,
            np.nan,
            dtype=np.float64,
        )

        for index, screen_distance_km in enumerate(distances_km):
            try:
                corrected_signal = self.corrector.correct(
                    window=window,
                    plane=plane,
                    screen_distance_km=float(screen_distance_km),
                    option=self.stationary_correction_option,
                )

                bp_field: NDArray[np.complex128] = self.propagator.backpropagate_2d(
                    field_z=corrected_signal.complex_signal,
                    z_m=corrected_signal.z_m,
                    distance_m=float(screen_distance_km) * _M_PER_KM,
                )

                v_value: float = self.propagator.amplitude_variance(bp_field)
                if math.isfinite(v_value):
                    v_raw[index] = float(v_value)
            except Exception as exc:
                _LOGGER.debug(
                    "BP distance failed for event=%s, L_mf=%.3f km, "
                    "screen_distance=%.3f km: %s",
                    getattr(window, "event_id", ""),
                    float(plane.l_mf_km),
                    float(screen_distance_km),
                    exc,
                )
                v_raw[index] = math.nan

        v_smooth: NDArray[np.float64] = self._smooth_v_curve_safely(
            distances_km=distances_km,
            v_raw=v_raw,
        )

        l0_km: float = math.nan
        v0: float = math.nan
        l1_km: float = math.nan
        l2_km: float = math.nan
        v1: float = math.nan
        v2: float = math.nan
        q_value: float = math.nan
        has_valid_minimum: bool = False

        try:
            l0_candidate_km, v0_candidate, valid_minimum = (
                self.qc.find_global_local_minimum(
                    distances_km=distances_km,
                    v_smooth=v_smooth,
                )
            )
            has_valid_minimum = bool(valid_minimum)

            if has_valid_minimum:
                l0_km = float(l0_candidate_km)
                v0 = float(v0_candidate)

                l1_candidate_km, l2_candidate_km, v1_candidate, v2_candidate, q_candidate = (
                    self.qc.compute_q(
                        distances_km=distances_km,
                        v_smooth=v_smooth,
                        l0_km=l0_km,
                    )
                )

                l1_km = float(l1_candidate_km)
                l2_km = float(l2_candidate_km)
                v1 = float(v1_candidate)
                v2 = float(v2_candidate)
                q_value = float(q_candidate)

                if not math.isfinite(q_value):
                    has_valid_minimum = False
        except Exception as exc:
            _LOGGER.debug(
                "V-curve minimum/Q analysis failed for event=%s, L_mf=%.3f km: %s",
                getattr(window, "event_id", ""),
                float(plane.l_mf_km),
                exc,
            )
            has_valid_minimum = False
            l0_km = math.nan
            v0 = math.nan
            l1_km = math.nan
            l2_km = math.nan
            v1 = math.nan
            v2 = math.nan
            q_value = math.nan

        if not has_valid_minimum:
            l0_km = math.nan
            v0 = math.nan
            l1_km = math.nan
            l2_km = math.nan
            v1 = math.nan
            v2 = math.nan
            q_value = math.nan

        return BpCurve(
            l_mf_km=float(plane.l_mf_km),
            distances_km=distances_km,
            v_raw=v_raw,
            v_smooth=v_smooth,
            l0_km=l0_km,
            v0=v0,
            l1_km=l1_km,
            l2_km=l2_km,
            v1=v1,
            v2=v2,
            q=q_value,
            has_valid_minimum=has_valid_minimum,
        )

    def compute_d_curve(
        self,
        window: SignalWindow,
        wavelength_m: float,
    ) -> DCurve:
        """Compute the full outer-loop geolocation decision curve ``D(L_mf)``.

        The method loops over the configured magnetic-field candidate grid,
        builds each BP plane once, computes the corresponding ``V(L)`` curve,
        stores ``L0``, ``Q``, and ``cos(alpha)``, then forms

            D(L_mf) = L0 - L_mf

        for valid minima.

        Args:
            window: Prepared 10-second high-rate signal window that has already
                passed pre-BP QC.
            wavelength_m: Carrier wavelength in meters. The injected propagator
                should have been constructed with the same wavelength.

        Returns:
            Completed ``DCurve`` containing all outer-loop diagnostics and all
            valid zero crossings. Multi-valued cases are marked but not rejected
            here.

        Raises:
            TypeError: If ``window`` has invalid type.
            ValueError: If ``wavelength_m`` is invalid.
        """
        self._validate_window(window)
        self._validate_wavelength_consistency(wavelength_m)

        l_mf_grid_km: NDArray[np.float64] = self._mf_distances_km.copy()
        l0_values_km: NDArray[np.float64] = np.full(
            l_mf_grid_km.shape,
            np.nan,
            dtype=np.float64,
        )
        d_values_km: NDArray[np.float64] = np.full(
            l_mf_grid_km.shape,
            np.nan,
            dtype=np.float64,
        )
        q_values: NDArray[np.float64] = np.full(
            l_mf_grid_km.shape,
            np.nan,
            dtype=np.float64,
        )
        cos_alpha_values: NDArray[np.float64] = np.full(
            l_mf_grid_km.shape,
            np.nan,
            dtype=np.float64,
        )

        for index, l_mf_km in enumerate(l_mf_grid_km):
            try:
                plane: BpPlaneGeometry = self.plane_builder.build(
                    window=window,
                    l_mf_km=float(l_mf_km),
                )
                cos_alpha_values[index] = float(plane.cos_alpha)

                curve: BpCurve = self.compute_v_curve(
                    window=window,
                    plane=plane,
                    wavelength_m=wavelength_m,
                )

                l0_values_km[index] = float(curve.l0_km)
                q_values[index] = float(curve.q)

                if curve.has_valid_minimum and math.isfinite(float(curve.l0_km)):
                    d_values_km[index] = float(curve.l0_km) - float(l_mf_km)
            except Exception as exc:
                _LOGGER.debug(
                    "Outer BP loop failed for event=%s, L_mf=%.3f km: %s",
                    getattr(window, "event_id", ""),
                    float(l_mf_km),
                    exc,
                )
                l0_values_km[index] = math.nan
                d_values_km[index] = math.nan
                q_values[index] = math.nan
                cos_alpha_values[index] = math.nan

        d_curve: DCurve = DCurve(
            l_mf_km=l_mf_grid_km,
            d_km=d_values_km,
            l0_km=l0_values_km,
            q=q_values,
            cos_alpha=cos_alpha_values,
            zero_crossings_km=[],
            is_multivalued=False,
        )

        try:
            zero_crossings_km: list[float] = self.qc.find_zero_crossings(d_curve)
        except Exception as exc:
            _LOGGER.debug(
                "D-curve zero-crossing detection failed for event=%s: %s",
                getattr(window, "event_id", ""),
                exc,
            )
            zero_crossings_km = []

        return DCurve(
            l_mf_km=l_mf_grid_km,
            d_km=d_values_km,
            l0_km=l0_values_km,
            q=q_values,
            cos_alpha=cos_alpha_values,
            zero_crossings_km=zero_crossings_km,
            is_multivalued=len(zero_crossings_km) > 1,
        )

    @staticmethod
    def _validate_window(window: SignalWindow) -> None:
        """Validate a signal-window input."""
        if not isinstance(window, SignalWindow):
            raise TypeError(
                f"window must be a SignalWindow, got {type(window).__name__}."
            )

    @staticmethod
    def _validate_plane(plane: BpPlaneGeometry) -> None:
        """Validate a BP-plane input."""
        if not isinstance(plane, BpPlaneGeometry):
            raise TypeError(
                f"plane must be a BpPlaneGeometry, got {type(plane).__name__}."
            )

    @staticmethod
    def _positive_finite_float(value: Any, name: str) -> float:
        """Validate a finite strictly positive scalar."""
        try:
            scalar_value: float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite positive scalar.") from exc

        if not math.isfinite(scalar_value) or scalar_value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}.")

        return scalar_value

    @staticmethod
    def _distance_grid_km(
        min_km: float,
        max_km: float,
        step_km: float,
        name: str,
    ) -> NDArray[np.float64]:
        """Create an inclusive configured paper distance grid.

        Args:
            min_km: Minimum grid distance in kilometers.
            max_km: Maximum grid distance in kilometers.
            step_km: Grid spacing in kilometers.
            name: Human-readable name for errors.

        Returns:
            One-dimensional float64 distance grid including both endpoints.

        Raises:
            ValueError: If grid inputs are invalid.
        """
        min_value_km: float = BackpropProcessor._positive_finite_float(
            min_km,
            f"{name}.min_km",
        )
        max_value_km: float = BackpropProcessor._positive_finite_float(
            max_km,
            f"{name}.max_km",
        )
        step_value_km: float = BackpropProcessor._positive_finite_float(
            step_km,
            f"{name}.step_km",
        )

        if max_value_km <= min_value_km:
            raise ValueError(f"{name}: max_km must be greater than min_km.")

        grid: NDArray[np.float64] = np.arange(
            min_value_km,
            max_value_km + _GRID_ENDPOINT_HALF_STEP_FACTOR * step_value_km,
            step_value_km,
            dtype=np.float64,
        )

        grid = grid[grid <= max_value_km + _GRID_ENDPOINT_HALF_STEP_FACTOR * step_value_km]

        if grid.size == 0:
            raise ValueError(f"{name}: generated grid is empty.")
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"{name}: generated grid contains non-finite values.")

        # Snap numerically close endpoint to the configured value for stable
        # diagnostics and equality checks.
        if abs(float(grid[-1]) - max_value_km) <= _GRID_ENDPOINT_HALF_STEP_FACTOR * step_value_km:
            grid[-1] = max_value_km

        if grid[0] != min_value_km:
            grid[0] = min_value_km

        if grid.size >= 2:
            spacings_km: NDArray[np.float64] = np.diff(grid)
            if not np.all(spacings_km > 0.0):
                raise ValueError(f"{name}: generated grid is not strictly increasing.")
            if not np.allclose(
                spacings_km[:-1] if spacings_km.size > 1 else spacings_km,
                step_value_km,
                rtol=1.0e-10,
                atol=1.0e-10,
            ):
                # The last interval may be affected only if config does not
                # divide the span exactly. AppConfig should prevent that, but
                # fail explicitly if it occurs.
                raise ValueError(
                    f"{name}: configured min/max/step do not form a regular grid."
                )

        return grid

    def _configured_stationary_option(self) -> int:
        """Return stationary-transmitter correction option.

        The paper/configuration default for FFT plane-wave propagation is
        Option 2. The public ``AppConfig`` currently does not expose this field,
        so the method uses a future-compatible flat attribute when present and
        otherwise defaults to ``2``.
        """
        option_value: int = int(
            getattr(
                self.config,
                "stationary_transmitter_correction_option",
                getattr(
                    self.config,
                    "stationary_correction_option",
                    constants.DEFAULT_STATIONARY_CORRECTION_OPTION,
                ),
            )
        )
        if option_value not in {1, 2}:
            raise ValueError(
                "Stationary-transmitter correction option must be 1 or 2, got "
                f"{option_value!r}."
            )
        return option_value

    def _validate_wavelength_consistency(self, wavelength_m: float) -> float:
        """Validate requested wavelength and compare it with the propagator.

        Args:
            wavelength_m: Carrier wavelength in meters supplied by caller.

        Returns:
            Validated wavelength.

        Raises:
            ValueError: If wavelength is not finite and positive.
        """
        wavelength_value_m: float = self._positive_finite_float(
            wavelength_m,
            "wavelength_m",
        )

        propagator_wavelength_m: float | None = getattr(
            self.propagator,
            "wavelength_m",
            None,
        )
        if propagator_wavelength_m is not None:
            propagator_wavelength_value_m: float = float(propagator_wavelength_m)
            if math.isfinite(propagator_wavelength_value_m) and not math.isclose(
                wavelength_value_m,
                propagator_wavelength_value_m,
                rel_tol=_WAVELENGTH_RELATIVE_TOLERANCE,
                abs_tol=_WAVELENGTH_ABSOLUTE_TOLERANCE_M,
            ):
                _LOGGER.warning(
                    "Requested wavelength %.12g m differs from injected "
                    "FftPropagator wavelength %.12g m. The injected propagator "
                    "controls FFT phase evolution.",
                    wavelength_value_m,
                    propagator_wavelength_value_m,
                )

        return wavelength_value_m

    def _smooth_v_curve_safely(
        self,
        distances_km: NDArray[np.float64],
        v_raw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Smooth a V-curve, returning NaNs if smoothing fails."""
        try:
            return self.qc.smooth_v_curve(
                distances_km=distances_km,
                v_raw=v_raw,
            )
        except Exception as exc:
            _LOGGER.debug("V-curve smoothing failed: %s", exc)
            return np.full_like(v_raw, np.nan, dtype=np.float64)


__all__ = ["BackpropProcessor"]
