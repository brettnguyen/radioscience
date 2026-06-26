"""Synthetic validation experiments for COSMIC-2 BP geolocation.

This module implements the controlled numerical experiments used to validate
the back-propagation method described in the paper:

    "Geolocation of the Ionospheric Irregularities in the Equatorial F Layer
    by Back Propagation of COSMIC-2 Radio Occultation Signals"

The experiments in this file are synthetic method checks, not real COSMIC-2
processing. They use:

* thin random phase screens,
* FFT plane-wave forward propagation,
* FFT coordinate-representation back propagation,
* normalized BP amplitude variance,

      V(L) = <A^2> / <A>^2 - 1,

as the localization metric.

The public ``SyntheticExperiments`` API follows the project design exactly:

    SyntheticExperiments(config, generator, propagator)
    SyntheticExperiments.run_all()
    SyntheticExperiments.single_phase_screen()
    SyntheticExperiments.outer_scale_estimation()
    SyntheticExperiments.orientation_error()
    SyntheticExperiments.two_screens_aligned()
    SyntheticExperiments.two_screens_misaligned()
    SyntheticExperiments.thermal_noise()
    SyntheticExperiments.multivalued_geometry()

This module deliberately performs no plotting and no file I/O. It returns rich
Python dictionaries containing arrays, scalar diagnostics, configuration notes,
and success flags. Persistence and visualization are handled by
``ResultWriter`` and ``Plotter`` outside this module.

Configuration note:
    The current ``AppConfig`` interface exposes the BP distance grid and random
    seed, while most synthetic-experiment parameters are represented in
    ``src.core.constants`` from the paper/config.yaml values. This module uses
    ``AppConfig`` when fields exist and otherwise uses those shared constants.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config import AppConfig
from src.core import constants
from src.propagation.fft_propagator import FftPropagator
from src.propagation.phase_screen import PhaseScreenGenerator


_M_PER_KM: float = 1000.0
_MIN_GRID_SAMPLES: int = 8
_SMALL_POSITIVE: float = 1.0e-15
_DEFAULT_MISALIGNED_GRID_SAMPLES: int = 257


class SyntheticExperiments:
    """Run synthetic BP validation experiments from the paper.

    Args:
        config: Application configuration.
        generator: Injected reproducible phase-screen generator.
        propagator: Injected FFT plane-wave propagator.

    The injected generator and propagator are stored and reused; this class does
    not instantiate replacements, preserving the dependency-injection design.
    """

    def __init__(
        self,
        config: AppConfig,
        generator: PhaseScreenGenerator,
        propagator: FftPropagator,
    ) -> None:
        """Initialize synthetic experiment runner.

        Args:
            config: Validated application configuration.
            generator: Phase-screen generator.
            propagator: FFT propagator.

        Raises:
            TypeError: If any dependency has an invalid type.
        """
        if not isinstance(config, AppConfig):
            raise TypeError(
                f"config must be an AppConfig, got {type(config).__name__}."
            )
        if not isinstance(generator, PhaseScreenGenerator):
            raise TypeError(
                "generator must be a PhaseScreenGenerator, got "
                f"{type(generator).__name__}."
            )
        if not isinstance(propagator, FftPropagator):
            raise TypeError(
                "propagator must be an FftPropagator, got "
                f"{type(propagator).__name__}."
            )

        self.config: AppConfig = config
        self.generator: PhaseScreenGenerator = generator
        self.propagator: FftPropagator = propagator
        self.random_seed: int = int(
            getattr(config, "random_seed", constants.DEFAULT_RANDOM_SEED)
        )
        self._noise_rng: np.random.Generator = np.random.default_rng(
            self.random_seed + 10_000
        )

    def run_all(self) -> dict[str, Any]:
        """Run all synthetic validation experiments.

        Returns:
            Dictionary keyed by experiment name. Each sub-dictionary contains
            input parameters, arrays, diagnostics, success flags, and notes.
        """
        return {
            "single_phase_screen": self.single_phase_screen(),
            "outer_scale_estimation": self.outer_scale_estimation(),
            "orientation_error": self.orientation_error(),
            "two_screens_aligned": self.two_screens_aligned(),
            "two_screens_misaligned": self.two_screens_misaligned(),
            "thermal_noise": self.thermal_noise(),
            "multivalued_geometry": self.multivalued_geometry(),
        }

    def single_phase_screen(self) -> dict[str, Any]:
        """Validate that BP V(L) minimum recovers a single phase-screen distance.

        Returns:
            Dictionary containing V(L), estimated distance, and success flag.
        """
        p: float = self._synthetic_p()
        outer_scale_km: float = self._synthetic_outer_scale_km()
        sigma_phi_rad: float = self._observed_sigma_phi_rad()
        true_distance_km: float = self._observed_screen_distance_km()
        wavelength_m: float = self._wavelength_m()

        z_m: NDArray[np.float64] = self._make_z_grid_1d()
        phase_screen_rad: NDArray[np.float64] = self.generator.generate_1d(
            z_m=z_m,
            p=p,
            outer_scale_km=outer_scale_km,
            sigma_phi_rad=sigma_phi_rad,
            wavelength_m=wavelength_m,
        )
        screen_field: NDArray[np.complex128] = np.exp(1j * phase_screen_rad).astype(
            np.complex128,
            copy=False,
        )

        receiver_field: NDArray[np.complex128] = self.propagator.propagate_2d(
            field_z=screen_field,
            z_m=z_m,
            distance_m=true_distance_km * _M_PER_KM,
        )

        distances_km, v_curve = self._compute_v_curve_1d(
            receiver_field=receiver_field,
            z_m=z_m,
        )
        estimated_distance_km, minimum_value = self._minimum_distance(
            distances_km=distances_km,
            v_curve=v_curve,
        )
        distance_error_km: float = estimated_distance_km - true_distance_km
        success: bool = (
            math.isfinite(estimated_distance_km)
            and abs(distance_error_km) <= 0.5 * self._bp_step_km() + 1.0e-9
        )

        return {
            "status": "completed",
            "success": success,
            "random_seed": self.random_seed,
            "parameters": {
                "p": p,
                "outer_scale_km": outer_scale_km,
                "sigma_phi_rad": sigma_phi_rad,
                "true_distance_km": true_distance_km,
                "wavelength_signal": constants.DEFAULT_SIGNAL,
                "wavelength_m": wavelength_m,
                "z_interval_km": self._modeled_spatial_interval_km(),
                "z_step_m": self._internal_grid_step_m(),
                "bp_min_distance_km": self._bp_min_km(),
                "bp_max_distance_km": self._bp_max_km(),
                "bp_step_km": self._bp_step_km(),
            },
            "distances_km": distances_km,
            "v_curve": v_curve,
            "estimated_distance_km": estimated_distance_km,
            "minimum_v": minimum_value,
            "distance_error_km": distance_error_km,
            "paper_alignment_note": (
                "For a single thin phase screen, the BP amplitude-variance "
                "minimum should occur at the true screen distance."
            ),
        }

    def outer_scale_estimation(self) -> dict[str, Any]:
        """Evaluate configured outer-scale cases for sigma_phi/S4 consistency.

        The full Figure 3 sweep requires outer-scale and distance arrays not
        exposed by the current AppConfig. This method therefore evaluates the
        configured target distance and the two paper/configuration outer scales:
        the fitted 9.7 km value and the adopted 10 km value.

        Returns:
            Dictionary with per-case S4 and sigma_phi/S4 diagnostics.
        """
        p: float = self._synthetic_p()
        sigma_phi_rad: float = self._observed_sigma_phi_rad()
        target_ratio_rad: float = constants.DEFAULT_OBSERVED_SIGMA_PHI_OVER_S4_RAD
        target_distance_km: float = self._observed_screen_distance_km()
        configured_outer_scale_km: float = self._synthetic_outer_scale_km()
        estimated_outer_scale_km: float = constants.DEFAULT_ESTIMATED_OUTER_SCALE_KM
        wavelength_m: float = self._wavelength_m()
        z_m: NDArray[np.float64] = self._make_z_grid_1d()

        outer_scale_cases_km: list[float] = self._unique_floats(
            [estimated_outer_scale_km, configured_outer_scale_km]
        )

        cases: list[dict[str, Any]] = []
        for outer_scale_km in outer_scale_cases_km:
            phase_screen_rad: NDArray[np.float64] = self.generator.generate_1d(
                z_m=z_m,
                p=p,
                outer_scale_km=outer_scale_km,
                sigma_phi_rad=sigma_phi_rad,
                wavelength_m=wavelength_m,
            )
            screen_field: NDArray[np.complex128] = np.exp(
                1j * phase_screen_rad
            ).astype(np.complex128, copy=False)
            receiver_field: NDArray[np.complex128] = self.propagator.propagate_2d(
                field_z=screen_field,
                z_m=z_m,
                distance_m=target_distance_km * _M_PER_KM,
            )
            s4: float = self._compute_s4_from_field(receiver_field)
            ratio: float = sigma_phi_rad / s4 if s4 > 0.0 else math.nan

            cases.append(
                {
                    "outer_scale_km": float(outer_scale_km),
                    "distance_km": target_distance_km,
                    "sigma_phi_rad": sigma_phi_rad,
                    "computed_s4": s4,
                    "computed_sigma_phi_over_s4_rad": ratio,
                    "ratio_error_from_target_rad": ratio - target_ratio_rad
                    if math.isfinite(ratio)
                    else math.nan,
                }
            )

        return {
            "status": "completed_partial_configured_cases",
            "success": True,
            "random_seed": self.random_seed,
            "parameters": {
                "p": p,
                "target_ratio_rad": target_ratio_rad,
                "target_distance_km": target_distance_km,
                "configured_outer_scale_km": configured_outer_scale_km,
                "estimated_outer_scale_km": estimated_outer_scale_km,
                "sigma_phi_rad": sigma_phi_rad,
                "observed_s4": constants.DEFAULT_OBSERVED_S4,
                "wavelength_signal": constants.DEFAULT_SIGNAL,
                "wavelength_m": wavelength_m,
            },
            "cases": cases,
            "configuration_warning": (
                "The current AppConfig does not expose full l0/L sweep arrays "
                "for reproducing the complete Figure 3 family. This result "
                "evaluates only the configured target distance and the "
                "paper/configuration outer scales 9.7 km and 10 km."
            ),
        }

    def orientation_error(self) -> dict[str, Any]:
        """Reproduce paper Equation (3) and Table 1 orientation-error values.

        Returns:
            Dictionary containing the analytic table of Delta L values.
        """
        delta_alpha_deg: float = constants.DEFAULT_ORIENTATION_ERROR_DELTA_ALPHA_DEG
        alpha_cases_deg: tuple[float, ...] = (
            constants.DEFAULT_ORIENTATION_ERROR_ALPHA_CASES_DEG
        )
        distance_cases_km: tuple[float, ...] = (
            constants.DEFAULT_ORIENTATION_ERROR_DISTANCE_CASES_KM
        )

        rows: list[dict[str, float]] = []
        for alpha_deg in alpha_cases_deg:
            for distance_km in distance_cases_km:
                delta_l_km: float = constants.orientation_error_delta_l_km(
                    distance_km=distance_km,
                    alpha_deg=alpha_deg,
                    delta_alpha_deg=delta_alpha_deg,
                )
                rows.append(
                    {
                        "alpha_deg": float(alpha_deg),
                        "distance_km": float(distance_km),
                        "delta_l_km": float(delta_l_km),
                        "delta_l_km_rounded": float(round(delta_l_km)),
                    }
                )

        return {
            "status": "completed",
            "success": True,
            "random_seed": self.random_seed,
            "delta_alpha_deg": delta_alpha_deg,
            "delta_alpha_rad": math.radians(delta_alpha_deg),
            "paper_formula": "DeltaL = 2 * L * tan(alpha) * DeltaAlpha",
            "table": rows,
            "expected_table_note": (
                "Rounded values should approximately reproduce Table 1: "
                "6/21/78 km at 1000 km, 17/63/235 km at 3000 km, and "
                "28/105/392 km at 5000 km for alpha=15/45/75 deg."
            ),
        }

    def two_screens_aligned(self) -> dict[str, Any]:
        """Validate BP behavior for two aligned phase screens.

        Returns:
            Dictionary with equal-strength and unequal-strength two-screen
            V(L) curves and interpretations.
        """
        p: float = self._synthetic_p()
        outer_scale_km: float = self._synthetic_outer_scale_km()
        sigma_phi_base_rad: float = self._observed_sigma_phi_rad()
        separation_km: float = constants.DEFAULT_TWO_SCREEN_SEPARATION_KM
        ratio: float = constants.DEFAULT_TWO_SCREEN_SIGMA_RATIO
        wavelength_m: float = self._wavelength_m()
        closer_screen_distance_km: float = self._observed_screen_distance_km()
        farther_screen_distance_km: float = closer_screen_distance_km + separation_km

        self._validate_screen_distances_inside_grid(
            [closer_screen_distance_km, farther_screen_distance_km],
            "two_screens_aligned",
        )

        z_m: NDArray[np.float64] = self._make_z_grid_1d()

        equal_receiver_field: NDArray[np.complex128] = self._forward_two_aligned_screens(
            z_m=z_m,
            closer_screen_distance_km=closer_screen_distance_km,
            separation_km=separation_km,
            p=p,
            outer_scale_km=outer_scale_km,
            closer_sigma_phi_rad=sigma_phi_base_rad,
            farther_sigma_phi_rad=sigma_phi_base_rad,
            wavelength_m=wavelength_m,
        )
        equal_distances_km, equal_v_curve = self._compute_v_curve_1d(
            receiver_field=equal_receiver_field,
            z_m=z_m,
        )
        equal_min_km, equal_min_v = self._minimum_distance(
            equal_distances_km,
            equal_v_curve,
        )

        closer_stronger_field: NDArray[np.complex128] = (
            self._forward_two_aligned_screens(
                z_m=z_m,
                closer_screen_distance_km=closer_screen_distance_km,
                separation_km=separation_km,
                p=p,
                outer_scale_km=outer_scale_km,
                closer_sigma_phi_rad=ratio * sigma_phi_base_rad,
                farther_sigma_phi_rad=sigma_phi_base_rad,
                wavelength_m=wavelength_m,
            )
        )
        closer_stronger_distances_km, closer_stronger_v_curve = (
            self._compute_v_curve_1d(closer_stronger_field, z_m)
        )
        closer_stronger_min_km, closer_stronger_min_v = self._minimum_distance(
            closer_stronger_distances_km,
            closer_stronger_v_curve,
        )

        farther_stronger_field: NDArray[np.complex128] = (
            self._forward_two_aligned_screens(
                z_m=z_m,
                closer_screen_distance_km=closer_screen_distance_km,
                separation_km=separation_km,
                p=p,
                outer_scale_km=outer_scale_km,
                closer_sigma_phi_rad=sigma_phi_base_rad,
                farther_sigma_phi_rad=ratio * sigma_phi_base_rad,
                wavelength_m=wavelength_m,
            )
        )
        farther_stronger_distances_km, farther_stronger_v_curve = (
            self._compute_v_curve_1d(farther_stronger_field, z_m)
        )
        farther_stronger_min_km, farther_stronger_min_v = self._minimum_distance(
            farther_stronger_distances_km,
            farther_stronger_v_curve,
        )

        return {
            "status": "completed",
            "success": True,
            "random_seed": self.random_seed,
            "parameters": {
                "p": p,
                "outer_scale_km": outer_scale_km,
                "sigma_phi_base_rad": sigma_phi_base_rad,
                "stronger_to_weaker_ratio": ratio,
                "separation_km": separation_km,
                "wavelength_signal": constants.DEFAULT_SIGNAL,
                "wavelength_m": wavelength_m,
                "screen_order_forward": (
                    "farther screen is applied first, then propagation by "
                    "separation, then closer screen, then propagation to receiver"
                ),
            },
            "screen_distances_km": {
                "closer_screen": closer_screen_distance_km,
                "farther_screen": farther_screen_distance_km,
            },
            "equal_strength": {
                "distances_km": equal_distances_km,
                "v_curve": equal_v_curve,
                "estimated_minimum_km": equal_min_km,
                "minimum_v": equal_min_v,
                "local_minima_km": self._local_minima_distances(
                    equal_distances_km,
                    equal_v_curve,
                ),
                "interpretation": "not_separately_resolvable",
                "paper_alignment_note": (
                    "Two aligned equal-strength phase screens are not expected "
                    "to produce two separately resolvable BP geolocations."
                ),
            },
            "closer_screen_stronger": {
                "distances_km": closer_stronger_distances_km,
                "v_curve": closer_stronger_v_curve,
                "estimated_minimum_km": closer_stronger_min_km,
                "minimum_v": closer_stronger_min_v,
                "expected_dominant_screen": "closer_screen",
                "expected_distance_km": closer_screen_distance_km,
                "success": abs(closer_stronger_min_km - closer_screen_distance_km)
                <= self._bp_step_km()
                if math.isfinite(closer_stronger_min_km)
                else False,
            },
            "farther_screen_stronger": {
                "distances_km": farther_stronger_distances_km,
                "v_curve": farther_stronger_v_curve,
                "estimated_minimum_km": farther_stronger_min_km,
                "minimum_v": farther_stronger_min_v,
                "expected_dominant_screen": "farther_screen",
                "expected_distance_km": farther_screen_distance_km,
                "success": abs(farther_stronger_min_km - farther_screen_distance_km)
                <= self._bp_step_km()
                if math.isfinite(farther_stronger_min_km)
                else False,
            },
        }

    def two_screens_misaligned(self) -> dict[str, Any]:
        """Validate smaller-scan-angle dominance for two misaligned screens.

        This experiment uses 3D forward propagation and 2D BP on extracted
        receiver-trajectory signals. Because the current AppConfig does not
        expose a dedicated 2D synthetic grid, this implementation spans the
        configured synthetic spatial interval and uses a computationally bounded
        square grid for reliability.

        Returns:
            Dictionary with one result per configured alpha pair.
        """
        p: float = self._synthetic_p()
        outer_scale_km: float = self._synthetic_outer_scale_km()
        sigma_phi_rad: float = self._observed_sigma_phi_rad()
        separation_km: float = constants.DEFAULT_TWO_SCREEN_SEPARATION_KM
        closer_screen_distance_km: float = self._observed_screen_distance_km()
        farther_screen_distance_km: float = closer_screen_distance_km + separation_km
        wavelength_m: float = self._wavelength_m()
        alpha_cases_deg: tuple[tuple[float, float], ...] = (
            constants.DEFAULT_MISALIGNED_ALPHA_CASES_DEG
        )

        self._validate_screen_distances_inside_grid(
            [closer_screen_distance_km, farther_screen_distance_km],
            "two_screens_misaligned",
        )

        y_m, z_m = self._make_yz_grid_2d()
        y_center_index: int = int(y_m.size // 2)

        cases: list[dict[str, Any]] = []
        for alpha1_deg, alpha2_deg in alpha_cases_deg:
            alpha1_rad: float = math.radians(alpha1_deg)
            alpha2_rad: float = math.radians(alpha2_deg)

            closer_phase_rad: NDArray[np.float64] = self.generator.generate_2d(
                y_m=y_m,
                z_m=z_m,
                alpha_rad=alpha1_rad,
                p=p,
                outer_scale_km=outer_scale_km,
                sigma_phi_rad=sigma_phi_rad,
                wavelength_m=wavelength_m,
            )
            farther_phase_rad: NDArray[np.float64] = self.generator.generate_2d(
                y_m=y_m,
                z_m=z_m,
                alpha_rad=alpha2_rad,
                p=p,
                outer_scale_km=outer_scale_km,
                sigma_phi_rad=sigma_phi_rad,
                wavelength_m=wavelength_m,
            )

            field_yz: NDArray[np.complex128] = np.exp(
                1j * farther_phase_rad
            ).astype(np.complex128, copy=False)
            field_yz = self.propagator.propagate_3d(
                field_yz=field_yz,
                y_m=y_m,
                z_m=z_m,
                distance_m=separation_km * _M_PER_KM,
            )
            field_yz = field_yz * np.exp(1j * closer_phase_rad)
            receiver_field_yz: NDArray[np.complex128] = self.propagator.propagate_3d(
                field_yz=field_yz,
                y_m=y_m,
                z_m=z_m,
                distance_m=closer_screen_distance_km * _M_PER_KM,
            )

            receiver_trajectory_signal: NDArray[np.complex128] = receiver_field_yz[
                y_center_index,
                :
            ].astype(np.complex128, copy=False)

            alpha1_projected_z_m: NDArray[np.float64] = z_m * math.cos(alpha1_rad)
            alpha2_projected_z_m: NDArray[np.float64] = z_m * math.cos(alpha2_rad)

            distances_km, v_alpha1 = self._compute_v_curve_1d(
                receiver_field=receiver_trajectory_signal,
                z_m=alpha1_projected_z_m,
            )
            _, v_alpha2 = self._compute_v_curve_1d(
                receiver_field=receiver_trajectory_signal,
                z_m=alpha2_projected_z_m,
            )

            min_alpha1_km, min_alpha1_v = self._minimum_distance(
                distances_km,
                v_alpha1,
            )
            min_alpha2_km, min_alpha2_v = self._minimum_distance(
                distances_km,
                v_alpha2,
            )

            expected_localized_screen: int = 1 if alpha1_deg < alpha2_deg else 2
            expected_distance_km: float = (
                closer_screen_distance_km
                if expected_localized_screen == 1
                else farther_screen_distance_km
            )

            selected_min_km: float = (
                min_alpha1_km if expected_localized_screen == 1 else min_alpha2_km
            )
            success: bool = (
                math.isfinite(selected_min_km)
                and abs(selected_min_km - expected_distance_km) <= self._bp_step_km()
            )

            cases.append(
                {
                    "alpha1_deg": float(alpha1_deg),
                    "alpha2_deg": float(alpha2_deg),
                    "screen1_distance_km": closer_screen_distance_km,
                    "screen2_distance_km": farther_screen_distance_km,
                    "distances_km": distances_km,
                    "v_curve_for_alpha1_plane": v_alpha1,
                    "v_curve_for_alpha2_plane": v_alpha2,
                    "estimated_minimum_alpha1_plane_km": min_alpha1_km,
                    "estimated_minimum_alpha2_plane_km": min_alpha2_km,
                    "minimum_v_alpha1_plane": min_alpha1_v,
                    "minimum_v_alpha2_plane": min_alpha2_v,
                    "localized_screen": expected_localized_screen
                    if success
                    else self._infer_localized_screen_from_minima(
                        min_alpha1_km=min_alpha1_km,
                        min_alpha2_km=min_alpha2_km,
                        screen1_distance_km=closer_screen_distance_km,
                        screen2_distance_km=farther_screen_distance_km,
                    ),
                    "expected_localized_screen": expected_localized_screen,
                    "expected_distance_km": expected_distance_km,
                    "success": success,
                }
            )

        return {
            "status": "completed",
            "success": all(bool(case["success"]) for case in cases) if cases else False,
            "random_seed": self.random_seed,
            "parameters": {
                "p": p,
                "outer_scale_km": outer_scale_km,
                "sigma_phi_rad": sigma_phi_rad,
                "alpha_cases_deg": [list(pair) for pair in alpha_cases_deg],
                "separation_km": separation_km,
                "screen1_distance_km": closer_screen_distance_km,
                "screen2_distance_km": farther_screen_distance_km,
                "wavelength_signal": constants.DEFAULT_SIGNAL,
                "wavelength_m": wavelength_m,
                "grid_interval_km": self._modeled_spatial_interval_km(),
                "grid_samples_y": int(y_m.size),
                "grid_samples_z": int(z_m.size),
            },
            "cases": cases,
            "configuration_warning": (
                "The current AppConfig/config.yaml does not expose a dedicated "
                "2D synthetic propagation grid. This experiment uses a bounded "
                "2D grid spanning the configured spatial interval to avoid "
                "unreproducible memory blow-up from a full 10 m by 10 m "
                "40 km square grid."
            ),
            "paper_alignment_note": (
                "For equal-strength misaligned screens, the smaller scan angle "
                "is expected to dominate localization."
            ),
        }

    def thermal_noise(self) -> dict[str, Any]:
        """Evaluate thermal-noise effects on BP V(L) curves.

        Returns:
            Dictionary containing baseline and noisy V(L) curves for configured
            sigma_phi and SNR cases.
        """
        p: float = self._synthetic_p()
        outer_scale_km: float = self._synthetic_outer_scale_km()
        screen_distance_km: float = constants.DEFAULT_NOISE_SCREEN_DISTANCE_KM
        sigma_phi_cases_rad: tuple[float, ...] = (
            constants.DEFAULT_NOISE_SIGMA_PHI_CASES_RAD
        )
        snr_1hz_cases_vv: tuple[float, ...] = constants.DEFAULT_NOISE_SNR_1HZ_CASES_VV
        sampling_rate_hz: float = constants.DEFAULT_NOISE_SAMPLING_RATE_HZ
        projected_scan_velocity_km_s: float = (
            constants.DEFAULT_PROJECTED_SCAN_VELOCITY_KM_S
        )
        internal_grid_step_m: float = constants.DEFAULT_NOISE_INTERNAL_GRID_STEP_M
        wavelength_m: float = self._wavelength_m()
        z_m: NDArray[np.float64] = self._make_z_grid_1d()

        observational_spacing_m: float = (
            projected_scan_velocity_km_s * _M_PER_KM / sampling_rate_hz
        )
        internal_noise_factor: float = math.sqrt(
            observational_spacing_m / internal_grid_step_m
        )

        cases: list[dict[str, Any]] = []
        for sigma_phi_rad in sigma_phi_cases_rad:
            phase_screen_rad: NDArray[np.float64] = self.generator.generate_1d(
                z_m=z_m,
                p=p,
                outer_scale_km=outer_scale_km,
                sigma_phi_rad=float(sigma_phi_rad),
                wavelength_m=wavelength_m,
            )
            screen_field: NDArray[np.complex128] = np.exp(
                1j * phase_screen_rad
            ).astype(np.complex128, copy=False)
            clean_receiver_field: NDArray[np.complex128] = self.propagator.propagate_2d(
                field_z=screen_field,
                z_m=z_m,
                distance_m=screen_distance_km * _M_PER_KM,
            )

            baseline_distances_km, baseline_v_curve = self._compute_v_curve_1d(
                receiver_field=clean_receiver_field,
                z_m=z_m,
            )
            baseline_min_km, baseline_min_v = self._minimum_distance(
                baseline_distances_km,
                baseline_v_curve,
            )
            baseline_depth: float = self._depth_metric(baseline_v_curve)

            cases.append(
                {
                    "sigma_phi_rad": float(sigma_phi_rad),
                    "snr_1hz_vv": math.inf,
                    "effective_snr": math.inf,
                    "noise_sigma_complex_component": 0.0,
                    "distances_km": baseline_distances_km,
                    "v_curve": baseline_v_curve,
                    "estimated_minimum_km": baseline_min_km,
                    "minimum_v": baseline_min_v,
                    "depth_metric": baseline_depth,
                    "minimum_detectable": math.isfinite(baseline_min_km)
                    and abs(baseline_min_km - screen_distance_km) <= self._bp_step_km(),
                    "case_type": "noiseless_baseline",
                }
            )

            for snr_1hz_vv in snr_1hz_cases_vv:
                effective_snr: float = constants.effective_snr_for_sampling(
                    snr_1hz_vv=snr_1hz_vv,
                    sampling_rate_hz=sampling_rate_hz,
                )
                noise_sigma: float = (
                    1.0 / effective_snr * internal_noise_factor
                    if effective_snr > 0.0
                    else math.nan
                )
                noisy_receiver_field: NDArray[np.complex128] = self._add_complex_noise(
                    field=clean_receiver_field,
                    noise_sigma=noise_sigma,
                )

                distances_km, v_curve = self._compute_v_curve_1d(
                    receiver_field=noisy_receiver_field,
                    z_m=z_m,
                )
                estimated_minimum_km, minimum_v = self._minimum_distance(
                    distances_km,
                    v_curve,
                )
                depth_metric: float = self._depth_metric(v_curve)
                minimum_detectable: bool = (
                    math.isfinite(estimated_minimum_km)
                    and abs(estimated_minimum_km - screen_distance_km)
                    <= self._bp_step_km()
                )

                cases.append(
                    {
                        "sigma_phi_rad": float(sigma_phi_rad),
                        "snr_1hz_vv": float(snr_1hz_vv),
                        "effective_snr": effective_snr,
                        "noise_sigma_complex_component": noise_sigma,
                        "distances_km": distances_km,
                        "v_curve": v_curve,
                        "estimated_minimum_km": estimated_minimum_km,
                        "minimum_v": minimum_v,
                        "depth_metric": depth_metric,
                        "minimum_detectable": minimum_detectable,
                        "case_type": "noisy",
                    }
                )

        return {
            "status": "completed",
            "success": True,
            "random_seed": self.random_seed,
            "parameters": {
                "p": p,
                "outer_scale_km": outer_scale_km,
                "phase_screen_distance_km": screen_distance_km,
                "sigma_phi_cases_rad": list(sigma_phi_cases_rad),
                "snr_1hz_cases_vv": list(snr_1hz_cases_vv),
                "sampling_rate_hz": sampling_rate_hz,
                "projected_scan_velocity_km_s": projected_scan_velocity_km_s,
                "observational_spacing_m": observational_spacing_m,
                "internal_grid_step_m": internal_grid_step_m,
                "internal_noise_factor": internal_noise_factor,
                "wavelength_signal": constants.DEFAULT_SIGNAL,
                "wavelength_m": wavelength_m,
            },
            "cases": cases,
            "paper_alignment_note": (
                "Thermal noise is expected to raise the V(L) baseline while "
                "leaving the gross curve shape similar; strong scintillation "
                "should remain detectable at SNR 200-600 V/V, while the "
                "0.25 rad threshold case may become marginal."
            ),
        }

    def multivalued_geometry(self) -> dict[str, Any]:
        """Return structured status for synthetic multivalued-geometry demo.

        The paper's Figure 20 demonstration uses values such as L0=2500 km and
        cos(alpha0)=0.4..1.0, but these parameters are not exposed by the
        provided AppConfig/config.yaml. To avoid silently hard-coding unconfigured
        figure-specific values, this method returns an explicit skipped result.

        Returns:
            Skipped-result dictionary explaining required missing parameters.
        """
        return {
            "status": "skipped",
            "success": False,
            "random_seed": self.random_seed,
            "reason": (
                "Required multivalued geometry parameters are not present in "
                "the provided AppConfig/config.yaml interface."
            ),
            "paper_reference": "Figures 19-20",
            "required_parameters_if_configured_later": {
                "reference_distance_l0_km": "e.g., paper example 2500 km",
                "cos_alpha_reference_values": "e.g., paper family 0.4..1.0 step 0.1",
                "synthetic_crossing_distances_km": (
                    "e.g., paper illustrative 1200, 2500, 4000 km"
                ),
            },
            "paper_alignment_note": (
                "Multi-valued geolocations are explained in the paper as a "
                "scan-angle-geometry degeneracy, not as multiple physical "
                "irregularity regions. This reproduction does not invent the "
                "missing Figure 20 parameter set."
            ),
        }

    def _forward_two_aligned_screens(
        self,
        z_m: NDArray[np.float64],
        closer_screen_distance_km: float,
        separation_km: float,
        p: float,
        outer_scale_km: float,
        closer_sigma_phi_rad: float,
        farther_sigma_phi_rad: float,
        wavelength_m: float,
    ) -> NDArray[np.complex128]:
        """Forward propagate through two aligned 1D phase screens.

        Forward order is transmitter side to receiver side: farther screen,
        propagation by separation, closer screen, propagation to receiver.
        """
        farther_phase_rad: NDArray[np.float64] = self.generator.generate_1d(
            z_m=z_m,
            p=p,
            outer_scale_km=outer_scale_km,
            sigma_phi_rad=farther_sigma_phi_rad,
            wavelength_m=wavelength_m,
        )
        closer_phase_rad: NDArray[np.float64] = self.generator.generate_1d(
            z_m=z_m,
            p=p,
            outer_scale_km=outer_scale_km,
            sigma_phi_rad=closer_sigma_phi_rad,
            wavelength_m=wavelength_m,
        )

        field: NDArray[np.complex128] = np.exp(1j * farther_phase_rad).astype(
            np.complex128,
            copy=False,
        )
        field = self.propagator.propagate_2d(
            field_z=field,
            z_m=z_m,
            distance_m=separation_km * _M_PER_KM,
        )
        field = field * np.exp(1j * closer_phase_rad)
        field = self.propagator.propagate_2d(
            field_z=field,
            z_m=z_m,
            distance_m=closer_screen_distance_km * _M_PER_KM,
        )
        return field.astype(np.complex128, copy=False)

    def _compute_v_curve_1d(
        self,
        receiver_field: NDArray[np.complex128],
        z_m: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute BP V(L) over the configured distance grid."""
        distances_km: NDArray[np.float64] = self._distance_grid_km()
        v_curve: NDArray[np.float64] = np.full(distances_km.shape, np.nan)

        for index, distance_km in enumerate(distances_km):
            try:
                bp_field: NDArray[np.complex128] = self.propagator.backpropagate_2d(
                    field_z=receiver_field,
                    z_m=z_m,
                    distance_m=float(distance_km) * _M_PER_KM,
                )
                v_value: float = self.propagator.amplitude_variance(bp_field)
                if math.isfinite(v_value):
                    v_curve[index] = v_value
            except Exception:
                v_curve[index] = math.nan

        return distances_km, v_curve

    def _distance_grid_km(self) -> NDArray[np.float64]:
        """Return inclusive BP distance grid from AppConfig."""
        min_km: float = self._bp_min_km()
        max_km: float = self._bp_max_km()
        step_km: float = self._bp_step_km()

        grid: NDArray[np.float64] = np.arange(
            min_km,
            max_km + 0.5 * step_km,
            step_km,
            dtype=np.float64,
        )
        grid = grid[grid <= max_km + 0.5 * step_km]
        if grid.size == 0:
            raise ValueError("Configured BP distance grid is empty.")
        grid[0] = min_km
        grid[-1] = max_km if abs(grid[-1] - max_km) <= 0.5 * step_km else grid[-1]
        return grid

    def _make_z_grid_1d(self) -> NDArray[np.float64]:
        """Create centered 1D spatial grid using configured synthetic settings."""
        interval_m: float = self._modeled_spatial_interval_km() * _M_PER_KM
        step_m: float = self._internal_grid_step_m()
        sample_count: int = max(_MIN_GRID_SAMPLES, int(round(interval_m / step_m)) + 1)
        return np.linspace(
            -0.5 * interval_m,
            0.5 * interval_m,
            num=sample_count,
            dtype=np.float64,
        )

    def _make_yz_grid_2d(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create bounded centered y/z grids for 3D synthetic propagation."""
        interval_m: float = self._modeled_spatial_interval_km() * _M_PER_KM
        sample_count: int = _DEFAULT_MISALIGNED_GRID_SAMPLES
        y_m: NDArray[np.float64] = np.linspace(
            -0.5 * interval_m,
            0.5 * interval_m,
            num=sample_count,
            dtype=np.float64,
        )
        z_m: NDArray[np.float64] = np.linspace(
            -0.5 * interval_m,
            0.5 * interval_m,
            num=sample_count,
            dtype=np.float64,
        )
        return y_m, z_m

    @staticmethod
    def _compute_s4_from_field(field: NDArray[np.complex128]) -> float:
        """Compute S4 from propagated complex field."""
        field_array: NDArray[np.complex128] = np.asarray(field, dtype=np.complex128)
        if field_array.size == 0:
            return math.nan
        if not np.all(np.isfinite(field_array.real)) or not np.all(
            np.isfinite(field_array.imag)
        ):
            return math.nan

        amplitude: NDArray[np.float64] = np.abs(field_array)
        intensity: NDArray[np.float64] = amplitude * amplitude
        mean_intensity: float = float(np.mean(intensity))
        if not math.isfinite(mean_intensity) or mean_intensity <= 0.0:
            return math.nan

        variance: float = float(np.mean(intensity * intensity) - mean_intensity**2)
        if variance < 0.0 and abs(variance) < 1.0e-14:
            variance = 0.0
        if variance < 0.0 or not math.isfinite(variance):
            return math.nan

        return float(math.sqrt(variance) / mean_intensity)

    @staticmethod
    def _minimum_distance(
        distances_km: NDArray[np.float64],
        v_curve: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Return distance and value of finite global minimum."""
        distance_array: NDArray[np.float64] = np.asarray(distances_km, dtype=np.float64)
        v_array: NDArray[np.float64] = np.asarray(v_curve, dtype=np.float64)
        finite_mask: NDArray[np.bool_] = np.isfinite(distance_array) & np.isfinite(v_array)
        if not np.any(finite_mask):
            return math.nan, math.nan

        finite_distances: NDArray[np.float64] = distance_array[finite_mask]
        finite_values: NDArray[np.float64] = v_array[finite_mask]
        minimum_index: int = int(np.argmin(finite_values))
        return float(finite_distances[minimum_index]), float(finite_values[minimum_index])

    @staticmethod
    def _local_minima_distances(
        distances_km: NDArray[np.float64],
        v_curve: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return finite local-minimum distances for diagnostics."""
        distance_array: NDArray[np.float64] = np.asarray(distances_km, dtype=np.float64)
        v_array: NDArray[np.float64] = np.asarray(v_curve, dtype=np.float64)
        if distance_array.size < 3 or distance_array.size != v_array.size:
            return np.empty(0, dtype=np.float64)

        minima: list[float] = []
        for index in range(1, v_array.size - 1):
            left: float = float(v_array[index - 1])
            center: float = float(v_array[index])
            right: float = float(v_array[index + 1])
            if not (math.isfinite(left) and math.isfinite(center) and math.isfinite(right)):
                continue
            if center <= left and center <= right:
                minima.append(float(distance_array[index]))
        return np.asarray(minima, dtype=np.float64)

    def _depth_metric(self, v_curve: NDArray[np.float64]) -> float:
        """Compute a simple V-curve minimum-depth diagnostic."""
        v_array: NDArray[np.float64] = np.asarray(v_curve, dtype=np.float64)
        finite_values: NDArray[np.float64] = v_array[np.isfinite(v_array)]
        if finite_values.size < 3:
            return math.nan

        min_value: float = float(np.min(finite_values))
        if min_value <= 0.0 or not math.isfinite(min_value):
            return math.nan

        edge_value: float = float(min(finite_values[0], finite_values[-1]))
        if not math.isfinite(edge_value):
            return math.nan

        return float(edge_value / min_value)

    def _add_complex_noise(
        self,
        field: NDArray[np.complex128],
        noise_sigma: float,
    ) -> NDArray[np.complex128]:
        """Add independent Gaussian thermal noise to real/imaginary components."""
        field_array: NDArray[np.complex128] = np.asarray(field, dtype=np.complex128)
        sigma: float = float(noise_sigma)
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"noise_sigma must be finite and nonnegative, got {sigma}.")

        mean_amplitude: float = float(np.mean(np.abs(field_array)))
        if not math.isfinite(mean_amplitude) or mean_amplitude <= 0.0:
            mean_amplitude = 1.0

        scale: float = sigma * mean_amplitude
        noise_real: NDArray[np.float64] = self._noise_rng.normal(
            loc=0.0,
            scale=scale,
            size=field_array.shape,
        )
        noise_imag: NDArray[np.float64] = self._noise_rng.normal(
            loc=0.0,
            scale=scale,
            size=field_array.shape,
        )
        return (field_array + noise_real + 1j * noise_imag).astype(
            np.complex128,
            copy=False,
        )

    @staticmethod
    def _infer_localized_screen_from_minima(
        min_alpha1_km: float,
        min_alpha2_km: float,
        screen1_distance_km: float,
        screen2_distance_km: float,
    ) -> int | None:
        """Infer localized screen from which minimum is nearer to screen distance."""
        candidates: list[tuple[float, int]] = []
        if math.isfinite(min_alpha1_km):
            candidates.append((abs(min_alpha1_km - screen1_distance_km), 1))
        if math.isfinite(min_alpha2_km):
            candidates.append((abs(min_alpha2_km - screen2_distance_km), 2))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _validate_screen_distances_inside_grid(
        self,
        distances_km: list[float],
        context: str,
    ) -> None:
        """Validate synthetic screen distances lie inside configured BP range."""
        min_km: float = self._bp_min_km()
        max_km: float = self._bp_max_km()
        for distance_km in distances_km:
            if not (min_km <= float(distance_km) <= max_km):
                raise ValueError(
                    f"{context}: synthetic screen distance {distance_km} km "
                    f"is outside configured BP range [{min_km}, {max_km}] km."
                )

    @staticmethod
    def _unique_floats(values: list[float]) -> list[float]:
        """Return unique finite floats preserving first occurrence order."""
        unique_values: list[float] = []
        for value in values:
            value_float: float = float(value)
            if not math.isfinite(value_float):
                continue
            if not any(math.isclose(value_float, existing) for existing in unique_values):
                unique_values.append(value_float)
        return unique_values

    def _wavelength_m(self) -> float:
        """Return wavelength used by the injected propagator."""
        wavelength_m: float = float(getattr(self.propagator, "wavelength_m"))
        if not math.isfinite(wavelength_m) or wavelength_m <= 0.0:
            return constants.GPS_L1_WAVELENGTH_M
        return wavelength_m

    def _bp_min_km(self) -> float:
        """Return configured BP minimum distance."""
        return self._positive_config_float(
            "bp_min_distance_km",
            constants.DEFAULT_BP_MIN_DISTANCE_KM,
        )

    def _bp_max_km(self) -> float:
        """Return configured BP maximum distance."""
        return self._positive_config_float(
            "bp_max_distance_km",
            constants.DEFAULT_BP_MAX_DISTANCE_KM,
        )

    def _bp_step_km(self) -> float:
        """Return configured BP distance step."""
        return self._positive_config_float(
            "bp_step_km",
            constants.DEFAULT_BP_STEP_KM,
        )

    def _synthetic_p(self) -> float:
        """Return configured/paper synthetic spectral index."""
        return constants.DEFAULT_SYNTHETIC_SPECTRAL_INDEX_P

    def _synthetic_outer_scale_km(self) -> float:
        """Return configured/paper synthetic outer scale."""
        return constants.DEFAULT_SYNTHETIC_OUTER_SCALE_KM

    def _observed_sigma_phi_rad(self) -> float:
        """Return configured/paper observed average sigma_phi."""
        return constants.DEFAULT_OBSERVED_SIGMA_PHI_RAD

    def _observed_screen_distance_km(self) -> float:
        """Return configured/paper average screen-receiver distance."""
        return constants.DEFAULT_OBSERVED_SCREEN_DISTANCE_KM

    def _modeled_spatial_interval_km(self) -> float:
        """Return configured/paper synthetic spatial interval."""
        return constants.DEFAULT_NOISE_MODELED_INTERVAL_KM

    def _internal_grid_step_m(self) -> float:
        """Return configured/paper synthetic internal grid step."""
        return constants.DEFAULT_NOISE_INTERNAL_GRID_STEP_M

    def _positive_config_float(self, name: str, default: float) -> float:
        """Read a positive float from AppConfig with a constants fallback."""
        value: Any = getattr(self.config, name, default)
        scalar: float = float(value)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}.")
        return scalar


__all__ = ["SyntheticExperiments"]
