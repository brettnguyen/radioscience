# COSMIC-2 Back-Propagation Geolocation Reproduction

This repository reproduces the methodology from:

**“Geolocation of the Ionospheric Irregularities in the Equatorial F Layer by Back Propagation of COSMIC-2 Radio Occultation Signals.”**

The goal is to reproduce the paper’s back-propagation, BP, geolocation method for ionospheric F-region plasma irregularities using COSMIC-2 high-rate GNSS radio occultation scintillation data.

The implementation has two main modes:

1. **Synthetic validation experiments**  
   Reproduce the paper’s numerical modeling of forward propagation, back propagation, phase screens, scan-angle sensitivity, two-screen cases, thermal noise, and multi-valued geolocation behavior.

2. **Real COSMIC-2 processing**  
   Process COSMIC-2 high-rate phase and amplitude/SNR data from POD antennas, apply IGRF-13-based BP geolocation, quality control, and generate geolocation products such as monthly maps and L1/L2 comparison statistics.

No machine learning training is involved. The configuration explicitly states:

```yaml
training:
  applicable: false
```

---

## 1. Scientific Method Summary

The method localizes ionospheric plasma irregularities along the GNSS transmitter to COSMIC-2 receiver line of sight using high-rate complex radio occultation signals.

The implementation follows the paper’s assumptions:

1. **Thin phase screen approximation**  
   Irregularities are represented as a thin phase screen. During back propagation from the receiver, amplitude variance reaches a minimum near the phase screen.

2. **Anisotropic field-aligned irregularities**  
   Equatorial F-region irregularities are assumed elongated and aligned with the geomagnetic field.

3. **IGRF-13 magnetic-field orientation**  
   The IGRF-13 field direction is evaluated along the GNSS-LEO line of sight and used to define the 2D back-propagation plane.

4. **Coordinate-domain back propagation**  
   The method uses high-rate phase and amplitude/SNR data in 10-second windows.

5. **Geolocation criterion**

For each candidate magnetic-field distance \(L_{\mathrm{mf}}\), BP returns the distance \(L_0\) of the minimum of the normalized amplitude variance:

\[
V(L) = \frac{\langle A^2 \rangle}{\langle A \rangle^2} - 1
\]

The geolocation condition is:

\[
D(L_{\mathrm{mf}}) = L_0 - L_{\mathrm{mf}} = 0
\]

Only single-valued zero crossings are accepted.

---

## 2. Repository Layout

The intended project layout is:

```text
.
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
├── data/
│   ├── cosmic2/
│   │   ├── hr/
│   │   └── orbits/
│   ├── gnss/
│   └── f107/
│       └── f107.csv
├── outputs/
└── src/
    ├── config.py
    ├── core/
    │   ├── constants.py
    │   └── types.py
    ├── data/
    │   ├── cosmic_loader.py
    │   ├── orbit_loader.py
    │   └── f107_loader.py
    ├── experiments/
    │   ├── synthetic_experiments.py
    │   └── real_data_experiments.py
    ├── geolocation/
    │   ├── backprop_processor.py
    │   ├── geolocator.py
    │   ├── qc.py
    │   └── stationary_correction.py
    ├── geometry/
    │   ├── bp_plane.py
    │   ├── coordinates.py
    │   ├── los.py
    │   └── magnetic_field.py
    ├── io/
    │   └── writers.py
    ├── propagation/
    │   ├── fft_propagator.py
    │   ├── kirchhoff.py
    │   └── phase_screen.py
    ├── signal/
    │   ├── preprocessing.py
    │   └── scintillation.py
    └── visualization/
        └── plots.py
```

The data and output paths are controlled by `config.yaml`:

```yaml
paths:
  cosmic2_hr_data_dir: "data/cosmic2/hr"
  receiver_orbit_dir: "data/cosmic2/orbits"
  gnss_orbit_clock_dir: "data/gnss"
  f107_data_path: "data/f107/f107.csv"
  output_dir: "outputs"
```

---

## 3. Installation

### 3.1 Python Environment

Use Python with the dependencies listed in `requirements.txt`.

Required packages are:

```text
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.1.0
pyarrow>=14.0.0
xarray>=2023.12.0
netCDF4>=1.6.5
h5py>=3.10.0
pyyaml>=6.0.1
pydantic>=2.5.0
tqdm>=4.66.0
matplotlib>=3.8.0
cartopy>=0.22.0
pyproj>=3.6.0
pymap3d>=3.0.0
astropy>=6.0.0
ppigrf>=1.0.0
numba>=0.59.0
pytest>=7.4.0
ruff>=0.1.0
```

Install with:

```bash
pip install -r requirements.txt
```

No third-party non-Python dependencies are required by the task specification.

---

## 4. Data Acquisition

### 4.1 COSMIC-2 High-Rate Data

COSMIC-2 high-rate phase and amplitude/SNR data are required for real-data processing.

Configured source:

```yaml
data:
  cosmic2:
    source: "UCAR COSMIC Data Analysis and Archive Center"
    doi: "https://doi.org/10.5065/T353-C093"
```

Data should be placed under:

```text
data/cosmic2/hr/
```

Required contents:

- COSMIC-2 high-rate phase data.
- COSMIC-2 amplitude/SNR data from POD antennas.
- Metadata identifying:
  - LEO satellite ID,
  - GNSS transmitter ID,
  - GNSS constellation,
  - signal name,
  - POD antenna direction,
  - sampling rate,
  - timestamps.

The configuration states:

```yaml
data:
  cosmic2:
    years:
      - 2021
      - 2023
    pod_antennas:
      - velocity_facing
      - anti_velocity_facing
    sampling_rates_hz:
      gps: 50
      glonass: 100
    amplitude_representation: "SNR scaled to 1-Hz band"
```

Important paper-specific notes:

- GPS signals are tracked in closed-loop mode.
- GLONASS signals are tracked in open-loop mode.
- GLONASS may require ground data-modulation removal and phase connection unless the CDAAC product already provides connected phase.
- The exact HR file schema and preprocessing state must be verified against CDAAC documentation.

---

### 4.2 COSMIC-2 Receiver Orbits

Receiver orbit data are required and should be placed under:

```text
data/cosmic2/orbits/
```

Configured field:

```yaml
data:
  cosmic2:
    receiver_orbits_required: true
```

These data are used to interpolate receiver position and velocity to each high-rate sample.

---

### 4.3 GNSS Orbits and Clocks

GNSS transmitter positions are required.

Configured source:

```yaml
data:
  gnss:
    orbit_clock_source: "International GNSS Service"
    required: true
```

Place GNSS orbit/clock products under:

```text
data/gnss/
```

Useful source:

- International GNSS Service products:  
  <https://igs.org/products/>

These products are used to compute or interpolate transmitter state vectors for the high-rate observation times.

---

### 4.4 IGRF-13 Magnetic Field

The BP plane is defined using the IGRF-13 magnetic-field direction.

Configured field:

```yaml
data:
  magnetic_field:
    model: "IGRF-13"
    source_reference: "Alken et al. 2021"
    use_field_direction_only: true
```

Useful source:

- IGRF model information:  
  <https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field>

The implementation uses `ppigrf` or a replaceable IGRF-13-compatible provider through `MagneticFieldModel`.

Only the magnetic-field direction is used, not the magnitude.

---

### 4.5 F10.7 Solar Flux Data

F10.7 data are used only for solar activity comparison, similar to the paper’s Figure 22.

Configured source:

```yaml
data:
  f107:
    source: "https://omniweb.gsfc.nasa.gov/form/dx1.html"
```

Place cached data at:

```text
data/f107/f107.csv
```

Useful source:

- NASA OMNIWeb:  
  <https://omniweb.gsfc.nasa.gov/form/dx1.html>

---

## 5. Configuration Overview

All scientific thresholds and processing settings should be read from `config.yaml`.

Downstream code should not hard-code method parameters except physical constants.

---

### 5.1 Processing Window

```yaml
processing:
  window_seconds: 10
```

All scintillation indices and BP geolocations are computed in 10-second windows, as in the paper.

---

### 5.2 Scintillation Thresholds

```yaml
processing:
  scintillation_indices:
    sigma_phi_threshold_rad: 0.25
```

The pre-BP phase scintillation requirement is:

\[
\sigma_\phi > 0.25 \ \mathrm{rad}
\]

The configured S4 definition is:

```yaml
s4_definition: "sqrt(<I^2> - <I>^2) / <I>, where I = A^2"
```

The paper reports sanity-check accepted-event averages:

```yaml
synthetic_experiments:
  observed_average_values:
    sigma_phi_rad: 1.65
    s4: 0.32
    sigma_phi_over_s4_rad: 5.25
```

---

### 5.3 Phase Detrending

The paper does not fully specify the phase detrending method.

The configuration records this ambiguity:

```yaml
processing:
  scintillation_indices:
    phase_detrending:
      method: "unspecified_in_paper"
      default_implementation: "polynomial"
      polynomial_order: 1
```

The first implementation should use polynomial detrending with order 1 unless sensitivity tests or CDAAC documentation indicate a better match.

---

### 5.4 Tangent-Point Filter

```yaml
processing:
  tangent_point_filter:
    min_height_km: 150
```

Before BP, require:

\[
h_{\mathrm{tan}} > 150 \ \mathrm{km}
\]

This follows the paper’s exclusion of observations that may involve sporadic E-layer effects.

---

### 5.5 Back-Propagation Distance Grid

```yaml
back_propagation:
  distance_grid:
    min_km: 100
    max_km: 6000
    step_km: 100
```

For each BP plane, the inner BP loop evaluates:

\[
L = 100, 200, \ldots, 6000 \ \mathrm{km}
\]

---

### 5.6 Magnetic-Field Candidate Grid

```yaml
back_propagation:
  magnetic_field_candidate_grid:
    min_km: 100
    max_km: 6000
    step_km: 100
```

The outer loop evaluates the magnetic field and BP plane at:

\[
L_{\mathrm{mf}} = 100, 200, \ldots, 6000 \ \mathrm{km}
\]

---

### 5.7 Propagation Method

```yaml
back_propagation:
  representation: "coordinate"
  propagation_method_default: "fft_plane_wave"
  alternative_method_available: "kirchhoff_diffraction_integral"
```

The first operational implementation uses FFT plane-wave BP.

The Kirchhoff diffraction integral is retained as an alternative validation path, but the parsed paper text did not include the full integral equation.

---

### 5.8 Stationary-Transmitter Correction

```yaml
back_propagation:
  stationary_transmitter_correction:
    enabled: true
    default_option: 2
```

The default is Option 2:

```yaml
option_2:
  x2_prime: "x20"
  recommended_for: "FFT plane-wave propagation after interpolation to uniform z grid"
```

This matches the design decision to use FFT/plane-wave BP first.

---

### 5.9 Wavefront Curvature Correction

```yaml
back_propagation:
  wavefront_curvature_correction:
    enabled: true
    correction_path_term: "z^2 * tan(alpha)^2 / (2R)"
```

The phase/path correction is:

\[
\Delta S_{\mathrm{curv}} =
\frac{z^2 \tan^2(\alpha)}{2R}
\]

where \(R\) is the transmitter-receiver distance.

---

### 5.10 V-Curve Smoothing

```yaml
back_propagation:
  v_curve:
    smoothing:
      enabled: true
      method: "second_order_sliding_polynomial_regression"
      window_km: 1000
      polynomial_order: 2
```

The paper smooths \(V(L)\) using second-order sliding polynomial regression in a 1000 km window.

---

### 5.11 Minimum Detection

```yaml
back_propagation:
  minimum_detection:
    require_global_local_minimum: true
    reject_endpoint_minimum: true
    min_l1_samples: 2
    min_l2_samples: 2
    sample_spacing_km: 100
```

The minimum of \(V(L)\) must be a valid global local minimum, not an endpoint minimum.

The monotonic-increase spans on both sides must each be at least two 100 km samples.

---

### 5.12 Final Quality Control

```yaml
back_propagation:
  quality_control:
    q_threshold: 1.2
    cos_alpha_threshold: 0.1
    require_single_valued_geolocation: true
    discard_multivalued_geolocations: true
```

Final accepted geolocations require:

\[
Q > 1.2
\]

\[
\cos(\alpha) > 0.1
\]

and single-valued geolocation only.

Multi-valued geolocations are discarded, following the paper.

---

### 5.13 Monthly Maps

```yaml
real_data_experiments:
  monthly_l1_maps:
    enabled: true
    years:
      - 2021
      - 2023
    signal: "L1"
    local_time_hours:
      - 18
      - 24
    bin_size_deg:
      latitude: 3
      longitude: 3
```

Monthly maps use accepted post-sunset geolocations:

\[
18 \leq \mathrm{LT} < 24
\]

with:

\[
3^\circ \times 3^\circ
\]

latitude-longitude bins.

The local-time reference is configured as:

```yaml
processing:
  local_time_filter:
    reference: "geolocation_longitude"
```

The paper does not explicitly state the local-time reference longitude, so this should be documented in outputs.

---

### 5.14 L1/L2 Comparison

```yaml
signals:
  l1_l2_comparison:
    start_date: "2021-02-19"
    end_date: "2021-03-01"
    gps_l2_allowed:
      - "L2C"
    gps_l2_excluded:
      - "L2P"
```

The body text of the paper specifies February 19 to March 1, 2021. The abstract mentions a 2-month comparison, but this configuration follows the body text.

Expected counts from the paper:

```yaml
real_data_experiments:
  l1_l2_comparison:
    expected_counts:
      l1_geolocations: 13000
      l2_geolocations: 11000
      common_l1_l2_cases: 8000
    zonal_difference_histogram_bin_deg: 0.5
```

---

## 6. Command-Line Usage

The intended command-line entry point is `main.py`.

Examples below use the provided `config.yaml`.

---

### 6.1 Run Synthetic Validation Experiments

```bash
python main.py --config config.yaml --mode synthetic
```

This should run:

- single phase screen localization,
- outer-scale estimation,
- orientation-error experiment,
- two aligned phase screens,
- two misaligned phase screens,
- thermal-noise experiment,
- multi-valued geometry experiment.

Outputs should be written under:

```text
outputs/
```

according to:

```yaml
outputs:
  save_diagnostics: true
  save_v_curves_for_debug_cases: true
  save_d_curves_for_debug_cases: true
```

---

### 6.2 Process One Year of COSMIC-2 L1 Data

```bash
python main.py --config config.yaml --mode process-year --year 2021 --signal L1
```

```bash
python main.py --config config.yaml --mode process-year --year 2023 --signal L1
```

These runs process COSMIC-2 HR L1 windows for the selected year and produce interval-level geolocation results.

---

### 6.3 Generate Monthly Maps

```bash
python main.py --config config.yaml --mode monthly-maps --year 2021 --signal L1
```

```bash
python main.py --config config.yaml --mode monthly-maps --year 2023 --signal L1
```

Monthly maps should use:

- accepted geolocations only,
- post-sunset local time 18–24 hr,
- 3° × 3° bins.

---

### 6.4 Run L1/L2 Comparison

```bash
python main.py --config config.yaml --mode l1-l2 --start-date 2021-02-19 --end-date 2021-03-01
```

This should:

- process L1,
- process L2/L2C,
- exclude GPS L2P,
- pair common cases by event/time metadata,
- generate zonal difference statistics and a 0.5°-bin histogram.

---

### 6.5 Plot F10.7 Solar Flux

```bash
python main.py --config config.yaml --mode f107
```

This should load or download F10.7 data and produce a Figure 22-style solar activity plot.

---

## 7. Synthetic Validation Logic

Synthetic experiments are configured under:

```yaml
synthetic_experiments:
  enabled: true
```

---

### 7.1 Phase Screen Parameters

```yaml
synthetic_experiments:
  phase_screen:
    approximation: "single_thin_phase_screen"
    spectral_index_p: 1.5
    outer_scale_km: 10
    spectrum_1d: "(k0^2 + kz^2)^(-p/2)"
```

The synthetic 1D phase screen spectrum is:

\[
\Phi(k_z) \propto (k_0^2 + k_z^2)^{-p/2}
\]

with:

\[
p = 1.5
\]

\[
l_0 = 10 \ \mathrm{km}
\]

\[
k_0 = \frac{2\pi}{l_0}
\]

---

### 7.2 Single Phase Screen

The expected behavior is:

- forward propagation produces increasing amplitude scintillation after the screen,
- back propagation reduces amplitude variance toward the screen,
- \(V(L)\) has a minimum at the true screen distance.

Configured reference distance:

```yaml
synthetic_experiments:
  observed_average_values:
    distance_from_phase_screen_to_receiver_km: 1900
```

---

### 7.3 Outer-Scale Estimation

The paper reports:

```yaml
synthetic_experiments:
  observed_average_values:
    sigma_phi_over_s4_rad: 5.25
    distance_from_phase_screen_to_receiver_km: 1900
    estimated_outer_scale_km: 9.7
```

The reproduction should verify that using the observed ratio:

\[
\langle \sigma_\phi / S_4 \rangle \approx 5.25 \ \mathrm{rad}
\]

and average distance:

\[
\langle L \rangle \approx 1900 \ \mathrm{km}
\]

leads to an outer scale close to:

\[
l_0 \approx 9.7 \ \mathrm{km}
\]

consistent with the chosen synthetic value of 10 km.

---

### 7.4 Orientation Error

Configured case:

```yaml
synthetic_experiments:
  modeled_cases:
    orientation_error:
      delta_alpha_deg: 0.6
      formula: "DeltaL = 2 * L * tan(alpha) * DeltaAlpha"
```

The expected relation is:

\[
\Delta L = 2 L \tan(\alpha) \Delta \alpha
\]

Table cases:

```yaml
table_cases:
  - alpha_deg: 15
    distances_km: [1000, 3000, 5000]
  - alpha_deg: 45
    distances_km: [1000, 3000, 5000]
  - alpha_deg: 75
    distances_km: [1000, 3000, 5000]
```

The implementation should reproduce the paper’s Table 1 behavior: geolocation error increases with both distance and scan angle.

---

### 7.5 Two Phase Screens With Same Orientation

Configured case:

```yaml
two_phase_screens_same_orientation:
  separation_km: 1000
  equal_sigma_phi_case: true
  unequal_sigma_phi_case: true
  sigma_phi_ratio_stronger_to_weaker: 1.5
```

Expected outcomes:

1. Equal-strength screens with the same orientation are not separately resolvable.
2. If one screen has 1.5 times stronger phase fluctuation, BP localizes the stronger screen.
3. This dominance does not depend on whether the stronger screen is closer to or farther from the receiver.

---

### 7.6 Two Phase Screens With Different Orientation

Configured case:

```yaml
two_phase_screens_different_orientation:
  forward_propagation_dimension: "3D"
  back_propagation_dimension: "2D"
  alpha_cases_deg:
    - [30, 60]
    - [60, 30]
```

Expected outcomes:

- With equal \(\sigma_\phi\), only the screen with smaller scan angle \(\alpha\) is correctly localized.
- This remains true when the screen order is swapped.

---

### 7.7 Thermal Noise

Configured case:

```yaml
thermal_noise:
  phase_screen_distance_km: 2000
  sigma_phi_cases_rad:
    - 1.65
    - 0.25
  snr_1hz_cases_vv:
    - 200
    - 400
    - 600
  average_snr_1hz_vv: 400
  assumed_sampling_rate_hz: 100
  assumed_projected_scan_velocity_km_s: 4
  modeled_spatial_interval_km: 40
  windowed_spatial_interval_km: 35
  internal_grid_step_m: 10
```

Expected outcomes:

- Noise raises the entire \(V(L)\) curve.
- The shape of \(V(L)\) remains approximately unchanged.
- For \(\sigma_\phi = 1.65\) rad, the minimum remains detectable.
- For \(\sigma_\phi = 0.25\) rad, the minimum may become marginal at lower SNR.

---

## 8. Real COSMIC-2 Processing Logic

The real-data pipeline is organized around `SignalWindow` objects representing 10-second intervals.

For each window:

1. Prepare signal:
   - clean arrays,
   - unwrap phase if needed,
   - detrend phase,
   - normalize amplitude/SNR,
   - construct complex signal.

2. Compute scintillation metrics:
   - \(\sigma_\phi\),
   - \(S_4\),
   - mean SNR.

3. Apply pre-BP QC:
   - \(\sigma_\phi > 0.25\) rad,
   - tangent height \(>150\) km.

4. For each \(L_{\mathrm{mf}}\):
   - compute candidate point along LOS,
   - evaluate IGRF-13 magnetic-field direction,
   - build BP plane,
   - compute scan angle \(\alpha\),
   - run inner BP over 100–6000 km.

5. For each BP distance:
   - apply stationary-transmitter correction,
   - apply phase/path correction,
   - apply curvature correction,
   - interpolate corrected complex signal to uniform \(z\),
   - back propagate using FFT plane-wave propagation,
   - compute \(V(L)\).

6. Smooth \(V(L)\).

7. Find the global local minimum \(L_0\).

8. Compute:

\[
D(L_{\mathrm{mf}}) = L_0 - L_{\mathrm{mf}}
\]

9. Find zero crossings of \(D\).

10. Reject:
    - no crossing,
    - multiple crossings,
    - invalid minimum,
    - \(Q \le 1.2\),
    - \(\cos(\alpha) \le 0.1\).

11. Convert accepted distance to geodetic latitude, longitude, and altitude.

---

## 9. Output Products

Configured output behavior:

```yaml
outputs:
  save_interval_results: true
  save_diagnostics: true
  save_v_curves_for_debug_cases: true
  save_d_curves_for_debug_cases: true
  save_monthly_maps: true
  save_l1_l2_histogram: true
  save_f107_plot: true
```

Expected products include:

- interval-level geolocation tables,
- rejection reason logs,
- \(V(L)\) diagnostic curves,
- \(D(L_{\mathrm{mf}})\) diagnostic curves,
- \(\cos \alpha(L)\) diagnostics,
- monthly 3° × 3° maps,
- L1/L2 paired comparison table,
- L1/L2 zonal difference histogram,
- F10.7 solar flux plot,
- synthetic experiment plots and arrays.

---

## 10. Reproduction Targets

### 10.1 Synthetic Targets

A successful synthetic reproduction should show:

1. \(V(L)\) minimum at the true phase screen.
2. Outer-scale estimate near 9.7 km when using the observed \(\sigma_\phi/S_4\) ratio and 1900 km distance.
3. Orientation error follows:

\[
\Delta L = 2 L \tan(\alpha) \Delta \alpha
\]

4. Two aligned equal-strength screens are not separately resolved.
5. For aligned unequal screens, the stronger screen dominates localization.
6. For misaligned equal-strength screens, the smaller scan-angle screen dominates localization.
7. Thermal noise raises the \(V(L)\) baseline.
8. Multi-valued geolocations can arise from scan-angle geometry rather than multiple physical irregularity regions.

---

### 10.2 Real-Data Targets

Configured expected accepted L1 geolocation counts:

```yaml
real_data_experiments:
  monthly_l1_maps:
    expected_total_geolocations:
      2021: 332000
      2023: 1108000
```

The monthly maps should reproduce the paper’s qualitative morphology:

- December–January: strongest activity mainly in the American/Atlantic sector.
- May–July: maxima in African and Pacific sectors, with weaker American-sector activity.
- September–October: activity at many longitudes.
- 2023 should have substantially more geolocations than 2021.

L1/L2 expected counts:

```yaml
real_data_experiments:
  l1_l2_comparison:
    expected_counts:
      l1_geolocations: 13000
      l2_geolocations: 11000
      common_l1_l2_cases: 8000
```

The L1-L2 zonal difference histogram should be centered near zero.

The paper reports that approximately 8% of COSMIC-2 cases are multi-valued. Multi-valued cases should be logged and discarded.

---

## 11. Important Ambiguities to Track

The configuration records several unresolved details:

```yaml
unclear_from_paper:
  - "Exact phase detrending method before sigma_phi calculation."
  - "Exact COSMIC-2 HR data field names and preprocessing state."
  - "Whether final large-scale processing used FFT plane-wave BP or Kirchhoff integral BP."
  - "Exact SNR/amplitude normalization before back propagation."
  - "Exact local-time reference point for post-sunset filtering."
  - "Sign convention of final geolocation vector should be verified."
  - "L1/L2 comparison period differs between abstract and body text."
```

These should be documented in any reproduction report.

Particular care is required for:

1. **Phase detrending**  
   The implementation default is polynomial order 1, but this is not confirmed by the paper.

2. **Amplitude normalization**  
   COSMIC-2 amplitude is represented as SNR scaled to a 1-Hz band. The exact normalization before BP is not fully specified.

3. **GLONASS open-loop processing**  
   Confirm whether phase demodulation and connection are already present in the input product.

4. **Geolocation vector sign convention**  
   The configured equation is:

   ```yaml
   equation: "R_geo = R_rx + L * n_rt"
   n_rt_definition: "(R_rx - R_tx) / |R_rx - R_tx|"
   ```

   This sign convention must be verified so that accepted geolocations lie along the receiver-transmitter line of sight.

5. **Operational propagation method**  
   The reproduction defaults to FFT plane-wave BP with stationary-correction Option 2, but the paper also discusses a Kirchhoff diffraction-integral approach.

---

## 12. Reproducibility Notes

The configured random seed is:

```yaml
reproducibility:
  random_seed: 42
```

Logging should include:

```yaml
reproducibility:
  log_rejection_reasons: true
  log_multivalued_cases: true
  log_processing_statistics: true
```

Recommended aggregate statistics to report:

- total 10-second windows,
- windows passing \(\sigma_\phi\),
- windows passing tangent-height filter,
- windows with valid BP minima,
- windows rejected by \(Q\),
- windows rejected by \(\cos \alpha\),
- multi-valued windows,
- final accepted geolocations,
- accepted-event mean \(\sigma_\phi\),
- accepted-event mean \(S_4\),
- accepted-event mean SNR,
- accepted-event mean geolocation distance.

These diagnostics are necessary to determine whether differences from the paper arise from data availability, preprocessing, detrending, propagation method, or QC implementation.