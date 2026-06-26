"""Real COSMIC-2 data experiments for BP geolocation reproduction.

This module orchestrates the paper's real-data experiments after the lower-level
COSMIC-2 loading and geolocation pipeline components have been constructed.

Implemented experiment responsibilities:
    * Process all high-rate COSMIC-2 windows for a year and signal selector.
    * Build accepted post-sunset monthly 3° x 3° geolocation bin tables.
    * Process the configured L1/L2 comparison period.
    * Pair common accepted L1/L2 geolocation cases.
    * Compute L1-L2 zonal longitude differences in degrees.

This module intentionally does not parse COSMIC-2 files directly, interpolate
orbits directly, compute scintillation indices, run back propagation directly,
evaluate magnetic fields, or reimplement scientific QC thresholds. Those tasks
belong to ``CosmicLoader``, ``Geolocator``, and lower-level modules.

Conventions:
    * Annual processing returns accepted and rejected rows so processing
      statistics and rejection reasons can be reproduced.
    * Monthly binning filters accepted rows only and uses ``local_time_hr``
      already computed at geolocation longitude.
    * L1/L2 pairing filters accepted rows only.
    * L1/L2 zonal differences are longitude differences in degrees, wrapped to
      ``[-180, 180)`` as in the paper's Figure 23 histogram.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import logging
import math
from typing import Any, Iterable, Iterator

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.config import AppConfig
from src.core import constants
from src.core.types import GeolocationResult, SignalWindow
from src.data.cosmic_loader import CosmicLoader
from src.geolocation.geolocator import Geolocator


_LOGGER = logging.getLogger(__name__)

_REQUIRED_MONTHLY_COLUMNS: tuple[str, ...] = (
    "accepted",
    "mid_time",
    "latitude_deg",
    "longitude_deg",
    "local_time_hr",
)

_REQUIRED_PAIR_COLUMNS: tuple[str, ...] = (
    "event_id",
    "leo_id",
    "gnss_id",
    "mid_time",
    "accepted",
    "latitude_deg",
    "longitude_deg",
    "altitude_km",
    "distance_km",
    "q",
    "cos_alpha",
)

_RESULT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "leo_id",
    "gnss_id",
    "signal_name",
    "mid_time",
    "accepted",
    "rejection_reason",
    "distance_km",
    "latitude_deg",
    "longitude_deg",
    "altitude_km",
    "local_time_hr",
    "sigma_phi_rad",
    "s4",
    "mean_snr_vv",
    "q",
    "cos_alpha",
    "d_slope",
    "is_multivalued",
)

_MONTHLY_BIN_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "lat_bin_min",
    "lat_bin_max",
    "lat_bin_center",
    "lon_bin_min",
    "lon_bin_max",
    "lon_bin_center",
    "count",
)

_DEFAULT_L1_SIGNAL: str = "L1"
_DEFAULT_L2_SIGNAL_SELECTOR: str = "L2"
_DEFAULT_L1_L2_WINDOW_SECONDS: int = constants.DEFAULT_WINDOW_SECONDS
_DEFAULT_LOCAL_TIME_START_HOUR: float = constants.DEFAULT_POSTSUNSET_START_HOUR
_DEFAULT_LOCAL_TIME_END_HOUR: float = constants.DEFAULT_POSTSUNSET_END_HOUR

_GPS_CONSTELLATION_KEYS: frozenset[str] = frozenset({"GPS", "G", "NAVSTAR"})
_GLONASS_CONSTELLATION_KEYS: frozenset[str] = frozenset({"GLONASS", "GLO", "R"})

_GPS_L2_ALLOWED: frozenset[str] = frozenset(constants.GPS_L2_ALLOWED_FOR_COMPARISON)
_GPS_L2_EXCLUDED: frozenset[str] = frozenset(constants.GPS_L2_EXCLUDED_FOR_COMPARISON)


class RealDataExperiments:
    """Orchestrate real COSMIC-2 BP geolocation experiments.

    Args:
        config: Validated application configuration.
        loader: COSMIC-2 high-rate data loader.
        geolocator: Per-window BP geolocation processor.

    Public methods follow the project design:
        * ``process_year(year, signal)``
        * ``process_l1_l2_period(start, end)``
        * ``make_monthly_bins(results)``
        * ``pair_l1_l2(l1_results, l2_results)``
    """

    def __init__(
        self,
        config: AppConfig,
        loader: CosmicLoader,
        geolocator: Geolocator,
    ) -> None:
        """Initialize real-data experiment orchestration.

        Args:
            config: Validated application configuration.
            loader: COSMIC-2 data loader.
            geolocator: Geolocation processor.

        Raises:
            TypeError: If dependencies have invalid types.
        """
        if not isinstance(config, AppConfig):
            raise TypeError(
                f"config must be an AppConfig, got {type(config).__name__}."
            )
        if not isinstance(loader, CosmicLoader):
            raise TypeError(
                f"loader must be a CosmicLoader, got {type(loader).__name__}."
            )
        if not isinstance(geolocator, Geolocator):
            raise TypeError(
                "geolocator must be a Geolocator, got "
                f"{type(geolocator).__name__}."
            )

        self.config: AppConfig = config
        self.loader: CosmicLoader = loader
        self.geolocator: Geolocator = geolocator

    def process_year(self, year: int, signal: str) -> pd.DataFrame:
        """Process all COSMIC-2 windows for one year and signal selector.

        The returned DataFrame contains both accepted and rejected results so
        rejection statistics can be reproduced.

        Args:
            year: UTC calendar year, e.g. 2021 or 2023.
            signal: Signal selector passed to ``CosmicLoader.iter_windows``,
                e.g. ``"L1"`` or ``"L2"``.

        Returns:
            DataFrame containing one row per processed 10-second window. If no
            windows are found, an empty DataFrame with stable result columns is
            returned.
        """
        year_value: int = self._validate_year(year)
        signal_selector: str = self._normalize_signal_name(signal)

        result_rows: list[dict[str, Any]] = []
        for window in self.loader.iter_windows(year=year_value, signal=signal_selector):
            result: GeolocationResult = self._process_window_safely(
                window=window,
                requested_signal=signal_selector,
            )
            result_rows.append(result.to_dict())

        return self._results_dataframe(result_rows)

    def process_l1_l2_period(self, start: date, end: date) -> pd.DataFrame:
        """Process and pair L1/L2 geolocations for a comparison period.

        The period convention is start-inclusive and end-exclusive:

            start <= window.mid_time.date() < end

        This follows a reproducible interval convention suitable for repeated
        period processing and avoids double counting at boundaries.

        Args:
            start: Inclusive UTC start date.
            end: Exclusive UTC end date.

        Returns:
            Paired accepted L1/L2 geolocation DataFrame containing
            ``zonal_difference_deg`` and additional diagnostic differences.

        Raises:
            ValueError: If the date interval is invalid.
        """
        start_date: date = self._coerce_date(start, "start")
        end_date: date = self._coerce_date(end, "end")
        if start_date >= end_date:
            raise ValueError(
                f"start must be earlier than end for [start, end) processing; "
                f"got {start_date.isoformat()} and {end_date.isoformat()}."
            )

        l1_results: pd.DataFrame = self._process_signal_period(
            start=start_date,
            end=end_date,
            signal_selector=_DEFAULT_L1_SIGNAL,
            include_window=self._include_l1_window,
        )

        l2_results: pd.DataFrame = self._process_signal_period(
            start=start_date,
            end=end_date,
            signal_selector=_DEFAULT_L2_SIGNAL_SELECTOR,
            include_window=self._include_l2_comparison_window,
        )

        paired_results: pd.DataFrame = self.pair_l1_l2(
            l1_results=l1_results,
            l2_results=l2_results,
        )

        paired_results.attrs["l1_total_rows"] = int(len(l1_results))
        paired_results.attrs["l2_total_rows"] = int(len(l2_results))
        paired_results.attrs["l1_accepted_rows"] = int(
            self._accepted_mask(l1_results).sum()
        ) if not l1_results.empty else 0
        paired_results.attrs["l2_accepted_rows"] = int(
            self._accepted_mask(l2_results).sum()
        ) if not l2_results.empty else 0
        paired_results.attrs["comparison_start_date"] = start_date.isoformat()
        paired_results.attrs["comparison_end_date_exclusive"] = end_date.isoformat()

        return paired_results

    def make_monthly_bins(self, results: pd.DataFrame) -> pd.DataFrame:
        """Create monthly post-sunset geolocation density bins.

        Filtering follows the paper's Figure 21 setup:
            * accepted geolocations only;
            * finite latitude/longitude/local-time coordinates;
            * post-sunset local time ``18 <= LT < 24`` by configuration/default;
            * latitude/longitude bin size from ``AppConfig.map_bin_deg``.

        Args:
            results: DataFrame returned by ``process_year``.

        Returns:
            DataFrame with one row per populated monthly latitude/longitude bin.

        Raises:
            TypeError: If ``results`` is not a DataFrame.
            ValueError: If required columns are missing or bin size is invalid.
        """
        if not isinstance(results, pd.DataFrame):
            raise TypeError(
                f"results must be a pandas.DataFrame, got {type(results).__name__}."
            )

        if results.empty:
            return self._empty_monthly_bins_dataframe()

        self._require_columns(results, _REQUIRED_MONTHLY_COLUMNS, "monthly binning")

        bin_size_deg: float = self._configured_map_bin_deg()
        self._validate_bin_size(bin_size_deg)

        local_time_start_hr: float = float(
            getattr(
                self.config,
                "post_sunset_start_hour",
                _DEFAULT_LOCAL_TIME_START_HOUR,
            )
        )
        local_time_end_hr: float = float(
            getattr(
                self.config,
                "post_sunset_end_hour",
                _DEFAULT_LOCAL_TIME_END_HOUR,
            )
        )

        if not (0.0 <= local_time_start_hr < 24.0 and 0.0 < local_time_end_hr <= 24.0):
            raise ValueError(
                "Configured local-time filter must satisfy "
                "0 <= start_hour < 24 and 0 < end_hour <= 24."
            )
        if local_time_start_hr >= local_time_end_hr:
            raise ValueError(
                "This reproduction expects a non-wrapping post-sunset local-time "
                f"interval; got {local_time_start_hr} to {local_time_end_hr}."
            )

        working: pd.DataFrame = results.copy()

        working["mid_time"] = pd.to_datetime(working["mid_time"], errors="coerce")
        working["latitude_deg"] = pd.to_numeric(
            working["latitude_deg"],
            errors="coerce",
        )
        working["longitude_deg"] = pd.to_numeric(
            working["longitude_deg"],
            errors="coerce",
        )
        working["local_time_hr"] = pd.to_numeric(
            working["local_time_hr"],
            errors="coerce",
        )

        accepted_mask: pd.Series = self._accepted_mask(working)
        finite_geo_mask: pd.Series = (
            np.isfinite(working["latitude_deg"].to_numpy(dtype=np.float64))
            & np.isfinite(working["longitude_deg"].to_numpy(dtype=np.float64))
            & np.isfinite(working["local_time_hr"].to_numpy(dtype=np.float64))
            & working["mid_time"].notna().to_numpy()
        )
        local_time_mask: pd.Series = (
            (working["local_time_hr"] >= local_time_start_hr)
            & (working["local_time_hr"] < local_time_end_hr)
        )

        filtered: pd.DataFrame = working.loc[
            accepted_mask & finite_geo_mask & local_time_mask
        ].copy()

        if filtered.empty:
            return self._empty_monthly_bins_dataframe()

        filtered["year"] = filtered["mid_time"].dt.year.astype(int)
        filtered["month"] = filtered["mid_time"].dt.month.astype(int)

        latitude_values: NDArray[np.float64] = filtered["latitude_deg"].to_numpy(
            dtype=np.float64
        )
        longitude_values: NDArray[np.float64] = self._wrap_longitude_deg(
            filtered["longitude_deg"].to_numpy(dtype=np.float64)
        )

        lat_bin_min, lat_bin_max, lat_bin_center = self._latitude_bins(
            latitude_values,
            bin_size_deg,
        )
        lon_bin_min, lon_bin_max, lon_bin_center = self._longitude_bins(
            longitude_values,
            bin_size_deg,
        )

        filtered["lat_bin_min"] = lat_bin_min
        filtered["lat_bin_max"] = lat_bin_max
        filtered["lat_bin_center"] = lat_bin_center
        filtered["lon_bin_min"] = lon_bin_min
        filtered["lon_bin_max"] = lon_bin_max
        filtered["lon_bin_center"] = lon_bin_center

        grouped: pd.DataFrame = (
            filtered.groupby(
                [
                    "year",
                    "month",
                    "lat_bin_min",
                    "lat_bin_max",
                    "lat_bin_center",
                    "lon_bin_min",
                    "lon_bin_max",
                    "lon_bin_center",
                ],
                dropna=False,
                observed=True,
            )
            .size()
            .reset_index(name="count")
        )

        grouped["count"] = grouped["count"].astype(int)
        grouped = grouped.sort_values(
            ["year", "month", "lat_bin_min", "lon_bin_min"],
            kind="mergesort",
        ).reset_index(drop=True)

        return grouped.loc[:, list(_MONTHLY_BIN_COLUMNS)]

    def pair_l1_l2(
        self,
        l1_results: pd.DataFrame,
        l2_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Pair accepted L1 and L2 geolocations and compute differences.

        Pairing keys are:
            * event_id
            * leo_id
            * gnss_id
            * mid_time_key

        where ``mid_time_key`` is floored to the configured processing-window
        duration, 10 seconds by default.

        Args:
            l1_results: L1 geolocation results DataFrame.
            l2_results: L2 geolocation results DataFrame.

        Returns:
            Inner-joined DataFrame with suffixes ``_l1`` and ``_l2``, plus:
                * ``zonal_difference_deg``
                * ``lat_difference_deg``
                * ``altitude_difference_km``
                * ``distance_difference_km``

        Raises:
            TypeError: If inputs are not DataFrames.
            ValueError: If required columns are missing.
        """
        if not isinstance(l1_results, pd.DataFrame):
            raise TypeError(
                "l1_results must be a pandas.DataFrame, got "
                f"{type(l1_results).__name__}."
            )
        if not isinstance(l2_results, pd.DataFrame):
            raise TypeError(
                "l2_results must be a pandas.DataFrame, got "
                f"{type(l2_results).__name__}."
            )

        if l1_results.empty or l2_results.empty:
            return self._empty_l1_l2_pairs_dataframe()

        self._require_columns(l1_results, _REQUIRED_PAIR_COLUMNS, "L1/L2 L1 pairing")
        self._require_columns(l2_results, _REQUIRED_PAIR_COLUMNS, "L1/L2 L2 pairing")

        l1: pd.DataFrame = l1_results.loc[self._accepted_mask(l1_results)].copy()
        l2: pd.DataFrame = l2_results.loc[self._accepted_mask(l2_results)].copy()

        if l1.empty or l2.empty:
            return self._empty_l1_l2_pairs_dataframe()

        l1 = self._prepare_pairing_dataframe(l1, label="L1")
        l2 = self._prepare_pairing_dataframe(l2, label="L2")

        l1 = self._deduplicate_pairing_keys(l1, label="L1")
        l2 = self._deduplicate_pairing_keys(l2, label="L2")

        pair_keys: list[str] = ["event_id", "leo_id", "gnss_id", "mid_time_key"]

        paired: pd.DataFrame = l1.merge(
            l2,
            how="inner",
            on=pair_keys,
            suffixes=("_l1", "_l2"),
            validate="one_to_one",
        )

        if paired.empty:
            return self._empty_l1_l2_pairs_dataframe()

        lon_l1: NDArray[np.float64] = paired["longitude_deg_l1"].to_numpy(
            dtype=np.float64
        )
        lon_l2: NDArray[np.float64] = paired["longitude_deg_l2"].to_numpy(
            dtype=np.float64
        )

        paired["zonal_difference_deg"] = self._wrap_longitude_deg(lon_l1 - lon_l2)
        paired["lat_difference_deg"] = (
            pd.to_numeric(paired["latitude_deg_l1"], errors="coerce")
            - pd.to_numeric(paired["latitude_deg_l2"], errors="coerce")
        )
        paired["altitude_difference_km"] = (
            pd.to_numeric(paired["altitude_km_l1"], errors="coerce")
            - pd.to_numeric(paired["altitude_km_l2"], errors="coerce")
        )
        paired["distance_difference_km"] = (
            pd.to_numeric(paired["distance_km_l1"], errors="coerce")
            - pd.to_numeric(paired["distance_km_l2"], errors="coerce")
        )

        paired = paired.sort_values(
            ["mid_time_key", "event_id", "leo_id", "gnss_id"],
            kind="mergesort",
        ).reset_index(drop=True)

        return paired

    def _process_signal_period(
        self,
        start: date,
        end: date,
        signal_selector: str,
        include_window: Any,
    ) -> pd.DataFrame:
        """Process one signal selector over a start-inclusive/end-exclusive period."""
        result_rows: list[dict[str, Any]] = []
        for year_value in self._years_in_period(start=start, end=end):
            for window in self.loader.iter_windows(year=year_value, signal=signal_selector):
                if not self._window_in_period(window=window, start=start, end=end):
                    continue
                if not include_window(window):
                    continue

                result: GeolocationResult = self._process_window_safely(
                    window=window,
                    requested_signal=signal_selector,
                )
                result_rows.append(result.to_dict())

        return self._results_dataframe(result_rows)

    def _process_window_safely(
        self,
        window: SignalWindow,
        requested_signal: str,
    ) -> GeolocationResult:
        """Resolve wavelength and process a window, returning rejection on failure."""
        try:
            wavelength_m: float = self._resolve_wavelength_m(
                window=window,
                requested_signal=requested_signal,
            )
        except Exception as exc:
            _LOGGER.warning(
                "Rejecting event=%s because signal wavelength could not be "
                "resolved from constellation=%r, signal=%r: %s",
                getattr(window, "event_id", ""),
                getattr(window, "constellation", ""),
                getattr(window, "signal_name", ""),
                exc,
            )
            return self._make_rejected_result(
                window=window,
                reason="unsupported_signal_wavelength",
            )

        try:
            return self.geolocator.process_window(
                window=window,
                wavelength_m=wavelength_m,
            )
        except Exception as exc:
            _LOGGER.warning(
                "Geolocator failed for event=%s: %s",
                getattr(window, "event_id", ""),
                exc,
            )
            return self._make_rejected_result(
                window=window,
                reason="geolocator_processing_failed",
            )

    def _resolve_wavelength_m(
        self,
        window: SignalWindow,
        requested_signal: str,
    ) -> float:
        """Resolve carrier wavelength from window metadata.

        GLONASS FDMA channel metadata is not part of ``SignalWindow``. If a
        GLONASS signal is encountered, this method uses the shared GLONASS base
        frequency constants with an explicit warning instead of silently falling
        back to GPS frequencies.

        Args:
            window: Signal window.
            requested_signal: Signal selector used by the caller, used only if
                ``window.signal_name`` is missing.

        Returns:
            Carrier wavelength in meters.

        Raises:
            ValueError: If signal metadata cannot be resolved.
        """
        signal_name: str = str(window.signal_name or requested_signal).strip()
        constellation: str = str(window.constellation or "").strip()

        if not signal_name:
            raise ValueError("Signal name is empty.")

        constellation_key: str = self._normalize_signal_name(constellation)
        signal_key: str = self._normalize_signal_name(signal_name)

        if constellation_key in _GLONASS_CONSTELLATION_KEYS or constellation_key.startswith(
            "GLO"
        ):
            if "L2" in signal_key:
                _LOGGER.warning(
                    "Using GLONASS L2 base frequency because SignalWindow does "
                    "not expose GLONASS FDMA channel metadata."
                )
                return constants.frequency_to_wavelength_m(
                    constants.GLONASS_L2_BASE_FREQUENCY_HZ
                )
            if "L1" in signal_key:
                _LOGGER.warning(
                    "Using GLONASS L1 base frequency because SignalWindow does "
                    "not expose GLONASS FDMA channel metadata."
                )
                return constants.frequency_to_wavelength_m(
                    constants.GLONASS_L1_BASE_FREQUENCY_HZ
                )
            raise ValueError(f"Unsupported GLONASS signal name {signal_name!r}.")

        return constants.signal_wavelength_m(
            signal_name=signal_name,
            constellation=constellation or None,
        )

    def _include_l1_window(self, window: SignalWindow) -> bool:
        """Return True if a window belongs to L1 processing."""
        signal_name: str = self._normalize_signal_name(window.signal_name)
        return "L1" in signal_name and "L2" not in signal_name

    def _include_l2_comparison_window(self, window: SignalWindow) -> bool:
        """Return True for L2 windows allowed by the paper/configuration.

        GPS L2P is excluded. GPS L2 is accepted only when explicitly identified
        as L2C. Non-GPS L2 signals are included when they are not excluded.
        """
        signal_name: str = self._normalize_signal_name(window.signal_name)
        constellation: str = self._normalize_signal_name(window.constellation)

        if "L2" not in signal_name:
            return False

        if any(excluded in signal_name for excluded in _GPS_L2_EXCLUDED):
            return False

        is_gps: bool = constellation in _GPS_CONSTELLATION_KEYS or constellation.startswith(
            "GPS"
        )
        if is_gps:
            return any(allowed in signal_name for allowed in _GPS_L2_ALLOWED)

        return True

    @staticmethod
    def _accepted_mask(results: pd.DataFrame) -> pd.Series:
        """Return a robust boolean accepted mask from a results DataFrame."""
        if "accepted" not in results.columns:
            return pd.Series(False, index=results.index)

        accepted_series: pd.Series = results["accepted"]
        if accepted_series.dtype == bool:
            return accepted_series.fillna(False)

        normalized: pd.Series = accepted_series.astype(str).str.strip().str.lower()
        return normalized.isin({"true", "1", "yes", "y", "accepted"})

    def _prepare_pairing_dataframe(self, dataframe: pd.DataFrame, label: str) -> pd.DataFrame:
        """Prepare accepted geolocation rows for L1/L2 pairing."""
        prepared: pd.DataFrame = dataframe.copy()

        prepared["mid_time"] = pd.to_datetime(prepared["mid_time"], errors="coerce")
        prepared = prepared.loc[prepared["mid_time"].notna()].copy()

        for column_name in (
            "latitude_deg",
            "longitude_deg",
            "altitude_km",
            "distance_km",
            "q",
            "cos_alpha",
        ):
            prepared[column_name] = pd.to_numeric(prepared[column_name], errors="coerce")

        finite_mask: NDArray[np.bool_] = (
            np.isfinite(prepared["latitude_deg"].to_numpy(dtype=np.float64))
            & np.isfinite(prepared["longitude_deg"].to_numpy(dtype=np.float64))
            & np.isfinite(prepared["altitude_km"].to_numpy(dtype=np.float64))
            & np.isfinite(prepared["distance_km"].to_numpy(dtype=np.float64))
        )
        prepared = prepared.loc[finite_mask].copy()

        if prepared.empty:
            return prepared

        prepared["longitude_deg"] = self._wrap_longitude_deg(
            prepared["longitude_deg"].to_numpy(dtype=np.float64)
        )
        prepared["mid_time_key"] = self._time_key(
            prepared["mid_time"],
            self._configured_window_seconds(),
        )

        for key_column in ("event_id", "leo_id", "gnss_id"):
            prepared[key_column] = prepared[key_column].astype(str)

        _LOGGER.debug(
            "Prepared %d accepted %s geolocations for L1/L2 pairing.",
            len(prepared),
            label,
        )

        return prepared

    def _deduplicate_pairing_keys(
        self,
        dataframe: pd.DataFrame,
        label: str,
    ) -> pd.DataFrame:
        """Handle duplicate L1/L2 pairing keys deterministically."""
        if dataframe.empty:
            return dataframe

        key_columns: list[str] = ["event_id", "leo_id", "gnss_id", "mid_time_key"]
        duplicate_mask: pd.Series = dataframe.duplicated(
            subset=key_columns,
            keep=False,
        )
        duplicate_count: int = int(duplicate_mask.sum())

        if duplicate_count == 0:
            return dataframe

        _LOGGER.warning(
            "Found %d duplicate accepted %s rows for L1/L2 pairing keys. "
            "Keeping the row with largest finite Q, then largest cos(alpha), "
            "then smallest absolute longitude for deterministic pairing.",
            duplicate_count,
            label,
        )

        deduped: pd.DataFrame = dataframe.copy()
        deduped["_q_sort"] = pd.to_numeric(deduped["q"], errors="coerce").fillna(
            -math.inf
        )
        deduped["_cos_alpha_sort"] = pd.to_numeric(
            deduped["cos_alpha"],
            errors="coerce",
        ).fillna(-math.inf)
        deduped["_abs_lon_sort"] = np.abs(
            pd.to_numeric(deduped["longitude_deg"], errors="coerce").fillna(math.inf)
        )

        deduped = deduped.sort_values(
            key_columns + ["_q_sort", "_cos_alpha_sort", "_abs_lon_sort"],
            ascending=[True, True, True, True, False, False, True],
            kind="mergesort",
        )
        deduped = deduped.drop_duplicates(subset=key_columns, keep="first")
        deduped = deduped.drop(
            columns=["_q_sort", "_cos_alpha_sort", "_abs_lon_sort"],
        ).reset_index(drop=True)

        return deduped

    @staticmethod
    def _time_key(times: pd.Series, window_seconds: int) -> pd.Series:
        """Normalize times to a configured processing-window key."""
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive for time-key creation.")

        timestamps: pd.Series = pd.to_datetime(times, errors="coerce", utc=True)
        frequency: str = f"{int(window_seconds)}s"
        return timestamps.dt.floor(frequency)

    @staticmethod
    def _wrap_longitude_deg(values: NDArray[np.float64] | pd.Series | float) -> NDArray[np.float64] | float:
        """Wrap longitude or longitude differences to [-180, 180)."""
        if np.isscalar(values):
            value: float = float(values)
            if not math.isfinite(value):
                return math.nan
            wrapped_value: float = ((value + 180.0) % 360.0) - 180.0
            if math.isclose(wrapped_value, 180.0, abs_tol=1.0e-12):
                wrapped_value = -180.0
            return float(wrapped_value)

        array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
        wrapped: NDArray[np.float64] = ((array + 180.0) % 360.0) - 180.0
        wrapped = np.where(np.isclose(wrapped, 180.0, atol=1.0e-12), -180.0, wrapped)
        wrapped = np.where(np.isfinite(array), wrapped, np.nan)
        return wrapped.astype(np.float64, copy=False)

    @staticmethod
    def _latitude_bins(
        latitudes_deg: NDArray[np.float64],
        bin_size_deg: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Assign latitude bins using [min, max) bins with 90° clipped into last bin."""
        latitudes: NDArray[np.float64] = np.asarray(latitudes_deg, dtype=np.float64)
        clipped_latitudes: NDArray[np.float64] = np.clip(latitudes, -90.0, 90.0)

        bin_index: NDArray[np.float64] = np.floor(
            (clipped_latitudes + 90.0) / bin_size_deg
        )
        max_index: float = round(180.0 / bin_size_deg) - 1.0
        bin_index = np.clip(bin_index, 0.0, max_index)

        bin_min: NDArray[np.float64] = -90.0 + bin_index * bin_size_deg
        bin_max: NDArray[np.float64] = bin_min + bin_size_deg
        bin_center: NDArray[np.float64] = 0.5 * (bin_min + bin_max)

        return bin_min, bin_max, bin_center

    @staticmethod
    def _longitude_bins(
        longitudes_deg: NDArray[np.float64],
        bin_size_deg: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Assign longitude bins using wrapped [-180, 180) coordinates."""
        longitudes: NDArray[np.float64] = np.asarray(
            RealDataExperiments._wrap_longitude_deg(longitudes_deg),
            dtype=np.float64,
        )

        bin_index: NDArray[np.float64] = np.floor(
            (longitudes + 180.0) / bin_size_deg
        )
        max_index: float = round(360.0 / bin_size_deg) - 1.0
        bin_index = np.clip(bin_index, 0.0, max_index)

        bin_min: NDArray[np.float64] = -180.0 + bin_index * bin_size_deg
        bin_max: NDArray[np.float64] = bin_min + bin_size_deg
        bin_center: NDArray[np.float64] = 0.5 * (bin_min + bin_max)
        bin_center = np.asarray(
            RealDataExperiments._wrap_longitude_deg(bin_center),
            dtype=np.float64,
        )

        return bin_min, bin_max, bin_center

    def _make_rejected_result(
        self,
        window: SignalWindow,
        reason: str,
    ) -> GeolocationResult:
        """Construct a rejected geolocation result for orchestration failures."""
        return GeolocationResult(
            event_id=getattr(window, "event_id", ""),
            leo_id=getattr(window, "leo_id", ""),
            gnss_id=getattr(window, "gnss_id", ""),
            signal_name=getattr(window, "signal_name", ""),
            mid_time=getattr(window, "mid_time", datetime(1970, 1, 1, tzinfo=timezone.utc)),
            accepted=False,
            rejection_reason=str(reason),
            distance_km=math.nan,
            latitude_deg=math.nan,
            longitude_deg=math.nan,
            altitude_km=math.nan,
            local_time_hr=math.nan,
            sigma_phi_rad=math.nan,
            s4=math.nan,
            mean_snr_vv=math.nan,
            q=math.nan,
            cos_alpha=math.nan,
            d_slope=math.nan,
            is_multivalued=False,
        )

    @staticmethod
    def _results_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a stable-schema results DataFrame."""
        if not rows:
            return pd.DataFrame(columns=list(_RESULT_COLUMNS))

        dataframe: pd.DataFrame = pd.DataFrame(rows)
        for column_name in _RESULT_COLUMNS:
            if column_name not in dataframe.columns:
                dataframe[column_name] = np.nan

        return dataframe.loc[:, list(_RESULT_COLUMNS)]

    @staticmethod
    def _empty_monthly_bins_dataframe() -> pd.DataFrame:
        """Return empty monthly-bin DataFrame with stable schema."""
        return pd.DataFrame(columns=list(_MONTHLY_BIN_COLUMNS))

    @staticmethod
    def _empty_l1_l2_pairs_dataframe() -> pd.DataFrame:
        """Return empty L1/L2 paired DataFrame with core expected columns."""
        return pd.DataFrame(
            columns=[
                "event_id",
                "leo_id",
                "gnss_id",
                "mid_time_key",
                "zonal_difference_deg",
                "lat_difference_deg",
                "altitude_difference_km",
                "distance_difference_km",
            ]
        )

    @staticmethod
    def _require_columns(
        dataframe: pd.DataFrame,
        required_columns: Iterable[str],
        context: str,
    ) -> None:
        """Raise a clear error if required columns are missing."""
        missing_columns: list[str] = [
            column_name
            for column_name in required_columns
            if column_name not in dataframe.columns
        ]
        if missing_columns:
            missing_text: str = ", ".join(missing_columns)
            raise ValueError(
                f"Missing required column(s) for {context}: {missing_text}."
            )

    @staticmethod
    def _validate_year(year: int) -> int:
        """Validate a UTC calendar year."""
        year_value: int = int(year)
        if year_value < 1995 or year_value > 2100:
            raise ValueError(f"year is outside expected range: {year_value}")
        return year_value

    @staticmethod
    def _coerce_date(value: date, name: str) -> date:
        """Coerce a date-like input to ``datetime.date``."""
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                value = value.astimezone(timezone.utc)
            return value.date()
        if isinstance(value, date):
            return value

        try:
            timestamp: pd.Timestamp = pd.to_datetime(value)
        except Exception as exc:
            raise ValueError(f"{name} must be date-like, got {value!r}.") from exc

        if pd.isna(timestamp):
            raise ValueError(f"{name} could not be converted to a valid date.")

        return timestamp.date()

    @staticmethod
    def _years_in_period(start: date, end: date) -> list[int]:
        """Return all UTC years touched by a [start, end) date interval."""
        if start >= end:
            return []

        years: list[int] = []
        current_year: int = start.year
        final_year: int = (end - pd.Timedelta(days=1)).year  # type: ignore[operator]
        for year_value in range(current_year, final_year + 1):
            years.append(int(year_value))
        return years

    @staticmethod
    def _window_in_period(window: SignalWindow, start: date, end: date) -> bool:
        """Return whether a window midpoint falls in [start, end)."""
        mid_time: datetime = getattr(window, "mid_time")
        if not isinstance(mid_time, datetime):
            return False

        if mid_time.tzinfo is None:
            mid_time_utc: datetime = mid_time.replace(tzinfo=timezone.utc)
        else:
            mid_time_utc = mid_time.astimezone(timezone.utc)

        mid_date: date = mid_time_utc.date()
        return start <= mid_date < end

    @staticmethod
    def _normalize_signal_name(value: str) -> str:
        """Normalize signal/constellation names for comparisons."""
        return str(value).strip().upper().replace("-", "_").replace(" ", "_")

    def _configured_map_bin_deg(self) -> float:
        """Return configured monthly-map bin size."""
        bin_size_deg: float = float(
            getattr(self.config, "map_bin_deg", constants.DEFAULT_MAP_BIN_DEG)
        )
        return bin_size_deg

    @staticmethod
    def _validate_bin_size(bin_size_deg: float) -> None:
        """Validate monthly-map bin size."""
        if not math.isfinite(bin_size_deg) or bin_size_deg <= 0.0:
            raise ValueError(f"map bin size must be finite and > 0, got {bin_size_deg}.")
        if abs((180.0 / bin_size_deg) - round(180.0 / bin_size_deg)) > 1.0e-9:
            raise ValueError(
                "Latitude span 180 degrees must be divisible by configured "
                f"bin size {bin_size_deg}."
            )
        if abs((360.0 / bin_size_deg) - round(360.0 / bin_size_deg)) > 1.0e-9:
            raise ValueError(
                "Longitude span 360 degrees must be divisible by configured "
                f"bin size {bin_size_deg}."
            )

    def _configured_window_seconds(self) -> int:
        """Return configured processing-window length for L1/L2 pairing keys."""
        window_seconds: int = int(
            getattr(
                self.config,
                "window_seconds",
                _DEFAULT_L1_L2_WINDOW_SECONDS,
            )
        )
        if window_seconds <= 0:
            raise ValueError(
                f"Configured window_seconds must be positive, got {window_seconds}."
            )
        return window_seconds


__all__ = ["RealDataExperiments"]
