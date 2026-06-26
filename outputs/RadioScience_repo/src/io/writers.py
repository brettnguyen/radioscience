## src/io/writers.py
"""Persistence utilities for COSMIC-2 back-propagation reproduction outputs.

This module implements the output layer for the project. It serializes already
computed geolocation tables, monthly bins, L1/L2 comparison tables, synthetic
experiment outputs, and diagnostic curves such as ``V(L)`` and ``D(L_mf)``.

The writer intentionally performs no scientific computation, quality control,
filtering, binning, plotting, data loading, or geolocation logic. If a caller
invokes a write method, this module assumes the caller has already decided that
the output should be saved according to the runtime configuration.

Supported public API:
    ResultWriter(output_dir)
    ResultWriter.write_results(results, name)
    ResultWriter.write_dataframe(df, name)
    ResultWriter.write_diagnostics(obj, name)
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.core.types import BpCurve, DCurve, GeolocationResult


_TABULAR_SUFFIXES: frozenset[str] = frozenset({".parquet", ".csv", ".json", ".jsonl"})
_DIAGNOSTIC_SUFFIXES: frozenset[str] = frozenset(
    {".npz", ".json", ".parquet", ".csv", ".jsonl"}
)


class ResultWriter:
    """Write tabular and diagnostic outputs for the reproduction pipeline.

    Args:
        output_dir: Base directory where outputs are written. Relative output
            names are resolved under this directory. The directory is created at
            construction time.

    The writer supports nested output names such as
    ``"diagnostics/event_001_v_curve.npz"`` and automatically creates parent
    directories before writing.
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize the writer and create the output directory.

        Args:
            output_dir: Output directory path from application configuration.

        Raises:
            ValueError: If ``output_dir`` is empty.
            OSError: If the directory cannot be created.
        """
        self.output_dir: Path = Path(output_dir).expanduser()
        if str(self.output_dir).strip() == "":
            raise ValueError("output_dir must be a non-empty path.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_results(self, results: list[GeolocationResult], name: str) -> Path:
        """Write a list of per-window geolocation results.

        The method preserves accepted and rejected cases exactly as supplied by
        the caller. It does not filter by QC status, local time, signal, or
        multi-valued flag.

        Args:
            results: Geolocation result records.
            name: Output filename relative to ``output_dir``. If no suffix is
                provided, ``.parquet`` is appended.

        Returns:
            Path to the written output file.

        Raises:
            ValueError: If an unsupported output suffix is requested.
            TypeError: If ``results`` contains an object without ``to_dict``.
        """
        rows: list[dict[str, Any]] = []

        for index, result in enumerate(results):
            if not isinstance(result, GeolocationResult):
                raise TypeError(
                    "write_results expects a list of GeolocationResult objects; "
                    f"item {index} has type {type(result).__name__}."
                )
            rows.append(result.to_dict())

        dataframe: pd.DataFrame = pd.DataFrame(rows)
        return self.write_dataframe(dataframe, name)

    def write_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        """Write a pandas DataFrame using a suffix-selected format.

        Supported formats:
            * ``.parquet``: ``DataFrame.to_parquet(index=False)``
            * ``.csv``: ``DataFrame.to_csv(index=False)``
            * ``.json``: records-oriented JSON with ISO datetime formatting
            * ``.jsonl``: JSON lines with ISO datetime formatting

        If ``name`` has no suffix, ``.parquet`` is appended.

        Args:
            df: DataFrame to serialize.
            name: Output filename relative to ``output_dir`` unless absolute.

        Returns:
            Path to the written output file.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If the suffix is unsupported.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"write_dataframe expects pandas.DataFrame, got {type(df).__name__}."
            )

        path: Path = self._resolve_path(name=name, default_suffix=".parquet")
        suffix: str = path.suffix.lower()

        if suffix not in _TABULAR_SUFFIXES:
            raise ValueError(
                f"Unsupported output format for DataFrame: {suffix!r}. "
                f"Supported formats are {sorted(_TABULAR_SUFFIXES)}."
            )

        self._ensure_parent(path)

        if suffix == ".parquet":
            df.to_parquet(path, index=False)
        elif suffix == ".csv":
            df.to_csv(path, index=False)
        elif suffix == ".json":
            df.to_json(path, orient="records", date_format="iso")
        elif suffix == ".jsonl":
            df.to_json(path, orient="records", lines=True, date_format="iso")
        else:
            raise ValueError(f"Unsupported output format for DataFrame: {suffix!r}.")

        return path

    def write_diagnostics(self, obj: Any, name: str) -> Path:
        """Write diagnostic objects such as BP curves and synthetic outputs.

        Supported diagnostic formats:
            * ``.npz``: compressed NumPy archive, default for diagnostics.
            * ``.json``: recursively JSON-safe representation.
            * ``.parquet``, ``.csv``, ``.jsonl``: for DataFrame diagnostics.

        If ``obj`` is a DataFrame, this method delegates to
        ``write_dataframe`` unless ``name`` explicitly ends in ``.npz``. For
        ``.npz`` DataFrame diagnostics, columns are stored as arrays.

        If ``name`` has no suffix, ``.npz`` is appended for non-DataFrame
        diagnostics and ``.parquet`` is appended for DataFrames.

        Args:
            obj: Diagnostic object to serialize. Supported objects include
                ``BpCurve``, ``DCurve``, ``dict``, NumPy arrays, scalars,
                dataclasses, lists/tuples, and pandas DataFrames.
            name: Output filename relative to ``output_dir`` unless absolute.

        Returns:
            Path to the written output file.

        Raises:
            ValueError: If the suffix is unsupported.
            TypeError: If JSON serialization encounters an unsupported type.
        """
        default_suffix: str = ".parquet" if isinstance(obj, pd.DataFrame) else ".npz"
        path: Path = self._resolve_path(name=name, default_suffix=default_suffix)
        suffix: str = path.suffix.lower()

        if suffix not in _DIAGNOSTIC_SUFFIXES:
            raise ValueError(
                f"Unsupported output format for diagnostics: {suffix!r}. "
                f"Supported formats are {sorted(_DIAGNOSTIC_SUFFIXES)}."
            )

        if isinstance(obj, pd.DataFrame) and suffix != ".npz":
            return self.write_dataframe(obj, str(path if path.is_absolute() else name))

        self._ensure_parent(path)

        if suffix == ".npz":
            mapping: dict[str, Any] = self._diagnostic_to_npz_mapping(obj)
            self._write_npz(mapping=mapping, path=path)
        elif suffix == ".json":
            json_safe_object: Any = self._json_safe(obj)
            with path.open("w", encoding="utf-8") as stream:
                json.dump(
                    json_safe_object,
                    stream,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=True,
                )
                stream.write("\n")
        elif suffix in _TABULAR_SUFFIXES:
            if isinstance(obj, pd.DataFrame):
                return self.write_dataframe(obj, str(path if path.is_absolute() else name))
            dataframe: pd.DataFrame = pd.DataFrame([self._json_safe(obj)])
            return self.write_dataframe(dataframe, str(path if path.is_absolute() else name))
        else:
            raise ValueError(f"Unsupported output format for diagnostics: {suffix!r}.")

        return path

    def _resolve_path(self, name: str, default_suffix: str) -> Path:
        """Resolve an output name to an output path.

        Args:
            name: User-provided output name.
            default_suffix: Suffix to append if ``name`` has none.

        Returns:
            Absolute or relative path to write.

        Raises:
            ValueError: If ``name`` or ``default_suffix`` is invalid.
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}.")

        stripped_name: str = name.strip()
        if not stripped_name:
            raise ValueError("Output name must be non-empty.")

        if not default_suffix.startswith("."):
            raise ValueError(
                f"default_suffix must start with '.', got {default_suffix!r}."
            )

        path: Path = Path(stripped_name).expanduser()
        if path.suffix == "":
            path = path.with_suffix(default_suffix)

        if path.is_absolute():
            return path

        return self.output_dir / path

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        """Create parent directories for an output path."""
        path.parent.mkdir(parents=True, exist_ok=True)

    def _diagnostic_to_npz_mapping(self, obj: Any) -> dict[str, Any]:
        """Convert a diagnostic object into an NPZ-compatible mapping.

        Args:
            obj: Diagnostic object.

        Returns:
            Mapping from archive key to array/scalar values.
        """
        if isinstance(obj, BpCurve):
            return self._curve_to_dict(obj)
        if isinstance(obj, DCurve):
            return self._curve_to_dict(obj)
        if isinstance(obj, pd.DataFrame):
            return self._dataframe_to_npz_mapping(obj)
        if isinstance(obj, Mapping):
            flattened: dict[str, Any] = {}
            self._flatten_mapping_for_npz(obj, prefix="", output=flattened)
            return flattened
        if isinstance(obj, np.ndarray):
            return {"array": obj}
        if is_dataclass(obj) and not isinstance(obj, type):
            flattened_dataclass: dict[str, Any] = {}
            self._flatten_mapping_for_npz(
                asdict(obj), prefix="", output=flattened_dataclass
            )
            return flattened_dataclass

        return {"value": self._to_npz_value(obj)}

    @staticmethod
    def _curve_to_dict(obj: BpCurve | DCurve) -> dict[str, Any]:
        """Convert a BP-related curve dataclass to a serializable dictionary.

        Args:
            obj: ``BpCurve`` or ``DCurve``.

        Returns:
            Mapping containing all curve fields needed for reproduction.
        """
        if isinstance(obj, BpCurve):
            return {
                "l_mf_km": np.asarray(obj.l_mf_km, dtype=np.float64),
                "distances_km": np.asarray(obj.distances_km, dtype=np.float64),
                "v_raw": np.asarray(obj.v_raw, dtype=np.float64),
                "v_smooth": np.asarray(obj.v_smooth, dtype=np.float64),
                "l0_km": np.asarray(obj.l0_km, dtype=np.float64),
                "v0": np.asarray(obj.v0, dtype=np.float64),
                "l1_km": np.asarray(obj.l1_km, dtype=np.float64),
                "l2_km": np.asarray(obj.l2_km, dtype=np.float64),
                "v1": np.asarray(obj.v1, dtype=np.float64),
                "v2": np.asarray(obj.v2, dtype=np.float64),
                "q": np.asarray(obj.q, dtype=np.float64),
                "has_valid_minimum": np.asarray(obj.has_valid_minimum, dtype=np.bool_),
            }

        if isinstance(obj, DCurve):
            return {
                "l_mf_km": np.asarray(obj.l_mf_km, dtype=np.float64),
                "d_km": np.asarray(obj.d_km, dtype=np.float64),
                "l0_km": np.asarray(obj.l0_km, dtype=np.float64),
                "q": np.asarray(obj.q, dtype=np.float64),
                "cos_alpha": np.asarray(obj.cos_alpha, dtype=np.float64),
                "zero_crossings_km": np.asarray(
                    obj.zero_crossings_km, dtype=np.float64
                ),
                "is_multivalued": np.asarray(obj.is_multivalued, dtype=np.bool_),
            }

        raise TypeError(
            f"_curve_to_dict expects BpCurve or DCurve, got {type(obj).__name__}."
        )

    @staticmethod
    def _dataframe_to_npz_mapping(df: pd.DataFrame) -> dict[str, Any]:
        """Convert a DataFrame into an NPZ-compatible mapping.

        Args:
            df: DataFrame diagnostic object.

        Returns:
            Mapping with one archive entry per column plus column metadata.
        """
        mapping: dict[str, Any] = {
            "__columns_json__": json.dumps([str(column) for column in df.columns]),
            "__index__": df.index.to_numpy(),
        }

        used_keys: set[str] = set(mapping)
        for column in df.columns:
            key: str = ResultWriter._unique_key(
                ResultWriter._sanitize_npz_key(f"column__{column}"), used_keys
            )
            used_keys.add(key)

            series: pd.Series = df[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                mapping[key] = series.astype("datetime64[ns]").astype(str).to_numpy()
            else:
                mapping[key] = ResultWriter._to_npz_value(series.to_numpy())

        return mapping

    def _flatten_mapping_for_npz(
        self,
        mapping: Mapping[Any, Any],
        prefix: str,
        output: dict[str, Any],
    ) -> None:
        """Flatten a nested mapping into archive keys.

        Nested dictionaries are flattened with ``__`` separators. Non-mapping
        leaves are converted to NPZ-compatible values.

        Args:
            mapping: Mapping to flatten.
            prefix: Current key prefix.
            output: Mutable destination mapping.
        """
        for raw_key, value in mapping.items():
            key_part: str = self._sanitize_npz_key(str(raw_key))
            key: str = key_part if not prefix else f"{prefix}__{key_part}"

            if isinstance(value, Mapping):
                self._flatten_mapping_for_npz(value, prefix=key, output=output)
            elif is_dataclass(value) and not isinstance(value, type):
                self._flatten_mapping_for_npz(asdict(value), prefix=key, output=output)
            else:
                unique_key: str = self._unique_key(key, set(output))
                output[unique_key] = self._to_npz_value(value)

    @staticmethod
    def _write_npz(mapping: dict[str, Any], path: Path) -> None:
        """Write an NPZ archive from a mapping.

        Args:
            mapping: Archive entries.
            path: Destination path.
        """
        if not mapping:
            mapping = {"__empty__": np.asarray(True, dtype=np.bool_)}

        archive_mapping: dict[str, Any] = {}
        used_keys: set[str] = set()

        for raw_key, value in mapping.items():
            key: str = ResultWriter._unique_key(
                ResultWriter._sanitize_npz_key(str(raw_key)), used_keys
            )
            used_keys.add(key)
            archive_mapping[key] = ResultWriter._to_npz_value(value)

        np.savez_compressed(path, **archive_mapping)

    @staticmethod
    def _to_npz_value(value: Any) -> Any:
        """Convert a Python/scientific value to something ``np.savez`` accepts.

        Args:
            value: Value to convert.

        Returns:
            NumPy array or scalar suitable for ``np.savez_compressed``.
        """
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return np.asarray(json.dumps(ResultWriter._json_safe(value)))
            return value

        if isinstance(value, pd.Series):
            return ResultWriter._to_npz_value(value.to_numpy())

        if isinstance(value, pd.Index):
            return ResultWriter._to_npz_value(value.to_numpy())

        if isinstance(value, np.generic):
            return np.asarray(value)

        if isinstance(value, (str, bytes, bool, int, float, complex)):
            return np.asarray(value)

        if value is None:
            return np.asarray("null")

        if isinstance(value, (datetime, date, pd.Timestamp)):
            return np.asarray(value.isoformat())

        if isinstance(value, Path):
            return np.asarray(str(value))

        if isinstance(value, (list, tuple)):
            try:
                array_value: np.ndarray = np.asarray(value)
            except (TypeError, ValueError):
                return np.asarray(json.dumps(ResultWriter._json_safe(value)))

            if array_value.dtype != object:
                return array_value

            return np.asarray(json.dumps(ResultWriter._json_safe(value)))

        if is_dataclass(value) and not isinstance(value, type):
            return np.asarray(json.dumps(ResultWriter._json_safe(asdict(value))))

        return np.asarray(json.dumps(ResultWriter._json_safe(value)))

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Recursively convert scientific Python objects to JSON-safe values.

        Args:
            obj: Object to convert.

        Returns:
            JSON-serializable representation.

        Raises:
            TypeError: If the object cannot be converted.
        """
        if isinstance(obj, GeolocationResult):
            return ResultWriter._json_safe(obj.to_dict())

        if isinstance(obj, (BpCurve, DCurve)):
            return ResultWriter._json_safe(ResultWriter._curve_to_dict(obj))

        if isinstance(obj, pd.DataFrame):
            return ResultWriter._json_safe(obj.to_dict(orient="records"))

        if isinstance(obj, pd.Series):
            return ResultWriter._json_safe(obj.to_list())

        if isinstance(obj, pd.Index):
            return ResultWriter._json_safe(obj.to_list())

        if isinstance(obj, np.ndarray):
            if np.iscomplexobj(obj):
                return {
                    "__complex_ndarray__": True,
                    "real": ResultWriter._json_safe(np.real(obj)),
                    "imag": ResultWriter._json_safe(np.imag(obj)),
                    "shape": list(obj.shape),
                }
            return obj.tolist()

        if isinstance(obj, np.floating):
            value: float = float(obj)
            return value

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, np.complexfloating):
            complex_value: complex = complex(obj)
            return {"__complex__": True, "real": complex_value.real, "imag": complex_value.imag}

        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}

        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()

        if isinstance(obj, Path):
            return str(obj)

        if obj is pd.NaT:
            return None

        if obj is None:
            return None

        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return obj
            return float(obj)

        if isinstance(obj, (str, int, bool)):
            return obj

        if isinstance(obj, Mapping):
            return {
                str(key): ResultWriter._json_safe(value)
                for key, value in obj.items()
            }

        if isinstance(obj, (list, tuple, set, frozenset)):
            return [ResultWriter._json_safe(value) for value in obj]

        if is_dataclass(obj) and not isinstance(obj, type):
            return ResultWriter._json_safe(asdict(obj))

        try:
            json.dumps(obj)
        except TypeError as exc:
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable "
                "by ResultWriter."
            ) from exc

        return obj

    @staticmethod
    def _sanitize_npz_key(key: str) -> str:
        """Sanitize an NPZ archive key.

        Args:
            key: Raw key.

        Returns:
            Safe, non-empty key string.
        """
        sanitized: str = key.strip().replace("/", "_").replace("\\", "_")
        sanitized = sanitized.replace("\n", "_").replace("\r", "_").replace("\t", "_")
        if not sanitized:
            return "value"
        return sanitized

    @staticmethod
    def _unique_key(key: str, used_keys: set[str]) -> str:
        """Return a unique key not present in ``used_keys``.

        Args:
            key: Desired key.
            used_keys: Existing keys.

        Returns:
            Unique key string.
        """
        if key not in used_keys:
            return key

        index: int = 1
        while f"{key}_{index}" in used_keys:
            index += 1
        return f"{key}_{index}"


__all__ = ["ResultWriter"]
