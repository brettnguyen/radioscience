## src/geometry/magnetic_field.py
"""IGRF magnetic-field wrapper for COSMIC-2 BP geolocation geometry.

The back-propagation method described in the paper assumes equatorial
ionospheric irregularities are elongated along geomagnetic field lines. For
each candidate distance along the GNSS-LEO line of sight, the magnetic-field
direction at that candidate point defines the irregularity orientation used by
``BpPlaneBuilder``.

This module isolates all external IGRF/ppigrf details behind the stable project
interface:

    MagneticFieldModel.field_vector(time, position_m) -> Vector3
    MagneticFieldModel.field_unit(time, position_m) -> Vector3

Conventions:
    * Input positions are ECEF meters.
    * Input time is the observation/evaluation ``datetime``.
    * Returned magnetic-field vectors are ECEF-frame ``Vector3`` values.
    * ``field_vector`` returns magnetic-field components in the same magnitude
      units as the backend, normally nanotesla.
    * ``field_unit`` returns a dimensionless ECEF unit vector.
    * Only the direction is used by the BP method, consistent with
      ``config.yaml``: ``use_field_direction_only: true``.

The configured paper model is IGRF-13. The public constructor therefore
accepts only IGRF-13 aliases. The ppigrf backend does not expose a stable
project-level model-selection API across all versions, so this wrapper validates
the requested model name and then delegates coefficient evaluation to the
installed ppigrf package.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, ClassVar, Mapping

import numpy as np
from numpy.typing import NDArray

try:
    import ppigrf
except ImportError as exc:  # pragma: no cover - exercised only when dependency missing.
    ppigrf = None  # type: ignore[assignment]
    _PPIGRF_IMPORT_ERROR: ImportError | None = exc
else:
    _PPIGRF_IMPORT_ERROR = None

from src.core.types import Vector3
from src.geometry.coordinates import CoordinateTransformer


_ZERO_FIELD_EPS_NT: float = 1.0e-12


class MagneticFieldModelError(ValueError):
    """Raised when magnetic-field evaluation fails."""


class MagneticFieldModel:
    """IGRF-13 magnetic-field model wrapper.

    Args:
        model_name: Magnetic-field model name. For strict reproduction of the
            paper, only ``"IGRF-13"`` and close aliases are supported.

    The class internally owns a ``CoordinateTransformer`` because the design
    interface intentionally exposes only ``MagneticFieldModel(model_name)``.
    """

    _SUPPORTED_MODEL_ALIASES: ClassVar[set[str]] = {
        "IGRF-13",
        "IGRF13",
        "IGRF_13",
    }

    def __init__(self, model_name: str = "IGRF-13") -> None:
        """Initialize the magnetic-field model.

        Args:
            model_name: Requested magnetic-field model. Must identify IGRF-13.

        Raises:
            ValueError: If the model name is unsupported.
            ImportError: If ppigrf is not installed.
        """
        if ppigrf is None:
            raise ImportError(
                "The ppigrf package is required for magnetic-field evaluation. "
                "Install it with the project requirements."
            ) from _PPIGRF_IMPORT_ERROR

        self.model_name: str = self._validate_model_name(model_name)
        self.transformer: CoordinateTransformer = CoordinateTransformer()

    def field_vector(self, time: datetime, position_m: Vector3) -> Vector3:
        """Evaluate the magnetic-field vector at an ECEF position.

        Args:
            time: Observation/evaluation time. Naive datetimes are interpreted
                as UTC; aware datetimes are converted to UTC.
            position_m: Candidate point in ECEF meters.

        Returns:
            Magnetic-field vector in the ECEF frame as ``Vector3``. The
            component magnitude is normally in nanotesla because ppigrf returns
            nanotesla, but downstream BP-plane construction uses only direction.

        Raises:
            TypeError: If ``time`` or ``position_m`` has an invalid type.
            ValueError: If inputs or evaluated field components are invalid.
            MagneticFieldModelError: If ppigrf evaluation fails.
        """
        evaluation_time: datetime = self._normalize_time(time)
        self._validate_position(position_m)

        lat_deg, lon_deg, alt_km = self.transformer.ecef_to_geodetic(position_m)

        b_east, b_north, b_up = self._evaluate_ppigrf_enu(
            time=evaluation_time,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            alt_km=alt_km,
        )

        field_ecef: Vector3 = self._enu_to_ecef(
            b_east=b_east,
            b_north=b_north,
            b_up=b_up,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
        )

        field_array: NDArray[np.float64] = field_ecef.to_array()
        field_norm: float = float(np.linalg.norm(field_array))
        if not math.isfinite(field_norm) or field_norm <= _ZERO_FIELD_EPS_NT:
            raise MagneticFieldModelError(
                "IGRF magnetic-field vector is zero, near-zero, or non-finite."
            )

        return field_ecef

    def field_unit(self, time: datetime, position_m: Vector3) -> Vector3:
        """Evaluate the unit magnetic-field direction at an ECEF position.

        Args:
            time: Observation/evaluation time.
            position_m: Candidate point in ECEF meters.

        Returns:
            Dimensionless ECEF unit vector in the magnetic-field direction.

        Raises:
            TypeError: If inputs have invalid types.
            ValueError: If the magnetic-field vector cannot be normalized.
            MagneticFieldModelError: If backend evaluation fails.
        """
        try:
            return self.field_vector(time=time, position_m=position_m).unit()
        except ValueError as exc:
            raise MagneticFieldModelError(
                "Failed to compute unit magnetic-field direction."
            ) from exc

    @classmethod
    def _validate_model_name(cls, model_name: str) -> str:
        """Validate and canonicalize a model name.

        Args:
            model_name: User/configuration supplied model name.

        Returns:
            Canonical model name ``"IGRF-13"``.

        Raises:
            ValueError: If the model name is empty or unsupported.
        """
        normalized_model_name: str = str(model_name).strip()
        if not normalized_model_name:
            raise ValueError("Magnetic-field model name must be non-empty.")

        normalized_upper: str = normalized_model_name.upper()
        supported_upper: set[str] = {
            model_alias.upper() for model_alias in cls._SUPPORTED_MODEL_ALIASES
        }

        if normalized_upper not in supported_upper:
            raise ValueError(
                "Unsupported magnetic-field model "
                f"{model_name!r}. This reproduction supports only IGRF-13."
            )

        return "IGRF-13"

    @staticmethod
    def _normalize_time(time: datetime) -> datetime:
        """Normalize an evaluation time for ppigrf.

        Args:
            time: Datetime supplied by the caller.

        Returns:
            Naive UTC ``datetime``. Naive input is assumed to already be UTC.

        Raises:
            TypeError: If ``time`` is not a ``datetime``.
            ValueError: If the timestamp is not finite.
        """
        if not isinstance(time, datetime):
            raise TypeError(f"time must be datetime, got {type(time).__name__}.")

        if time.tzinfo is None:
            normalized_time: datetime = time
        else:
            normalized_time = time.astimezone(timezone.utc).replace(tzinfo=None)

        timestamp: float = float(normalized_time.replace(tzinfo=timezone.utc).timestamp())
        if not math.isfinite(timestamp):
            raise ValueError(f"time has a non-finite timestamp: {time!r}.")

        return normalized_time

    @staticmethod
    def _validate_position(position_m: Vector3) -> None:
        """Validate an ECEF position vector.

        Args:
            position_m: Position expected to be ``Vector3`` with finite meters.

        Raises:
            TypeError: If ``position_m`` is not a ``Vector3``.
            ValueError: If components are non-finite.
        """
        if not isinstance(position_m, Vector3):
            raise TypeError(
                "position_m must be a Vector3, "
                f"got {type(position_m).__name__}."
            )

        position_array: NDArray[np.float64] = position_m.to_array()
        if position_array.shape != (3,) or not np.all(np.isfinite(position_array)):
            raise ValueError("position_m must contain finite ECEF coordinates.")

    def _evaluate_ppigrf_enu(
        self,
        time: datetime,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
    ) -> tuple[float, float, float]:
        """Evaluate ppigrf and return local East/North/Up components.

        The ppigrf public function convention is normally:

            Be, Bn, Bu = ppigrf.igrf(lon, lat, h, date)

        where longitude and latitude are in degrees and altitude ``h`` is in
        kilometers above the WGS84 ellipsoid. Components are east, north, and
        up in nanotesla.

        Args:
            time: Naive UTC evaluation datetime.
            lat_deg: Geodetic latitude in degrees.
            lon_deg: East-positive longitude in degrees.
            alt_km: Altitude in kilometers above the WGS84 ellipsoid.

        Returns:
            Tuple ``(B_east, B_north, B_up)``.

        Raises:
            MagneticFieldModelError: If ppigrf evaluation or component parsing
                fails.
        """
        self._validate_geodetic_inputs(lat_deg=lat_deg, lon_deg=lon_deg, alt_km=alt_km)

        try:
            raw_result: Any = ppigrf.igrf(lon_deg, lat_deg, alt_km, time)  # type: ignore[union-attr]
        except Exception as first_exc:
            decimal_year: float = self._decimal_year(time)
            try:
                raw_result = ppigrf.igrf(lon_deg, lat_deg, alt_km, decimal_year)  # type: ignore[union-attr]
            except Exception as second_exc:
                raise MagneticFieldModelError(
                    "ppigrf.igrf failed for both datetime and decimal-year "
                    "evaluation inputs. Check ppigrf installation/API and "
                    f"candidate geodetic point lat={lat_deg:.6f}, "
                    f"lon={lon_deg:.6f}, alt_km={alt_km:.3f}."
                ) from second_exc

            # Preserve the datetime failure as context where possible without
            # changing the public exception type.
            if raw_result is None:
                raise MagneticFieldModelError(
                    "ppigrf.igrf returned None after datetime evaluation failed "
                    f"with: {first_exc!r}"
                )

        b_east, b_north, b_up = self._parse_ppigrf_result(raw_result)
        return b_east, b_north, b_up

    @staticmethod
    def _validate_geodetic_inputs(lat_deg: float, lon_deg: float, alt_km: float) -> None:
        """Validate geodetic inputs passed to ppigrf.

        Args:
            lat_deg: Latitude in degrees.
            lon_deg: Longitude in degrees.
            alt_km: Altitude in kilometers.

        Raises:
            MagneticFieldModelError: If values are invalid.
        """
        latitude: float = float(lat_deg)
        longitude: float = float(lon_deg)
        altitude: float = float(alt_km)

        if not math.isfinite(latitude):
            raise MagneticFieldModelError("Latitude for IGRF evaluation is non-finite.")
        if not math.isfinite(longitude):
            raise MagneticFieldModelError("Longitude for IGRF evaluation is non-finite.")
        if not math.isfinite(altitude):
            raise MagneticFieldModelError("Altitude for IGRF evaluation is non-finite.")
        if latitude < -90.0 or latitude > 90.0:
            raise MagneticFieldModelError(
                f"Latitude for IGRF evaluation must be in [-90, 90], got {latitude}."
            )

    @classmethod
    def _parse_ppigrf_result(cls, raw_result: Any) -> tuple[float, float, float]:
        """Parse ppigrf output into scalar East/North/Up components.

        Args:
            raw_result: Return value from ``ppigrf.igrf``.

        Returns:
            Tuple ``(B_east, B_north, B_up)``.

        Raises:
            MagneticFieldModelError: If the result cannot be parsed or contains
                non-finite components.
        """
        if isinstance(raw_result, Mapping):
            b_east = cls._mapping_component(
                raw_result,
                accepted_keys=("Be", "B_east", "east", "E", "e"),
            )
            b_north = cls._mapping_component(
                raw_result,
                accepted_keys=("Bn", "B_north", "north", "N", "n"),
            )
            b_up = cls._mapping_component(
                raw_result,
                accepted_keys=("Bu", "B_up", "up", "U", "u"),
            )
        else:
            try:
                result_sequence: tuple[Any, ...] = tuple(raw_result)
            except TypeError as exc:
                raise MagneticFieldModelError(
                    "ppigrf.igrf returned an object that is neither a mapping nor "
                    "an iterable component tuple."
                ) from exc

            if len(result_sequence) < 3:
                raise MagneticFieldModelError(
                    "ppigrf.igrf returned fewer than three magnetic components."
                )

            b_east = cls._scalar_component(result_sequence[0], "B_east")
            b_north = cls._scalar_component(result_sequence[1], "B_north")
            b_up = cls._scalar_component(result_sequence[2], "B_up")

        for component_name, component_value in {
            "B_east": b_east,
            "B_north": b_north,
            "B_up": b_up,
        }.items():
            if not math.isfinite(component_value):
                raise MagneticFieldModelError(
                    f"{component_name} from ppigrf is non-finite."
                )

        return b_east, b_north, b_up

    @classmethod
    def _mapping_component(
        cls,
        mapping: Mapping[Any, Any],
        accepted_keys: tuple[str, ...],
    ) -> float:
        """Extract a scalar component from a mapping-like backend result.

        Args:
            mapping: Mapping returned by an IGRF backend.
            accepted_keys: Possible component names.

        Returns:
            Scalar component value.

        Raises:
            MagneticFieldModelError: If none of the keys are present or the
                value is not scalar.
        """
        direct_mapping: dict[str, Any] = {str(key): value for key, value in mapping.items()}
        lower_mapping: dict[str, Any] = {
            str(key).lower(): value for key, value in mapping.items()
        }

        for key in accepted_keys:
            if key in direct_mapping:
                return cls._scalar_component(direct_mapping[key], key)
            lower_key: str = key.lower()
            if lower_key in lower_mapping:
                return cls._scalar_component(lower_mapping[lower_key], key)

        raise MagneticFieldModelError(
            "Could not find expected magnetic component in backend mapping. "
            f"Accepted keys: {accepted_keys}."
        )

    @staticmethod
    def _scalar_component(value: Any, name: str) -> float:
        """Convert a backend component to a finite scalar float.

        Args:
            value: Scalar or scalar-like array value.
            name: Component name for error messages.

        Returns:
            Float scalar.

        Raises:
            MagneticFieldModelError: If the value is empty, non-scalar, or
                cannot be converted to float.
        """
        try:
            array: NDArray[Any] = np.asarray(value)
        except Exception as exc:
            raise MagneticFieldModelError(
                f"Could not convert {name} from ppigrf to an array."
            ) from exc

        if array.size != 1:
            raise MagneticFieldModelError(
                f"{name} from ppigrf must be scalar for scalar input; "
                f"got size {array.size}."
            )

        try:
            scalar_value: float = float(np.ravel(array)[0])
        except (TypeError, ValueError) as exc:
            raise MagneticFieldModelError(
                f"Could not convert {name} from ppigrf to float."
            ) from exc

        if not math.isfinite(scalar_value):
            raise MagneticFieldModelError(f"{name} from ppigrf is non-finite.")

        return scalar_value

    def _enu_to_ecef(
        self,
        b_east: float,
        b_north: float,
        b_up: float,
        lat_deg: float,
        lon_deg: float,
    ) -> Vector3:
        """Convert local East/North/Up magnetic components to ECEF.

        Args:
            b_east: East component from ppigrf.
            b_north: North component from ppigrf.
            b_up: Up component from ppigrf.
            lat_deg: Geodetic latitude in degrees.
            lon_deg: East-positive longitude in degrees.

        Returns:
            ECEF-frame magnetic-field vector.

        Raises:
            MagneticFieldModelError: If the converted vector is invalid.
        """
        east_unit, north_unit, up_unit = self.transformer.enu_basis(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
        )

        field_array: NDArray[np.float64] = (
            float(b_east) * east_unit.to_array()
            + float(b_north) * north_unit.to_array()
            + float(b_up) * up_unit.to_array()
        )

        if field_array.shape != (3,) or not np.all(np.isfinite(field_array)):
            raise MagneticFieldModelError(
                "Converted ECEF magnetic-field vector is invalid."
            )

        return Vector3(
            x=float(field_array[0]),
            y=float(field_array[1]),
            z=float(field_array[2]),
        )

    @staticmethod
    def _decimal_year(time: datetime) -> float:
        """Convert a datetime to decimal year for backend fallback calls.

        Args:
            time: Naive UTC datetime.

        Returns:
            Decimal year.

        Raises:
            MagneticFieldModelError: If conversion fails.
        """
        if time.tzinfo is not None:
            time_utc: datetime = time.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            time_utc = time

        year_start: datetime = datetime(time_utc.year, 1, 1)
        next_year_start: datetime = datetime(time_utc.year + 1, 1, 1)

        elapsed_seconds: float = (time_utc - year_start).total_seconds()
        year_seconds: float = (next_year_start - year_start).total_seconds()

        if not math.isfinite(elapsed_seconds) or not math.isfinite(year_seconds):
            raise MagneticFieldModelError("Could not compute finite decimal year.")
        if year_seconds <= 0.0:
            raise MagneticFieldModelError("Invalid calendar year length.")

        return float(time_utc.year) + elapsed_seconds / year_seconds


__all__ = ["MagneticFieldModel", "MagneticFieldModelError"]
