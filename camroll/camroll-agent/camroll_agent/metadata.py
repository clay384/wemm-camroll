"""Original EXIF and offline administrative geocoding, with explicit provenance.

No filesystem timestamps, captions, QA annotations or network geocoders are used.
Coordinates refer to WGS84 EXIF GPS; EXIF times without offsets remain naive.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import math
import re
from pathlib import Path

METADATA_VERSION = "original-exif-offline-admin-v1"


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def extract_exif(image) -> dict:
    exif = image.getexif()
    try:
        detail = exif.get_ifd(34665)
    except (ValueError, KeyError, TypeError):
        detail = {}
    def value(tag):
        return detail.get(tag) or exif.get(tag)
    result = {"metadata_version": METADATA_VERSION, "captured_at": None,
              "capture_timezone": "unknown", "capture_precision": None,
              "capture_source": None, "gps": None,
              "exif_orientation": int(exif.get(274) or 1)}
    for tag, offset_tag, label in ((36867, 36881, "DateTimeOriginal"),
                                    (36868, 36882, "DateTimeDigitized"),
                                    (306, 36880, "DateTime")):
        raw = value(tag)
        if not isinstance(raw, str):
            continue
        try:
            captured = dt.datetime.strptime(raw.strip()[:19], "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
        offset = value(offset_tag)
        if isinstance(offset, str) and re.fullmatch(r"[+-]\d{2}:\d{2}", offset.strip()):
            offset = offset.strip()
            hour, minute = map(int, offset[1:].split(":"))
            if hour < 24 and minute < 60:
                delta = dt.timedelta(hours=hour, minutes=minute)
                captured = captured.replace(tzinfo=dt.timezone(delta if offset[0] == "+" else -delta))
                result["capture_timezone"] = offset
        result.update(captured_at=captured.isoformat(), capture_precision="second",
                      capture_source="exif:" + label, capture_raw=raw)
        break
    try:
        gps = exif.get_ifd(34853)
        def coordinate(number, reference, positive, negative):
            values = gps[number]
            ref = gps[reference]
            if isinstance(ref, bytes):
                ref = ref.decode("ascii")
            ref = str(ref).strip().upper()
            if ref not in (positive, negative) or len(values) != 3:
                raise ValueError("Invalid GPS hemisphere")
            degrees, minutes, seconds = map(float, values)
            if degrees < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
                raise ValueError("Invalid GPS DMS")
            return (degrees + minutes / 60 + seconds / 3600) * (-1 if ref == negative else 1)
        latitude = coordinate(2, 1, "N", "S")
        longitude = coordinate(4, 3, "E", "W")
        if math.isfinite(latitude) and math.isfinite(longitude) and -90 <= latitude <= 90 and -180 <= longitude <= 180:
            result["gps"] = {"latitude": latitude, "longitude": longitude,
                             "crs": "EPSG:4326", "source": "exif:GPSInfo"}
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, AttributeError):
        pass
    return result


class OfflineGeocoder:
    """Load PyGeoCN polygons once, falling back to GeoNames nearest settlement.

    Missing optional data is explicit; it never causes an online lookup. Neither
    polygon containment nor nearest-city output proves a venue/POI.
    """
    def __init__(self):
        self._china = None
        self._china_attempted = False
        self._cache = {}
        self.provenance = {"version": METADATA_VERSION, "network": False,
                           "pygeo_cn_version": _version("pygeo-cn"),
                           "geonames_package_version": _version("reverse_geocoder")}

    def _load_china(self):
        if self._china_attempted:
            return
        self._china_attempted = True
        try:
            import PyGeoCN
            import geopandas as gpd
            root = Path(PyGeoCN.__file__).parent
            frames, hashes = [], {}
            for level in ("province", "city", "district"):
                path = root / f"china_{level}.geojson"
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                frame = gpd.read_file(path)
                if frame.crs is not None:
                    frame = frame.to_crs("EPSG:4326")
                frames.append((level, frame))
            self._china = frames
            self.provenance["china_data_sha256"] = hashes
        except (ImportError, OSError, ValueError, AttributeError):
            self._china = None

    def lookup(self, gps):
        if not gps:
            return {"status": "missing_gps", "precision": "unknown", "source": None}
        lat, lon = gps.get("latitude"), gps.get("longitude")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            return {"status": "invalid_gps", "precision": "unknown", "source": None}
        key = (lat, lon)
        if key in self._cache:
            return dict(self._cache[key])
        result = {}
        if 15 <= lat <= 55 and 70 <= lon <= 140:
            self._load_china()
            if self._china:
                from shapely.geometry import Point
                point = Point(lon, lat)
                address = {}
                for level, frame in self._china:
                    matched = frame[frame.geometry.covers(point)]
                    if len(matched) == 1:
                        address[level] = str(matched.iloc[0]["name"])
                if address:
                    parts = list(dict.fromkeys(address.values()))
                    result = {"status": "resolved", "source": "PyGeoCN:Tianditu",
                              "precision": next(l for l in ("district", "city", "province") if l in address),
                              "label": "".join(parts), "country": "CN", **address}
        if not result:
            try:
                import reverse_geocoder as rg
                match = rg.search((lat, lon), mode=1)[0]
                a, b = math.radians(lat), math.radians(float(match["lat"]))
                delta_lat = b - a
                delta_lon = math.radians(float(match["lon"]) - lon)
                h = math.sin(delta_lat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(delta_lon / 2) ** 2
                distance = 6371.0088 * 2 * math.asin(min(1, math.sqrt(h)))
                result = {"status": "resolved", "source": "GeoNames", "precision": "nearest_city",
                          "label": ", ".join(dict.fromkeys(str(match[k]) for k in ("name", "admin1", "cc") if match.get(k))),
                          "city": match.get("name", ""), "province": match.get("admin1", ""),
                          "district": match.get("admin2", ""), "country": match.get("cc", ""),
                          "distance_km": round(distance, 3)}
            except (ImportError, OSError, ValueError, KeyError, IndexError):
                result = {"status": "unavailable", "source": None, "precision": "unknown"}
        result["poi_supported"] = False
        result["provenance"] = dict(self.provenance)
        self._cache[key] = dict(result)
        return result


def place_matches(query: str, place: dict) -> bool:
    """A filter can name a known administrative part, never invent a POI."""
    query = query.casefold().strip()
    if not query or place.get("status") != "resolved":
        return False
    values = [str(place.get(k) or "").casefold() for k in ("label", "country", "province", "city", "district")]
    return any(query in value for value in values if value)
