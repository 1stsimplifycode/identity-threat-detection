"""Geography helpers: distance calculation and synthetic city pools.

City coordinates below are real-world reference points used only to give
generated geo fields plausible, internally-consistent latitude/longitude
values -- they do not imply any claim about real user populations or real
traffic at these locations.
"""
from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# country -> list of (city, lat, lon) -- used for a user's home location.
HOME_CITY_POOL: dict[str, list[tuple[str, float, float]]] = {
    "US": [
        ("New York", 40.7128, -74.0060),
        ("San Francisco", 37.7749, -122.4194),
        ("Austin", 30.2672, -97.7431),
        ("Chicago", 41.8781, -87.6298),
    ],
    "GB": [("London", 51.5074, -0.1278), ("Manchester", 53.4808, -2.2426)],
    "DE": [("Berlin", 52.5200, 13.4050), ("Munich", 48.1351, 11.5820)],
    "IN": [
        ("Bengaluru", 12.9716, 77.5946),
        ("Pune", 18.5204, 73.8567),
        ("Hyderabad", 17.3850, 78.4867),
    ],
    "CA": [("Toronto", 43.6532, -79.3832), ("Vancouver", 49.2827, -123.1207)],
    "AU": [("Sydney", -33.8688, 151.2093), ("Melbourne", -37.8136, 144.9631)],
    "BR": [("Sao Paulo", -23.5505, -46.6333)],
    "JP": [("Tokyo", 35.6762, 139.6503)],
    "SG": [("Singapore", 1.3521, 103.8198)],
    "FR": [("Paris", 48.8566, 2.3522)],
}

# (city, lat, lon, country) -- wider pool spanning multiple continents, used
# for legitimate business travel *and* as candidate attack destinations, so
# "a login from an unusual city" is never by itself a giveaway of an attack.
GLOBAL_CITY_POOL: list[tuple[str, float, float, str]] = [
    ("New York", 40.7128, -74.0060, "US"),
    ("London", 51.5074, -0.1278, "GB"),
    ("Berlin", 52.5200, 13.4050, "DE"),
    ("Bengaluru", 12.9716, 77.5946, "IN"),
    ("Toronto", 43.6532, -79.3832, "CA"),
    ("Sydney", -33.8688, 151.2093, "AU"),
    ("Sao Paulo", -23.5505, -46.6333, "BR"),
    ("Tokyo", 35.6762, 139.6503, "JP"),
    ("Singapore", 1.3521, 103.8198, "SG"),
    ("Paris", 48.8566, 2.3522, "FR"),
    ("Moscow", 55.7558, 37.6173, "RU"),
    ("Lagos", 6.5244, 3.3792, "NG"),
    ("Dubai", 25.2048, 55.2708, "AE"),
    ("Mexico City", 19.4326, -99.1332, "MX"),
    ("Johannesburg", -26.2041, 28.0473, "ZA"),
    ("Jakarta", -6.2088, 106.8456, "ID"),
    ("Seoul", 37.5665, 126.9780, "KR"),
    ("Cairo", 30.0444, 31.2357, "EG"),
]
