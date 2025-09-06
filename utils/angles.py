import math

__all__ = [
    'deg2rad',
    'rad2deg',
    'ring_angle_distance',
    'eci_to_geodetic_lat_deg',
]

def deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * math.pi / 180.0

def rad2deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi

def ring_angle_distance(x: float, y: float) -> float:
    """Return the minimal distance between two angles on a ring [0, 2*pi)."""
    diff = abs(x - y) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)

def eci_to_geodetic_lat_deg(r):
    """Approximate geodetic latitude from an ECI position vector."""
    x, y, z = r
    rho = math.sqrt(x * x + y * y + z * z)
    if rho == 0:
        return 0.0
    return rad2deg(math.asin(z / rho))
