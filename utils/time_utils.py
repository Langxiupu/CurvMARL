from datetime import datetime, timedelta, timezone

__all__ = ['parse_iso_utc', 'add_seconds']

def parse_iso_utc(iso: str) -> datetime:
    """Parse an ISO 8601 UTC timestamp into a datetime object."""
    if iso.endswith('Z'):
        iso = iso[:-1]
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def add_seconds(dt: datetime, seconds: int) -> datetime:
    """Return a datetime increased by the given seconds."""
    return dt + timedelta(seconds=seconds)
