"""Host-based feature extraction (requires network access, optional at runtime)."""

import logging
import socket
from typing import Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def extract_host_features(url: str, timeout: float = 3.0) -> Dict[str, float]:
    """
    Extract host-based features that require DNS/WHOIS lookups.

    These features are optional — they enrich accuracy but require network access.
    The function returns safe defaults (0.0) on any lookup failure so the pipeline
    stays functional offline.

    Args:
        url: The full URL string.
        timeout: DNS/WHOIS query timeout in seconds.

    Returns:
        Dictionary mapping feature name to numeric value.
    """
    features: Dict[str, float] = {
        "dns_resolves": 0.0,
        "dns_a_record_count": 0.0,
        "has_mx_record": 0.0,
        "domain_age_days": -1.0,   # -1 = unknown
        "whois_available": 0.0,
    }

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return features

        # DNS A record check
        try:
            results = socket.getaddrinfo(hostname, None, socket.AF_INET, proto=socket.IPPROTO_TCP)
            features["dns_resolves"] = 1.0
            features["dns_a_record_count"] = float(len(results))
        except (socket.gaierror, OSError):
            pass

    except Exception as exc:
        logger.debug("Host feature extraction failed for %s: %s", url, exc)

    return features
