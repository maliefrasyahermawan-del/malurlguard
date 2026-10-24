"""Lexical feature extraction from URL strings (no network required)."""

import logging
import math
import re
from urllib.parse import urlparse, parse_qs
from typing import Dict

import tldextract

logger = logging.getLogger(__name__)

# Suspicious keywords commonly found in phishing/malware URLs
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "logon", "log-in",
    "secure", "security", "verify", "verification",
    "account", "accounts", "update", "confirm", "confirmation",
    "banking", "bank", "payment", "pay", "checkout",
    "paypal", "ebay", "amazon", "apple", "microsoft",
    "google", "facebook", "instagram", "twitter",
    "support", "helpdesk", "service", "customer",
    "password", "passwd", "credential",
    "webscr", "cmd=", "dispatch",
    "token", "session", "auth",
    "free", "prize", "winner", "lucky", "congratulation",
    "click", "here", "offer",
]

# Common TLDs considered legitimate
COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "mil",
    "id", "co.id", "ac.id", "go.id", "or.id",
    "uk", "co.uk", "de", "fr", "jp", "au", "ca",
    "io", "dev", "app", "tech",
}


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string."""
    if not text:
        return 0.0
    freq: Dict[str, float] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def _max_consecutive_consonants(text: str) -> int:
    """Return the maximum run of consecutive consonants in a string."""
    consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
    max_run = current = 0
    for ch in text:
        if ch in consonants:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _vowel_ratio(text: str) -> float:
    """Ratio of vowels to total alphabetic characters."""
    vowels = set("aeiouAEIOU")
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return round(sum(1 for c in alpha if c in vowels) / len(alpha), 4)


def extract_lexical_features(url: str) -> Dict[str, float]:
    """
    Extract 30+ lexical features from a URL string without making network calls.

    Args:
        url: The full URL string to analyse.

    Returns:
        Dictionary mapping feature name to numeric value.
    """
    features: Dict[str, float] = {}

    try:
        parsed = urlparse(url)
    except Exception:
        logger.warning("Failed to parse URL: %s", url)
        parsed = urlparse("")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    try:
        extracted = tldextract.extract(url)
        subdomain = extracted.subdomain or ""
        domain = extracted.domain or ""
        suffix = extracted.suffix or ""
        registered_domain = getattr(extracted, "top_domain_under_public_suffix", None) or extracted.registered_domain or ""
    except Exception:
        subdomain = domain = suffix = registered_domain = ""

    # ── Length features ──────────────────────────────────────────────────────
    features["url_length"] = len(url)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)
    features["query_length"] = len(query)
    features["fragment_length"] = len(fragment)

    # ── Character count features ──────────────────────────────────────────────
    features["num_dots"] = url.count(".")
    features["num_dashes"] = url.count("-")
    features["num_underscores"] = url.count("_")
    features["num_slashes"] = url.count("/")
    features["num_equals"] = url.count("=")
    features["num_questionmarks"] = url.count("?")
    features["num_ampersands"] = url.count("&")
    features["num_percent"] = url.count("%")
    features["num_at"] = url.count("@")

    digits = sum(1 for c in url if c.isdigit())
    letters = sum(1 for c in url if c.isalpha())
    features["num_digits"] = digits
    features["num_letters"] = letters
    features["ratio_digits_to_total"] = round(digits / max(len(url), 1), 4)

    # Characters that are not alphanumeric and not standard URL chars (: / . - _ ? = &)
    standard_url_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:/?=&.-_~%#+@")
    features["num_special_chars"] = sum(1 for c in url if c not in standard_url_chars)

    # ── Structural / suspicious pattern features ──────────────────────────────
    # IPv4 address as hostname
    ipv4_re = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    features["has_ip_as_hostname"] = float(bool(ipv4_re.match(hostname)))

    features["has_at_symbol"] = float("@" in url)

    # Explicit port (not 80/443)
    port = parsed.port
    features["has_port"] = float(port is not None and port not in (80, 443))

    features["has_https"] = float(parsed.scheme == "https")

    # Double-slash redirect in path (not counting the scheme separator)
    path_no_lead = path.lstrip("/")
    features["has_double_slash_in_path"] = float("//" in path_no_lead)

    # ── TLD features ──────────────────────────────────────────────────────────
    features["tld_length"] = len(suffix)
    features["tld_is_common"] = float(suffix in COMMON_TLDS)

    # ── Subdomain features ────────────────────────────────────────────────────
    subdomains = [s for s in subdomain.split(".") if s] if subdomain else []
    features["num_subdomains"] = len(subdomains)
    features["subdomain_length_avg"] = (
        round(sum(len(s) for s in subdomains) / len(subdomains), 2) if subdomains else 0.0
    )

    # ── Keyword features ──────────────────────────────────────────────────────
    url_lower = url.lower()
    features["has_suspicious_keyword"] = float(
        any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS)
    )
    features["suspicious_keyword_count"] = float(
        sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    )

    # ── Entropy features ──────────────────────────────────────────────────────
    features["shannon_entropy_hostname"] = _shannon_entropy(hostname)
    features["shannon_entropy_path"] = _shannon_entropy(path)

    # ── Encoding / obfuscation features ──────────────────────────────────────
    features["has_punycode"] = float("xn--" in hostname.lower())
    # "https" literally appearing inside the hostname (anti-pattern)
    features["has_https_in_hostname"] = float("https" in hostname.lower())
    # Percent encoding ratio (high = obfuscation)
    features["percent_encoding_ratio"] = round(url.count("%") / max(len(url), 1), 4)

    # ── Token-level features ──────────────────────────────────────────────────
    tokens = re.split(r"[./\-_]", hostname + path)
    tokens = [t for t in tokens if t]
    features["longest_token_length"] = max((len(t) for t in tokens), default=0)
    features["num_tokens"] = len(tokens)

    # ── Consonant / vowel features ────────────────────────────────────────────
    features["max_consecutive_consonants"] = float(_max_consecutive_consonants(hostname))
    features["vowel_ratio"] = _vowel_ratio(hostname)

    # ── Query parameter features ──────────────────────────────────────────────
    try:
        params = parse_qs(query)
        features["num_query_params"] = float(len(params))
    except Exception:
        features["num_query_params"] = 0.0

    # ── Domain length ─────────────────────────────────────────────────────────
    features["domain_length"] = len(domain)
    features["registered_domain_length"] = len(registered_domain)

    return features


def feature_names() -> list[str]:
    """Return ordered list of feature names (for consistent column ordering)."""
    sample = extract_lexical_features("https://example.com/path?q=1#anchor")
    return list(sample.keys())
