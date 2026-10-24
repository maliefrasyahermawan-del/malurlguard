"""
Generate a synthetic URL dataset (10,000 samples) for MVP testing.

Labels: 0=benign, 1=phishing, 2=malware, 3=defacement
Usage: python scripts/generate_synthetic.py [--output data/synthetic_urls.csv]
"""

import csv
import os
import random
import sys
import argparse
from pathlib import Path

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# ── Benign templates ──────────────────────────────────────────────────────────
BENIGN_DOMAINS = [
    "github.com", "wikipedia.org", "stackoverflow.com", "python.org",
    "docs.python.org", "scikit-learn.org", "pandas.pydata.org",
    "numpy.org", "tensorflow.org", "pytorch.org",
    "medium.com", "dev.to", "hashnode.dev", "towardsdatascience.com",
    "arxiv.org", "semanticscholar.org", "acm.org", "ieee.org",
    "telkomuniversity.ac.id", "itb.ac.id", "ui.ac.id", "ugm.ac.id",
    "bca.co.id", "mandiri.co.id", "tokopedia.com", "gojek.com",
    "detik.com", "kompas.com", "tempo.co",
    "amazon.com", "ebay.com", "reddit.com", "twitter.com",
    "youtube.com", "google.com", "microsoft.com", "apple.com",
    "cloudflare.com", "fastly.com", "netlify.app", "vercel.app",
    "aws.amazon.com", "azure.microsoft.com", "cloud.google.com",
]

BENIGN_PATHS = [
    "/",
    "/about",
    "/contact",
    "/blog/posts/introduction-to-machine-learning",
    "/user/johndoe/repo/blob/main/README.md",
    "/wiki/Machine_learning",
    "/questions/12345/how-to-use-pandas",
    "/documentation/v2/api-reference",
    "/products/category/electronics",
    "/news/technology/2024/01/15/cybersecurity-update",
    "/downloads/stable/release-notes.html",
    "/en/latest/tutorial.html",
    "/static/images/logo.png",
    "/api/v1/users",
    "/search?q=python+tutorial&page=1",
    "/index.php",
    "/home",
    "/dashboard",
    "/profile",
]

BENIGN_SUBDOMAINS = ["", "www", "docs", "api", "blog", "mail", "support", "cdn", "static"]


# ── Phishing templates ────────────────────────────────────────────────────────
PHISHING_BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook",
    "instagram", "netflix", "ebay", "bankofamerica", "chase",
    "bca", "mandiri", "bni", "bribank",
]

PHISHING_VERBS = [
    "secure", "verify", "update", "confirm", "login", "signin",
    "validate", "authenticate", "account", "alert",
]

PHISHING_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".click",
    ".top", ".pw", ".ru", ".cn", ".info", ".biz",
]

PHISHING_PATHS = [
    "/login.php?redirect=account",
    "/account/verify?token={token}&id={num}",
    "/signin?next=%2Fdashboard",
    "/update/security/step1.php?sess={token}",
    "/confirm?email=user@example.com&code={num}",
    "/webscr?cmd=_login-submit&dispatch={token}",
    "/authenticate?auth_key={token}&source=email",
    "/reset-password?uid={num}&token={token}",
]

PHISHING_IPS = [
    "192.168.1.{}", "10.0.0.{}", "172.16.0.{}",
    "185.220.101.{}", "45.33.32.{}",
]


def _rand_token(length: int = 16) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choices(chars, k=length))


def _rand_num() -> str:
    return str(random.randint(100, 9999999))


# ── Malware templates ─────────────────────────────────────────────────────────
MALWARE_PATTERNS = [
    "http://{ip}/{token}.exe",
    "http://{domain}/download/{token}.bat",
    "http://{domain}/wp-content/uploads/{token}/shell.php",
    "http://{domain}/cgi-bin/{token}.pl?cmd=wget",
    "http://{domain}/admin/config.php?file=../../etc/passwd",
    "http://dl.{domain}/malware/{token}/dropper.zip",
    "ftp://{ip}/pub/{token}.tar.gz",
    "http://{ip}:8080/{token}/payload.bin",
    "http://{domain}/update.php?src=http://evil.tk/{token}.exe",
    "http://{domain}/?page=php://filter/convert.base64-encode/resource={token}",
]

MALWARE_DOMAINS = [
    "darkhotel.ru", "freeware-downloads.net", "crackz.pw",
    "keygen-free.tk", "warez.xyz", "piratebay.biz",
    "update-flash.com", "java-update.net", "adobe-flash-player.info",
]


# ── Defacement templates ──────────────────────────────────────────────────────
DEFACEMENT_PATTERNS = [
    "http://{domain}/hacked-by-{group}.html",
    "http://{domain}/index.php?id={num}%27%20OR%201=1--",
    "http://{domain}/wp-admin/options.php",
    "http://{domain}/?author=1&FUZZ=../../../../etc/shadow",
    "http://{domain}/phpmyadmin/index.php",
    "http://{domain}/.git/config",
    "http://{domain}/backup.sql",
    "http://{domain}/admin/?action=upload&file=shell.php",
]

DEFACEMENT_GROUPS = [
    "anonymous", "lulzsec", "TeamXsploit", "HackerGroup1337",
    "CyberCaliphate", "r00t3d",
]

DEFACEMENT_DOMAINS = [
    "smallbusiness.example.com", "localmunicipality.go.id",
    "schoolsite.edu", "churchwebsite.org", "shopsite.biz",
]


# ── Generators ────────────────────────────────────────────────────────────────

def generate_benign(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        subdomain = random.choice(BENIGN_SUBDOMAINS)
        domain = random.choice(BENIGN_DOMAINS)
        host = f"{subdomain}.{domain}" if subdomain else domain
        scheme = "https" if random.random() > 0.1 else "http"
        path = random.choice(BENIGN_PATHS)
        url = f"{scheme}://{host}{path}"
        rows.append({"url": url, "label": 0, "source": "synthetic_benign"})
    return rows


def generate_phishing(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        brand = random.choice(PHISHING_BRANDS)
        verb = random.choice(PHISHING_VERBS)
        tld = random.choice(PHISHING_TLDS)
        use_ip = random.random() < 0.25

        if use_ip:
            ip_template = random.choice(PHISHING_IPS)
            ip = ip_template.format(random.randint(1, 254))
            host = ip
        else:
            sep = random.choice(["-", ".", ""])
            host = f"{brand}{sep}{verb}{tld}"

        path_template = random.choice(PHISHING_PATHS)
        path = path_template.format(
            token=_rand_token(), num=_rand_num()
        )
        # Classic @ obfuscation trick occasionally
        if random.random() < 0.15:
            url = f"http://{random.choice(BENIGN_DOMAINS)}@{host}{path}"
        else:
            url = f"http://{host}{path}"
        rows.append({"url": url, "label": 1, "source": "synthetic_phishing"})
    return rows


def generate_malware(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        pattern = random.choice(MALWARE_PATTERNS)
        domain = random.choice(MALWARE_DOMAINS)
        ip_template = random.choice(PHISHING_IPS)
        ip = ip_template.format(random.randint(1, 254))
        url = pattern.format(
            domain=domain, ip=ip, token=_rand_token(), num=_rand_num()
        )
        rows.append({"url": url, "label": 2, "source": "synthetic_malware"})
    return rows


def generate_defacement(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        pattern = random.choice(DEFACEMENT_PATTERNS)
        domain = random.choice(DEFACEMENT_DOMAINS)
        group = random.choice(DEFACEMENT_GROUPS)
        url = pattern.format(
            domain=domain, group=group, token=_rand_token(), num=_rand_num()
        )
        rows.append({"url": url, "label": 3, "source": "synthetic_defacement"})
    return rows


def run_generate(output_path: str, total: int = 10_000) -> None:
    """Generate synthetic dataset and save to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Class distribution: 50% benign, 25% phishing, 15% malware, 10% defacement
    n_benign = total // 2
    n_phishing = total // 4
    n_malware = int(total * 0.15)
    n_defacement = total - n_benign - n_phishing - n_malware

    print(f"Generating {total} synthetic URLs:")
    print(f"  Benign:      {n_benign}")
    print(f"  Phishing:    {n_phishing}")
    print(f"  Malware:     {n_malware}")
    print(f"  Defacement:  {n_defacement}")

    rows = (
        generate_benign(n_benign)
        + generate_phishing(n_phishing)
        + generate_malware(n_malware)
        + generate_defacement(n_defacement)
    )

    random.shuffle(rows)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOK: Saved {len(rows)} URLs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic URL dataset.")
    parser.add_argument("--output", default="data/synthetic_urls.csv")
    parser.add_argument("--total", type=int, default=10_000)
    args = parser.parse_args()
    run_generate(output_path=args.output, total=args.total)
