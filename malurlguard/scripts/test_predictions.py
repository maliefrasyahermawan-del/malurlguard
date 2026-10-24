"""Quick prediction test script."""
import logging
logging.disable(logging.CRITICAL)

import sys
sys.path.insert(0, ".")

from malurlguard.inference.predictor import Predictor

p = Predictor()

urls = [
    ("phishing",  "http://paypal-verify-account.tk/login.php?id=123"),
    ("benign",    "https://github.com/torvalds/linux"),
    ("malware",   "http://185.220.101.5/update.exe?token=abc123"),
    ("benign",    "https://telkomuniversity.ac.id/en/admission"),
    ("phishing",  "http://free-prize-winner.cf/claim?id=9999"),
    ("defacement","http://smallbusiness.example.com/hacked-by-anonymous.html"),
]

print("=== MalUrlGuard Prediction Test ===\n")
all_pass = True
for expected, url in urls:
    r = p.predict(url)
    label = r["label"]
    conf  = r["confidence"]
    ms    = r["latency_ms"]
    status = "PASS" if label == expected else "WARN"
    if label != expected:
        all_pass = False
    print("[{}] [{:12s} {:.0f}%] {:3.1f}ms  {}".format(
        status, label, conf*100, ms, url[:65]))

print()
if all_pass:
    print("All predictions match expected labels.")
else:
    print("Some predictions differ from expected (check above).")
