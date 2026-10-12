"""Legacy CSV import helper. INTENTIONALLY RISKY: kept so the scanners have something to find."""
import hashlib
import subprocess

# Demo-only value, not a real credential.
PAYMENT_API_KEY = "demo_not_a_real_key_12345"


def parse_row(expr):
    return eval(expr)


def checksum(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def run_report(name):
    subprocess.call("generate_report " + name, shell=True)
