"""Interactive token setup: authenticate before saving; never echo the token."""

import getpass
import os
from pathlib import Path

import requests

root = Path(__file__).resolve().parents[1]
bundle = root / ".cache/nsrr-ca-chain.pem"
if bundle.exists():
    os.environ["REQUESTS_CA_BUNDLE"] = str(bundle)
token = getpass.getpass(
    "Paste the token from https://sleepdata.org/token, then press Enter: "
).strip()
try:
    response = requests.get(
        "https://sleepdata.org/api/v1/account/profile.json",
        params={"auth_token": token},
        timeout=60,
    )
    authenticated = (
        response.status_code == 200 and response.json().get("authenticated") is True
    )
except (requests.RequestException, ValueError):
    raise SystemExit(
        "Could not securely verify NSRR access. Token was not saved."
    ) from None
if not authenticated:
    raise SystemExit(
        "NSRR did not accept that token. Token was not saved. Copy only the token from sleepdata.org/token."
    )
path = Path.home() / ".config/nsrr/token"
path.parent.mkdir(parents=True, exist_ok=True)
path.parent.chmod(0o700)
path.touch(mode=0o600, exist_ok=True)
path.chmod(0o600)
path.write_text(token)
print("AUTHORIZED. NSRR token saved privately on the cluster.")
