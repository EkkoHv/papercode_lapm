"""Download and verify the public Swiss Jura data distributed with gstat."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


COMMIT = "2a578765502dd29520dcc3b40af42c953237faa3"
URL = (
    "https://raw.githubusercontent.com/r-spatial/gstat/"
    f"{COMMIT}/data/jura.rda"
)
EXPECTED_SHA256 = "f1d9e8a7e6686aa7473a1ce452fffe74f0c45a3e2ba464e35a9163c0d447d845"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/swiss_jura/jura.rda"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, args.output)
    observed = sha256(args.output)
    if observed != EXPECTED_SHA256:
        args.output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file failed SHA-256 verification: {observed}"
        )
    print(f"Verified Swiss Jura data: {args.output.resolve()}")
    print(f"Source commit: {COMMIT}")
    print(f"SHA-256: {observed}")


if __name__ == "__main__":
    main()
