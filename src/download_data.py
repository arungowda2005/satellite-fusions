
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "labels.geojson": {
        "url": "https://zenodo.org/records/7257688/files/labels.geojson?download=1",
        "md5": "a065c6d975bbdcd6a5471ceed0b007f9",
    },
    "features.tar.gz": {
        "url": "https://zenodo.org/records/7257688/files/features.tar.gz?download=1",
        "md5": "2a97ef61b955fa4ba1abbe4a575b9943",
    },
}


def calculate_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(name, details):
    destination = RAW_DIR / name

    if not destination.exists():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(details["url"], destination)

    actual = calculate_md5(destination)
    expected = details["md5"]

    print(f"\nFile: {name}")
    print(f"MD5: {actual}")

    if actual != expected:
        raise ValueError(f"Checksum mismatch for {name}")

    print("Checksum verified!")


if __name__ == "__main__":
    for name, details in FILES.items():
        download_file(name, details)

    print("\nAll dataset files verified successfully.")
