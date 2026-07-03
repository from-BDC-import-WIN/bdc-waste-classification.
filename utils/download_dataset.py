import subprocess
import sys
from pathlib import Path


FOLDER_ID = "1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7"
OUTPUT_DIR = Path("/Users/faqihfirmanpratama/Documents/BDC/data")


def check_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def download():
    import gdown

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    print(f"Downloading from: {url}")
    print(f"Saving to: {OUTPUT_DIR}")

    gdown.download_folder(
        url=url,
        output=str(OUTPUT_DIR),
        quiet=False,
        use_cookies=False,
    )

    print("\nDone. Files saved to:", OUTPUT_DIR)
    files = list(OUTPUT_DIR.rglob("*"))
    print(f"Total files: {len([f for f in files if f.is_file()])}")


if __name__ == "__main__":
    check_gdown()
    download()
