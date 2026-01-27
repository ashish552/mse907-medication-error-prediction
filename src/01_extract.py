import zipfile
from pathlib import Path

ZIP_PATH = Path("data/raw/mimic-iv-clinical-database-demo-2.2.zip")
OUT_DIR = Path("data/raw/mimic_demo_2.2")

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Zip not found at: {ZIP_PATH.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(OUT_DIR)

    print(" Extracted to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
