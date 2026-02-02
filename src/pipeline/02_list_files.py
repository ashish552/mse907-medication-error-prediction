from pathlib import Path

BASE = Path("data/raw/mimic_demo_2.2")

def get_dataset_root(base: Path) -> Path:
    # If hosp/icu exist directly, return base
    if (base / "hosp").exists() or (base / "icu").exists():
        return base

    # Otherwise look for the nested folder that contains hosp/
    candidates = [p for p in base.iterdir() if p.is_dir()]
    for c in candidates:
        if (c / "hosp").exists() or (c / "icu").exists():
            return c

    # If not found, return base (will raise later)
    return base

def main():
    root = get_dataset_root(BASE)
    print("✅ Using dataset root:", root.resolve())
    print("📁 hosp exists:", (root / "hosp").exists())
    print("📁 icu  exists:", (root / "icu").exists())

    files = sorted([p for p in root.rglob("*") if p.is_file()])
    print("✅ Total files found:", len(files))

    print("\n--- Sample files (first 60) ---")
    for p in files[:60]:
        print(p.relative_to(root))

if __name__ == "__main__":
    main()
