import re
import sys
from pathlib import Path


def update_version(file_path, new_version):
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: File not found: {file_path}")
        return

    content = path.read_text(encoding="utf-8")
    # Regex to match version = "x.y.z"
    pattern = r'(^version\s*=\s*)(["\'])([^"\']+)(["\'])'

    if not re.search(pattern, content, re.MULTILINE):
        print(f"Warning: Could not find version string in {file_path}")
        return

    new_content = re.sub(pattern, f"\\g<1>\\g<2>{new_version}\\g<4>", content, flags=re.MULTILINE)

    if content != new_content:
        path.write_text(new_content, encoding="utf-8")
        print(f"Updated {file_path} to version {new_version}")
    else:
        print(f"No changes made to {file_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python update_versions.py <new_version>")
        sys.exit(1)

    new_version = sys.argv[1]

    files_to_update = [
        "classifiers/pyproject.toml",
        "anomaly_detectors/pyproject.toml",
        "lmi_utils/pyproject.toml",
        "object_detectors/pyproject.toml",
    ]

    base_dir = Path(__file__).resolve().parent.parent.parent.parent

    for relative_path in files_to_update:
        file_path = base_dir / relative_path
        update_version(file_path, new_version)


if __name__ == "__main__":
    main()
