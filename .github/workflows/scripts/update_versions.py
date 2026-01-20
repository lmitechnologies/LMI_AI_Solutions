import sys
from pathlib import Path

import tomlkit

FILES_TO_UPDATE = [
    "classifiers/pyproject.toml",
    "anomaly_detectors/pyproject.toml",
    "lmi_utils/pyproject.toml",
    "object_detectors/pyproject.toml",
]


def update_version(file_path, new_version):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    if "project" in doc and "version" in doc["project"]:
        old_ver = doc["project"]["version"]
        if old_ver != new_version:
            doc["project"]["version"] = new_version
            path.write_text(tomlkit.dumps(doc), encoding="utf-8")
            print(f"Updated {file_path} from {old_ver} to version {new_version}")
        else:
            print(f"No changes needed for {file_path} (already version {new_version})")


def main():
    if len(sys.argv) != 2:
        print("Usage: python update_versions.py <new_version>")
        sys.exit(1)

    new_version = sys.argv[1]

    current_cwd = Path.cwd()
    for relative_path in FILES_TO_UPDATE:
        file_path = current_cwd / relative_path
        update_version(file_path, new_version)


if __name__ == "__main__":
    main()
