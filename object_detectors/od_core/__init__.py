import importlib
import pkgutil
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)

package_root = Path(__file__).resolve().parent.parent

package_root_str = str(package_root)
if package_root_str not in sys.path:
    log.info(f"Adding package root to sys.path: {package_root_str}")
    sys.path.insert(0, package_root_str)
else:
    log.debug(f"Package root already in sys.path: {package_root_str}")


def _discover_and_import_modules_absolute(base_dir_path: Path, base_package_name: str):
    log.debug(f"Scanning for modules/packages in '{base_dir_path}' using base package name '{base_package_name}'")
    if not base_dir_path.is_dir():
        log.warning(f"Directory not found, skipping scan: {base_dir_path}")
        return 0
    if not (base_dir_path / "__init__.py").exists():
         log.warning(f"Directory '{base_dir_path}' is missing an __init__.py, skipping scan (it's not a package).")
         return 0

    imported_count = 0
    for module_info in pkgutil.iter_modules([str(base_dir_path)]):
        module_name = module_info.name
        is_package = module_info.ispkg
        absolute_module_path = f"{base_package_name}.{module_name}"

        try:
            imported_module = importlib.import_module(absolute_module_path)
            log.info(f"  Successfully imported {'package' if is_package else 'module'}: {absolute_module_path}")
            imported_count += 1
        except ImportError as e:
            log.warning(f"  Could not import {absolute_module_path}. Error: {e}. Check dependencies or PYTHONPATH.")
        except Exception as e:
            log.error(f"  Error during import or execution of {absolute_module_path}: {e}", exc_info=True)

    return imported_count

MODEL_PARENT_DIRS = ['yolov8_lmi', 'ultralytics_lmi', 'detectron2_lmi']

log.info(f"[{__name__}] Initializing automatic model registration from sibling directories...")

total_modules_scanned = 0
for dir_name in MODEL_PARENT_DIRS:
    current_dir_abs_path = package_root / dir_name
    current_package_name = dir_name

    log.info(f"--- Scanning directory '{dir_name}' (Package: {current_package_name}) ---")
    modules_found = _discover_and_import_modules_absolute(current_dir_abs_path, current_package_name)
    total_modules_scanned += modules_found

log.info(f"[{__name__}] Finished automatic model registration scan. Scanned ~{total_modules_scanned} modules/subpackages across configured directories.")