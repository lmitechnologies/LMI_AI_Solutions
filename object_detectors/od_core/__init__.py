import importlib
import pkgutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)

def _discover_and_import_modules(package_path: Path, package_name: str):
    log.debug(f"Scanning for modules/packages in '{package_path}' for package '{package_name}'")
    if not package_path.is_dir():
        log.warning(f"Directory not found, skipping scan: {package_path}")
        return 0

    imported_count = 0
    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name
        is_package = module_info.ispkg
        relative_module_path = f".{module_name}"
        full_module_name_for_log = f"{package_name}.{module_name}"

        try:
            imported_module = importlib.import_module(relative_module_path, package=package_name)
            log.info(f"  Successfully imported {'package' if is_package else 'module'}: {full_module_name_for_log}")
            imported_count += 1
        except ImportError as e:
            log.warning(f"  Could not import {full_module_name_for_log}. Error: {e}")
        except Exception as e:
            log.error(f"  Error during import or execution of {full_module_name_for_log}: {e}", exc_info=True)
    return imported_count

MODEL_PARENT_DIRS = ['yolov8_lmi', 'ultralytics_lmi', 'detectron2_lmi']

log.info(f"[{__name__}] Initializing automatic model registration...")

package_root = Path(__file__).resolve().parent.parent

total_modules_scanned = 0
for dir_rel_path in MODEL_PARENT_DIRS:
    current_dir_abs_path = package_root / dir_rel_path
    current_package_name = f"{dir_rel_path.replace('/', '.')}"

    log.info(f"--- Scanning directory '{dir_rel_path}' (Package: {current_package_name}) ---")
    modules_found = _discover_and_import_modules(current_dir_abs_path, current_package_name)
    total_modules_scanned += modules_found

log.info(f"[{__name__}] Finished automatic model registration scan. Scanned ~{total_modules_scanned} modules/subpackages across configured directories.")