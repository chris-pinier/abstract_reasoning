from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = PACKAGE_DIR.parents[1]
REPO_ROOT = ANALYSIS_DIR.parent
CONFIG_DIR = REPO_ROOT / "config"
SCRIPTS_DIR = ANALYSIS_DIR / "scripts"
DATA_DIR = ANALYSIS_DIR / "data"
RESULTS_DIR = ANALYSIS_DIR / "results"
