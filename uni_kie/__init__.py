"""My module"""
from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parents[1]  # repo folder -> unimodal-kie
PATH_TESTS = ROOT / "tests"
PATH_UNI_KIE = ROOT / "uni_kie"
PATH_DATA = PATH_UNI_KIE / "datasets"
