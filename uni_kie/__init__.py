"""uni-kie module"""
import logging
from datetime import datetime
from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parents[1]  # repo folder -> unimodal-kie
PATH_TESTS = ROOT / "tests"
PATH_UNI_KIE = ROOT / "uni_kie"
PATH_DATA = PATH_UNI_KIE / "datasets"
PATH_LOGS = ROOT / "logs"


if not PATH_LOGS.is_dir():
    PATH_LOGS.mkdir(parents=True, exist_ok=True)

LOGFILE = PATH_LOGS / f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"


def create_logger(
    name="",
    filename=str(LOGFILE),
    level=logging.DEBUG,
    logformat="%(asctime)s - %(name)s - %(levelname)s - %(lineno)s - %(funcName)s() - %(message)s",
):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename, delay=True)
    formatter = logging.Formatter(logformat)
    handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
