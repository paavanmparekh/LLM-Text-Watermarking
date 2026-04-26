"""
Shared logger for the PRC watermarking scheme.

Creates a timestamped .log file inside
    llm_watermarking/watermarks/prc/logs/
on first import. Both generation.py and detection.py import this module.
"""

import logging
import os
from datetime import datetime

_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FILE = os.path.join(_LOG_DIR, f"prc_{_timestamp}.log")

logger = logging.getLogger("prc_watermark")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_fh)
