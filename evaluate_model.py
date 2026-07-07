#!/usr/bin/env python3
"""Executable wrapper for the Turkish ASR evaluation CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from turkish_asr_eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
