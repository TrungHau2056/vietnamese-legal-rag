#!/usr/bin/env python3
from __future__ import annotations

# Submission building logic is available in scripts/merge_questions_and_generate_answers.py.
# This file is kept as a conventional source entry point for repository organization.

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "merge_questions_and_generate_answers.py"), run_name="__main__")
