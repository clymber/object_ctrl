import os
import sys
from io import StringIO
from pathlib import Path

import pytest

from object_ctrl import config


def test_configure_stdio_path_relative_to_filters_both_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    config.configure_stdio_relative_path()
    prefix = f"{tmp_path.resolve()}{os.sep}"
    sys.stdout.write(f"{prefix}stdout.txt")
    sys.stderr.write(f"{prefix}stderr.txt")

    assert stdout.getvalue() == "stdout.txt"
    assert stderr.getvalue() == "stderr.txt"
