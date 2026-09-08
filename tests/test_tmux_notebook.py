"""
Check detached notebook settings against a simulated pre-existing tmux server.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _write_executable(path: Path, source: str) -> None:
    """
    Write a test command that runs under the current Python interpreter.
    """
    path.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def notebook_launcher(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """
    Build a disposable launcher with fake nbconvert and stale tmux environment.
    """
    project = tmp_path / "project with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    for name in ("tmux_notebook.sh", "activate_renku_env.sh"):
        shutil.copy2(repository / "scripts" / name, scripts / name)
    environment = project / "environment with spaces"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/activate").write_text(
        'export VIRTUAL_ENV="$OBJCTRL_RENKU_VENV"\n'
        'export PATH="$VIRTUAL_ENV/bin:$PATH"\n',
        encoding="utf-8",
    )
    notebook = project / "test notebook.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    capture = tmp_path / "captured.json"
    _write_executable(
        environment / "bin/jupyter-nbconvert",
        """
import json
import os
import sys
from pathlib import Path

keys = {
    "RFDETR_MODE", "RFDETR_SMOKE", "RFDETR_RUN_DIR", "RFDETR_BENCHMARK",
    "RFDETR_EXTRA", "RFDETR_STALE", "YOLOX_TINY_SMOKE", "ULTRALYTICS_WORKERS",
    "HF_HOME", "TORCH_HOME", "PYTHONPATH", "PYTHONHOME", "NOTEBOOK_KERNEL",
    "VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES",
}
Path(os.environ["TEST_CAPTURE_FILE"]).write_text(json.dumps({
    "environment": {key: os.environ[key] for key in keys if key in os.environ},
    "arguments": sys.argv[1:],
}))
""",
    )
    commands = tmp_path / "commands"
    commands.mkdir()
    _write_executable(
        commands / "tmux",
        """
import os
import subprocess
import sys

arguments = sys.argv[1:]
if arguments[0] == "has-session":
    raise SystemExit(1)
assert arguments[0] == "new-session", arguments
server = {
    key: value for key, value in os.environ.items()
    if not key.startswith(("RFDETR_", "YOLOX_TINY_", "ULTRALYTICS_"))
}
server.update({
    "RFDETR_MODE": "resume", "RFDETR_SMOKE": "stale-smoke",
    "RFDETR_RUN_DIR": "stale-run", "RFDETR_STALE": "old-server-only",
    "YOLOX_TINY_SMOKE": "1", "ULTRALYTICS_WORKERS": "77",
    "HF_HOME": "/old/server/cache", "TORCH_HOME": "/old/server/torch",
    "PYTHONPATH": "/old/server/packages", "CUDA_VISIBLE_DEVICES": "99",
})
directory_index = arguments.index("-c") + 1
result = subprocess.run(
    arguments[directory_index + 1], shell=True, executable="/bin/bash",
    cwd=arguments[directory_index], env=server,
)
raise SystemExit(result.returncode)
""",
    )
    caller = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("RFDETR_", "YOLOX_TINY_", "ULTRALYTICS_"))
        and key
        not in {
            "HF_HOME",
            "TORCH_HOME",
            "PYTHONHOME",
            "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES",
        }
    }
    caller.update(
        {
            "PATH": str(commands) + os.pathsep + os.environ["PATH"],
            "OBJCTRL_RENKU_VENV": str(environment),
            "OBJCTRL_RENKU_KERNEL_NAME": "object-ctrl-renku-rfdetr",
            "NOTEBOOK_KERNEL": "object-ctrl-renku-rfdetr",
            "TEST_CAPTURE_FILE": str(capture),
        }
    )
    return notebook, caller, capture


def _launch(notebook: Path, environment: dict[str, str]) -> None:
    """
    Exercise the actual launcher without starting Jupyter, tmux, or training.
    """
    subprocess.run(
        [
            "bash",
            str(notebook.parent / "scripts/tmux_notebook.sh"),
            "run",
            "--file",
            str(notebook),
            "--session",
            "test-run",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_tmux_replays_caller_settings_literally(notebook_launcher) -> None:
    """
    Preserve smoke/resume controls, literal shell metacharacters, and empty values.
    """
    notebook, caller, capture = notebook_launcher
    marker = notebook.parent / "must-not-execute"
    run_dir = f"runs/a 'quoted' directory $(touch '{marker}')"
    supplied = {
        "RFDETR_MODE": "resume",
        "RFDETR_SMOKE": "1",
        "RFDETR_RUN_DIR": run_dir,
        "RFDETR_EXTRA": "future control\nsecond line",
        "RFDETR_BENCHMARK": "",
        "HF_HOME": "models/a cache",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    caller.update(supplied)
    _launch(notebook, caller)
    result = json.loads(capture.read_text())
    for key, value in supplied.items():
        assert result["environment"][key] == value
    assert result["environment"]["VIRTUAL_ENV"] == caller["OBJCTRL_RENKU_VENV"]
    assert (
        "--ExecutePreprocessor.kernel_name=object-ctrl-renku-rfdetr"
        in result["arguments"]
    )
    assert "RFDETR_STALE" not in result["environment"]
    assert not marker.exists()


def test_tmux_unsets_absent_controls_from_old_server(notebook_launcher) -> None:
    """
    Unset caller settings must not inherit old smoke, resume, GPU, or cache values.
    """
    notebook, caller, capture = notebook_launcher
    _launch(notebook, caller)
    environment = json.loads(capture.read_text())["environment"]
    for key in (
        "RFDETR_MODE",
        "RFDETR_SMOKE",
        "RFDETR_RUN_DIR",
        "RFDETR_STALE",
        "YOLOX_TINY_SMOKE",
        "ULTRALYTICS_WORKERS",
        "HF_HOME",
        "TORCH_HOME",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "PYTHONHOME",
    ):
        assert key not in environment
