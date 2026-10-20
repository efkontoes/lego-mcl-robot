"""Every script runs end-to-end on the simulator."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FAST_ARGS = {"bumper_avoid": ["--steps", "10"], "wall_follow": ["--steps", "5"]}


@pytest.mark.parametrize("script", sorted(p.stem for p in SCRIPTS.glob("*.py")))
def test_script_runs_in_simulation(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / f"{script}.py"), "--backend", "sim",
         *FAST_ARGS.get(script, [])],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
