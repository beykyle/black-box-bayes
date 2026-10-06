"""A static dynesty run that stopped on --maxcall can be resumed and run to completion."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dynesty")

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp, extra):
    cmd = [sys.executable, "-m", "black_box_bayes.cli", "--input", str(tmp / "toy_config.pkl"),
           "--output", str(tmp / "out"), "--sampler", "dynesty", "--no-mpi", "--dynesty-run", "static",
           "--nlive", "40", "--dlogz", "0.5", "--seed", "3",
           "--dynesty-checkpoint", str(tmp / "out" / "ckpt.pkl"), "--dynesty-checkpoint-every", "1",
           "--dynesty-native-results", str(tmp / "out" / "res.npz"), "--dynesty-history", "none",
           "--idata-results", str(tmp / "out" / "idata.nc")] + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(tmp))


def test_static_resume_after_maxcall(tmp_path):
    # the pickle references the toy_model module, which the CLI finds next to the pickle
    shutil.copy(ROOT / "examples" / "toy" / "toy_model.py", tmp_path / "toy_model.py")
    sys.path.insert(0, str(tmp_path))
    import toy_model  # noqa: E402

    toy_model.make_config(str(tmp_path / "toy_config.pkl"))
    first = _run(tmp_path, ["--maxcall", "400"])
    assert first.returncode == 0, first.stderr[-2000:]
    r1 = np.load(tmp_path / "out" / "res.npz")
    assert int(np.sum(r1["ncall"])) >= 400
    second = _run(tmp_path, ["--dynesty-resume"])
    assert second.returncode == 0, second.stderr[-2000:]
    assert "Removed previously added live points" in second.stdout + second.stderr
    r2 = np.load(tmp_path / "out" / "res.npz")
    assert int(np.sum(r2["ncall"])) > int(np.sum(r1["ncall"]))
    assert "resuming a finished static run" not in second.stderr
