"""Importing ptemcee through the CLI outside an MPI launcher must not initialize MPI."""

import subprocess
import sys

import pytest

pytest.importorskip("ptemcee")
pytest.importorskip("mpi4py")

SCRIPT = """
import sys
from black_box_bayes import cli
cli._ptemcee_imports()
import mpi4py.MPI as MPI
print("initialized", MPI.Is_initialized())
"""


def test_ptemcee_import_leaves_mpi_uninitialized():
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "initialized False" in out.stdout, out.stdout
