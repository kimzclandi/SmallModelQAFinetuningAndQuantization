"""Regression checks for evidence loss and unsafe acceptance invocation."""
import subprocess
import sys
from pathlib import Path
import pytest
from scripts import chinese_study, coverage_study, quantization_study, teacher_study

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("module,func,root_name", [
    (chinese_study, "summarize", "R"), (coverage_study, "summarize", "R"),
    (quantization_study, "summarize", "R"), (teacher_study, "analyze", "ROOT")])
def test_missing_frozen_summary_is_not_recreated(tmp_path, monkeypatch, module, func, root_name):
    original = getattr(module, root_name)
    for p in original.iterdir():
        if p.name != "summary.json":
            (tmp_path/p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
    monkeypatch.setattr(module, root_name, tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing frozen summary"):
        getattr(module, func)(read_only=True)
    assert not (tmp_path/"summary.json").exists()

def test_acceptance_rejects_optimized_python():
    run = subprocess.run([sys.executable, "-O", "scripts/acceptance.py", "--output", "work/unused"], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode != 0 and "Assertions must be enabled" in run.stderr

def test_acceptance_rejects_frozen_output():
    run = subprocess.run([sys.executable, "scripts/acceptance.py", "--output", "reports/unused"], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode != 0 and "new subdirectory" in run.stderr


def test_acceptance_rejects_existing_output(tmp_path):
    import tempfile
    (ROOT/"work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/"work") as folder:
        marker = Path(folder)/"sentinel.txt"
        marker.write_text("keep")
        run = subprocess.run([sys.executable, "scripts/acceptance.py", "--output", folder], cwd=ROOT, capture_output=True, text=True)
        assert run.returncode != 0 and "FileExistsError" in run.stderr
        assert marker.read_text() == "keep"
