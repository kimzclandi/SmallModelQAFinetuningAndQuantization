import pytest
from scripts import report_output


def test_report_directory_is_new_and_under_work(tmp_path, monkeypatch):
    monkeypatch.setattr(report_output, 'ROOT', tmp_path)
    output = report_output.create_output_directory(tmp_path / 'work' / 'report')
    assert output.is_dir()
    marker = output / 'RESULTS.md'
    marker.write_text('original')
    with pytest.raises(FileExistsError):
        report_output.create_output_directory(output)
    assert marker.read_text() == 'original'


@pytest.mark.parametrize('relative', ['reports/new', 'data/new', 'work', 'work/../reports/new'])
def test_frozen_and_nonisolated_paths_rejected(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(report_output, 'ROOT', tmp_path)
    with pytest.raises(ValueError):
        report_output.create_output_directory(tmp_path / relative)


def test_symlink_escape_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(report_output, 'ROOT', tmp_path)
    (tmp_path / 'work').mkdir()
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'work' / 'escape').symlink_to(tmp_path / 'reports', target_is_directory=True)
    with pytest.raises(ValueError):
        report_output.create_output_directory(tmp_path / 'work' / 'escape' / 'new')
