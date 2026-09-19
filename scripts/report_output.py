"""Output guard for regenerating readable reports without changing evidence."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def create_output_directory(path):
    output = Path(path).resolve()
    work = ROOT / 'work'
    if not output.is_relative_to(work) or output == work:
        raise ValueError('Output must be a new subdirectory of repository work/.')
    output.mkdir(parents=True, exist_ok=False)
    return output


def report_output():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New work/ directory for report and derived sidecars')
    args = parser.parse_args()
    try:
        return create_output_directory(args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
