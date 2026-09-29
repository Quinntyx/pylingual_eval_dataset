import pathlib
import tqdm
import click
import csv

from datetime import datetime

import signal

from pylingual.utils.generate_bytecode import CompileError
from pylingual.decompiler import decompile
from pylingual.utils.version import PythonVersion

TIMEOUT_SECONDS = 300
FIELDNAMES = ["pyc_file", "py_file", "identifier", "success", "category", "notes"]

def _timeout_handler(signum, frame):
    raise TimeoutError()

def decompile_with_timeout(pyc_file, target_out_dir):
    signal.alarm(TIMEOUT_SECONDS)
    try:
        return decompile(pyc_file, target_out_dir)
    finally:
        signal.alarm(0)


def evaluate(pyc_list: pathlib.Path, out_dir: pathlib.Path):
    start_time = datetime.now()
    signal.signal(signal.SIGALRM, _timeout_handler)

    out_dir = out_dir / f"pylingual-{start_time:%Y-%m-%d_%H-%M-%S}"
    results_dir = out_dir / "decompilation_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pyc_files = [pathlib.Path(line.strip()) for line in pyc_list.read_text().splitlines() if line.strip()]

    attempted = succeeded = 0

    with (out_dir / "evaluation_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        progress = tqdm.tqdm(pyc_files)
        for pyc_file in progress:
            identifier = pyc_file.parent.name
            target_out_dir = results_dir / f"{identifier}.py"
            row = {"pyc_file": pyc_file, "py_file": identifier, "identifier": "FILE"}
            attempted += 1

            try:
                py_file = decompile_with_timeout(pyc_file, target_out_dir)
            except Exception as err:
                writer.writerow({**row, "py_file": "", "success": False, "category": "DECOMPILER ERROR", "notes": repr(err)})
            else:
                ok = all(r.success for r in py_file.equivalence_results)
                succeeded += ok
                writer.writerow({**row, "success": ok, "category": "Equal" if ok else "Different", "notes": ""})
                writer.writerows(
                    {
                        **row,
                        "identifier": identifier,
                        "success": r.success,
                        "notes": str(r) if isinstance(r, CompileError) else r.message,
                    }
                    for r in py_file.equivalence_results
                )

            progress.set_postfix(file_success=f"{succeeded}/{attempted} ({succeeded / attempted:.2%})")

    rate = f"{succeeded / attempted:.2%}" if attempted else "N/A"
    (out_dir / "elapsed_time.txt").write_text(
        f"Elapsed Time: {datetime.now() - start_time}\n"
        f"File success: {succeeded}/{attempted} {rate}\n"
    )


@click.command(help= "Evaluation script for pylingual")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=pathlib.Path))
@click.option("-p", "--pylingual-version", default="v1", type=str, help="The PyLingual version you want to evaluate on")
@click.option("-l", "--pyc-list", default=None, type=click.Path(exists=True, path_type=pathlib.Path), help="A list of file paths to evaluate on")
@click.option("-v", "--version", default=None, type=str, help="If using a specific PyLingual version, choose which Python version to evaluate on")
def main(out_dir, pylingual_version, pyc_list, version):
    
    if pyc_list:
        evaluate(pyc_list, out_dir)
        return

    root_dir = pathlib.Path(f"pylingual{pylingual_version}")

    if version:
        ver = PythonVersion(version)
        lists = [root_dir / f"{ver.major}{ver.minor}-pyc-list.txt"]
    else:
        lists = sorted(root_dir.glob("*-pyc-list.txt"))

    for pyc_list in lists:
        ver = PythonVersion(pyc_list.name.split("-")[0])
        evaluate(pyc_list, out_dir / f"python-{ver.major}.{ver.minor}")

if __name__ == "__main__":
    main()
