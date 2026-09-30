import csv
import multiprocessing as mp
import os
import pathlib
import signal
import subprocess
from datetime import datetime

import click
import tqdm

from pylingual.utils.version import PythonVersion

TIMEOUT_SECONDS = 300 # 5-minute timeout for decompiling one file
FIELDNAMES = ["pyc_file", "py_file", "identifier", "success", "category", "notes"]


# worker functions

def _timeout_handler(signum, frame):
    raise TimeoutError()


def _init_worker(gpu_queue):
    """Runs once per worker process to claim a GPU"""
    gpu = gpu_queue.get()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    signal.signal(signal.SIGALRM, _timeout_handler)


def _process_file(task):
    """Decompile one file. Returns plain picklable data for the parent to write."""
    # Imported here so CUDA_VISIBLE_DEVICES is set before loading torch
    from pylingual.decompiler import decompile
    from pylingual.utils.generate_bytecode import CompileError

    pyc_file, target_out_dir = task
    signal.alarm(TIMEOUT_SECONDS)
    try:
        py_file = decompile(pyc_file, target_out_dir)
    except Exception as err:
        return pyc_file, repr(err), None
    finally:
        signal.alarm(0)

    results = [
        (r.success, str(r) if isinstance(r, CompileError) else r.message)
        for r in py_file.equivalence_results
    ]
    return pyc_file, None, results


# main evaluation function
def evaluate(pool, pyc_list: pathlib.Path, out_dir: pathlib.Path):
    start_time = datetime.now()

    out_dir = out_dir / f"pylingual-{start_time:%Y-%m-%d_%H-%M-%S}"
    results_dir = out_dir / "decompilation_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pyc_files = [pathlib.Path(line.strip()) for line in pyc_list.read_text().splitlines() if line.strip()]
    tasks = [(p, results_dir / f"{p.parent.name}.py") for p in pyc_files]

    attempted = succeeded = 0

    with (out_dir / "evaluation_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        # decompile pyc files
        progress = tqdm.tqdm(total=len(tasks))
        for pyc_file, error, results in pool.imap_unordered(_process_file, tasks):
            identifier = pyc_file.parent.name
            row = {"pyc_file": pyc_file, "py_file": identifier, "identifier": "FILE"}
            attempted += 1

            if error is not None:
                writer.writerow({**row, "py_file": "", "success": False, "category": "DECOMPILER ERROR", "notes": error})
            else:
                ok = all(success for success, _ in results)
                succeeded += ok
                writer.writerow({**row, "success": ok, "category": "Equal" if ok else "Different", "notes": ""})
                writer.writerows(
                    {**row, "identifier": identifier, "success": success, "notes": notes}
                    for success, notes in results
                )

            # update progress bar
            f.flush()  
            progress.update(1)
            progress.set_postfix(file_success=f"{succeeded}/{attempted} ({succeeded / attempted:.2%})")
        progress.close()

    # final stats and time
    rate = f"{succeeded / attempted:.2%}" if attempted else "N/A"
    (out_dir / "elapsed_time.txt").write_text(
        f"Elapsed Time: {datetime.now() - start_time}\n"
        f"File success: {succeeded}/{attempted} {rate}\n"
    )


def detect_gpus() -> list[int]:
    """Count GPUs via nvidia-smi so the parent never initializes CUDA."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return list(range(sum(1 for line in out.splitlines() if line.startswith("GPU "))))


@click.command(help="Evaluation script for pylingual")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=pathlib.Path))
@click.option("-p", "--pylingual-version", default="v1", type=click.Choice(["v1", "v2"]), help="The PyLingual version you want to evaluate on")
@click.option("-l", "--pyc-list", default=None, type=click.Path(exists=True, path_type=pathlib.Path), help="A list of file paths to evaluate on")
@click.option("-v", "--version", default=None, type=str, help="If using a specific PyLingual version, choose which Python version to evaluate on")
@click.option("-g", "--gpus", default=None, type=str, help="Comma-separated GPU ids to use, e.g. 0,1,2 (default: all detected GPUs)")
@click.option("-w", "--workers-per-gpu", default=1, type=click.IntRange(min=1), help="Worker processes per GPU (default: 1)")
def main(out_dir, pylingual_version, pyc_list, version, gpus, workers_per_gpu):
    gpu_ids = [int(g) for g in gpus.split(",")] if gpus else detect_gpus()
    if not gpu_ids:
        click.echo("No GPUs found, running a single worker on CPU.")
        slots = [None]
    else:
        slots = [g for g in gpu_ids for _ in range(workers_per_gpu)]
    click.echo(f"Starting {len(slots)} worker(s) on GPUs: {gpu_ids or 'none'}")

    if pyc_list:
        lists = [(pyc_list, out_dir)]
    else:
        root_dir = pathlib.Path(f"pylingual{pylingual_version}")
        if version:
            ver = PythonVersion(version)
            paths = [root_dir / f"{ver.major}{ver.minor}-pyc-list.txt"]
        else:
            paths = sorted(root_dir.glob("*-pyc-list.txt"))
        lists = []
        for path in paths:
            ver = PythonVersion(path.name.split("-")[0])
            lists.append((path, out_dir / f"python-{ver.major}.{ver.minor}"))

    # "spawn" gives each worker a clean process, so CUDA is never inherited from the parent
    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    for slot in slots:
        gpu_queue.put(slot)

    with ctx.Pool(len(slots), initializer=_init_worker, initargs=(gpu_queue,)) as pool:
        for path, target in lists:
            evaluate(pool, path, target)


if __name__ == "__main__":
    main()
