# Pylingual Evaluation Dataset
Evaluation dataset and runner used for [Pylingual](https://github.com/syssec-utd/pylingual). This dataset originates from the [Pylingual Paper](https://ieeexplore.ieee.org/document/11023256) and has since been extended to support newer Python releases. 

The dataset is based on a set of files sampled from PyPI in 2023. 

## Setup
To run the Pylingual dataset evaluation, you will need:
- Linux
- Python 3.12+, to run Pylingual itself
- The `uv` package manager
- A CUDA-capable GPU with at least 12GB of VRAM (recommended)

CPU evaluation is also available as a fallback, but it will be substantially slower. 

In addition to these requirements, each target Python version is also required during evaluation so that Pylingual can recompile generated source to check for perfect decompilation against the original `.pyc`. However, for versions 3.8 and up, Pylingual is capable of automatic installation via `uvx` as needed. However, Python 3.6 and 3.7 are end-of-life and must be manually installed via `pyenv` prior to running the eval. 

Pylingual can install `pyenv` and the required interpreters: 
```sh
uv run pylingual --init-pyenv
```
Select Python 3.6 and 3.7 when prompted. 

To install remaining dependencies, run `uv sync` prior to evaluation. Pylingual segmentation and translation models are downloaded automatically from Hugging Face on first use and cached for later evaluations.  

### Redis translation cache

Pylingual caches translation-model predictions in redis, keyed by Python version and normalized bytecode. With the cache enabled, identical statements are only translated once — across workers, files, and evaluation runs — which speeds up evaluation considerably. This is the same setup the production deployment (`pylingual-webserver-deployment`) uses.

Start the cache with:
```sh
docker compose up -d redis
```

Then either pass `--redis-host 127.0.0.1` to `eval.py`, or set `PYLINGUAL_REDIS_HOST=127.0.0.1` in your environment. Without it, the evaluator runs with caching disabled (matching the old behavior). Note the cache lives in the container's memory: it is lost when the container is removed, so `docker compose restart redis` preserves it but re-creating it does not.

## Usage
Run an evaluation with:
```sh
uv run eval.py <output directory> [options]
```

| Option | Description |
| ------ | ----------- |
| `-p`, `--pylingual-version` | Dataset to evaluate on: `v1` or `v2` (default: `v1`) |
| `-v`, `--version` | Python version to evaluate on, e.g. `3.13`. If omitted, every version in the dataset is evaluated |
| `-l`, `--pyc-list` | Path to a custom text file of `.pyc` paths to evaluate. Overrides `-p` and `-v` |
| "-g", "--gpus" | Comma-separated GPU ids to use, uses all gpus on system by default |
| "-w", "--workers-per-gpu" | Worker processes per GPU, one worker by default |
| `-r`, `--redis-host` | Host of a redis translation cache, e.g. `127.0.0.1`. Defaults to `$PYLINGUAL_REDIS_HOST`; unset means caching is disabled |

Version text files are included in the repository for every supported Python release and contain an enumeration of paths to `.pyc` files used for that Python version.

Examples:

```sh
# Evaluate a single Python version (3.13) on the v1 dataset
uv run eval.py results/ -v 3.13

# Evaluate every Python version on the v2 dataset
uv run eval.py results/ -p v2

# Evaluate a specific list of .pyc files
uv run eval.py results/ -l pylingualv1/313-pyc-list.txt
```

When evaluating by dataset (`-p`/`-v`), results for each Python version are written to their own subdirectory, `results/python-<version>/pylingual-<timestamp>/`. When using `-l`, results go directly to `results/pylingual-<timestamp>/`.

## Output Format
Each evaluation creates a timestamped directory inside the requested output directory:

```
results/
└── pylingual-2026-09-23_12-34-56/
    ├── evaluation_results.csv
    ├── elapsed_time.txt
    └── decompilation_results/
        ├── ...
        └── ...
```

### `evaluation_results.csv`

Contains the file-level result for every attempted .pyc file, followed by the individual equivalence results produced for files that successfully reach equivalence checking.

| Column | Description |
| ------ | ----------- |
| pyc_file | Path to the input .pyc file |
| py_file | Parent directory name of the input file |
| identifier | FILE for file-level rows; dataset identifier for equivalence-result rows | 
| success | Whether the result succeeded | 
| category | Equal, Different, or DECOMPILER ERROR for file-level rows |
| notes | Equivalence result message or decompiler exception |

File-level results use three categories:

- `Equal` — all equivalence checks succeeded
- `Different` — decompilation completed, but one or more equivalence checks failed
- `DECOMPILER ERROR` — PyLingual raised an exception or exceeded the 300-second timeout

For example:

```csv
pyc_file,py_file,identifier,success,category,notes
python-3.13/000001/example.pyc,000001,FILE,True,Equal,
python-3.13/000002/example.pyc,000002,FILE,False,Different,
python-3.13/000003/example.pyc,,FILE,False,DECOMPILER ERROR,TimeoutError()
```
The CSV is written incrementally, so results already completed remain available when an evaluation is interrupted.

### `decompilation_results/`

Contains the Python source generated by PyLingual.

Output filenames are derived from the parent directory of each input .pyc file. For example:
```
python-3.13/000001/example.pyc
```
produces:
```
decompilation_results/000001.py
```
### `elapsed_time.txt`

Written after the full evaluation completes.
```
Elapsed Time: 2:17:43.182901
File success: 7089/10000 70.89%
```
File success reports the number of files for which every equivalence check succeeded divided by the total number of files attempted.

## Current Evaluation

[Pylingual v0.0.1](https://github.com/syssec-utd/pylingual/releases/tag/v0.0.1)

| Version | Decompilation Rate | 
|---------|--------------|
| 3.6     |    85.36%    |
|3.7      | 83.84%       |
| 3.8     | 82.53%       |
| 3.9     |   84.82%     |
| 3.10    | 84.61%       |
| 3.11    | 86.7 %       |
| 3.12    | 84.36%       |
| 3.13    | 70.96%       |
| 3.14    | 70.81%       |
| 3.15    | 70.89%       |
