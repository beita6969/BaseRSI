# BaseRSI

Code for *BaseRSI: Recursive Self-Improvement of a Base Vision-Language Model via Training-Free Harness Internalization*.

BaseRSI lets a base vision-language model improve its harness and then write what the harness adds into its own weights, through training-free edits computed from forward passes alone.

- **Outer loop.** The base model proposes harness entries from its own failures, and an acceptance test keeps the entries whose gain generalizes.
- **Inner loop.** The base model with an accepted entry in the request is the teacher and the same model on the original request is the student. The difference the entry makes at every block is written into the weights, round after round on the student's own responses.
- **Result.** One checkpoint with the same architecture as the base model, which answers the original request alone.

## Install

Python 3.12.

```bash
pip install ".[gpu]"            # the pipeline
pip install ".[gpu,sources]"    # also rebuilds the development questions from the official benchmark files
```

## Commands

| Command | What it does |
|---|---|
| `basersi data` | builds the development questions from the official benchmark files and checks them against the shipped manifest |
| `basersi serve` | starts a model server, or the dispatcher that sends the loop's requests to the servers |
| `basersi loop` | runs generations of the outer loop |
| `basersi judge` | runs the acceptance tests |
| `basersi view-edit` | writes a view entry into the patch embedding |
| `basersi procedure-edit` | writes a procedure entry into MLP neurons over several teacher and student rounds |
| `basersi export` | writes the edits into one checkpoint |
| `basersi pipeline` | runs all steps in order from one config |

Every command lists its inputs and options with `--help`.

## Run

1. Download the base model `Qwen/Qwen3.8-27B` and the official files of the five benchmarks named in `basersi/config/sources.yaml`.
2. Write a config that sets `work_dir`, `data_dir`, `model`, `data.sources` and `loop.queue_dir`; every other key has a default in `basersi/config/`.
3. Start a model server and the dispatcher (`basersi serve server`, `basersi serve dispatcher`).
4. Run the pipeline:

```bash
basersi pipeline --config my_config.yaml --list    # show the steps
basersi pipeline --config my_config.yaml           # run them
```

## Data and Model

- Development questions: 640 questions from five benchmarks. The repository ships their manifest (identifiers and hashes); `basersi data build` rebuilds the questions from the official files.
- Evaluation questions: <https://huggingface.co/datasets/beita6969/BaseRSI-Dataset>
- Edited checkpoint: <https://huggingface.co/beita6969/BaseRSI-27B>
