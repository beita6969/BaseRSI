# BaseRSI: Recursive Self-Improvement of a Base Vision-Language Model via Training-Free Harness Internalization

<div align="center">

### Harness entries found by the model itself, written into its weights by training-free edits, no harness at deployment

<a href="https://huggingface.co/beita6969/BaseRSI-27B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face model BaseRSI-27B"></a>
<a href="https://huggingface.co/datasets/beita6969/BaseRSI-Dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E" alt="Hugging Face dataset BaseRSI-Dataset"></a>

<img src="figs/figure1.png" alt="BaseRSI overview" width="60%">

</div>

## Overview

BaseRSI makes a base vision-language model stronger by letting it improve its **harness** and then write what the
harness adds into its own weights. It follows a paradigm of *Recursive Harness Internalization*: an outer loop
evolves the harness, the entries that rewrite a request with a procedure in the prompt or a finer view of the image,
and an inner loop moves every accepted entry into the weights through training-free edits, round after round on the
model's own responses. The base model fills every role with forward passes alone, and the edited checkpoint answers
the original request without the harness. The repository implements the three components of the paper:

- **Recursive Harness Internalization**: two nested loops. In the outer loop the base model proposes entries from
  its own failures and an acceptance test keeps those whose gain generalizes. In the inner loop the base model with
  an accepted entry in the request is the teacher, the same model on the original request is the student, and the
  edit is refitted in rounds on the student's own responses.
- **ReGap**: every transformer block is supervised with the teacher–student gap, the difference that the entry
  makes at that block alone, and keeps it as a low-dimensional update of the block's MLP weights.
- **ReKey**: the site of an edit is measured, the gaps are written by regression into a few key–value neurons of
  every block, and the strength of the edit is measured on held-out responses.

## Method

<div align="center">
  <img src="figs/figure3.png" alt="BaseRSI framework" width="95%">
</div>

### Harness entries

A harness is a set of entries. An entry has a scope and an action: the scope, answered by the base model as a
router, tells whether the entry applies to a request, and the action rewrites the request by adding a procedure to
the prompt or a finer view of the image. The entries are the parameters of the harness, as the weights are those of
the model.

### Recursive Harness Internalization

**Outer loop.** In every generation the base model reads its own answers on the fitting questions and proposes
candidate entries. A candidate is scored on the validation questions against the current system at an equal token
budget and kept when its gain generalizes:

```text
accept c  if  p_gen(c) = max over b of p({d_b'(c)}, b' != b) <= alpha_c   and no benchmark is significantly worse
```

d_b(c) is the paired accuracy gain of adding c on the validation questions of benchmark b, p is the one-sided
sign-flip p-value of the macro mean, and alpha_c is the Holm threshold over the candidates of the generation. An
entry for one benchmark is admitted when its gain there is significant and no other benchmark is significantly worse.

**Inner loop.** An accepted entry still acts through the request, so the inner loop moves it into the weights. The
edit is fitted in rounds on the student's own responses:

```text
D(j) = D(j-1) + D(y),     y ~ student with edit e(j-1),     e(j) = ReKey(D(j))
```

The first two rounds replay the responses of the teacher and of the unedited student, the later rounds those of the
edited student. Every round computes its labels D(y) with the unedited base model, and the refit on the pooled
responses replaces the previous edit. Every edit is fitted on the base weights and takes its own neurons or the
patch kernel, so the edits add up in one checkpoint without refitting.

### ReGap

The gap of block l is what the entry adds at that block alone: the block is run on the teacher's response states
with and without the entry in the request,

```text
g_l = F_l(s^c_{l-1}; z^c_{l-1}) - F_l(s^c_{l-1}; z_{l-1})
```

where F_l is the block, z and s are the states at the prompt and response positions, and c marks the teacher. The
student keeps the gap as a small set of key–value neurons in the block's MLP:

```text
E_l(x) = gamma * w_beta(k_l . x) * sum_{i=0..r} (u_{l,i} . x) v_{l,i},     w_beta(z) = z * sigmoid(beta * z)
```

Each term is one SwiGLU neuron: its gate row is the key k_l, its up row the readout u, its down column the value v.
An edit rewrites r + 1 neurons per block; every other weight stays bit-identical and the MLP keeps its architecture.

### ReKey

- **Site.** Before any fit, the student's own responses are replayed with the outputs of the linear-attention
  layers set to those of the teacher, and the fraction of the divergence from the teacher that this removes is
  measured. A high fraction sends an entry to the MLP neurons; a view entry with a low fraction goes to the patch
  embedding, where the patch kernel K is resampled to K' = B^T K for the up-sampling B of the image. In the released
  pipeline the fraction is measured and recorded for the entry that the rounds compile, and the view entry is
  written into the patch embedding once it is accepted.

  ```text
  eta(c) = 1 - KL(teacher || student with the teacher's linear-attention outputs) / KL(teacher || student)
  ```

- **Reading.** Teacher-forced forward passes of the base model over a response give the pairs (x_l(t), g_l(t)) of
  MLP input and gap at sampled response positions.
- **Writing.** For every block, a ridge regression reads the gap from the MLP input and is reduced to the r
  directions that carry most of its predictions. One more neuron carries the mean gap, and the key is fitted by
  ridge regression to tell the response positions where the entry applies from all others.

  ```text
  W_l = argmin over W of E || g - b_l - W^T P_l x ||^2 + lambda * ||W||^2
  ```

- **Strength.** The step is chosen by a line search on held-out student responses:

  ```text
  gamma = argmin over Gamma of KL(teacher || student with the edit at step gamma)
  ```

## Repository Layout

```text
basersi/cli/                        entry points: basersi <command> (python -m basersi.cli)
basersi/pipeline.py                 all steps in order from one config
basersi/data/                       development questions from the official benchmark files
basersi/engine/                     model server, request dispatcher, reads, router, grader
basersi/loop/                       outer loop: proposer, generations, reports
basersi/judge/                      acceptance tests: sign-flip statistics, equal-token comparison
basersi/view_edit/                  view entry into the patch embedding (kernel resampling, image-token bounds)
basersi/procedure_edit/lab.py       paired teacher and student passes of the base model
basersi/procedure_edit/gap_capture.py   ReGap: block gaps and MLP inputs
basersi/procedure_edit/site.py      ReKey: site
basersi/procedure_edit/fit.py       ReKey: ridge fit and rank reduction
basersi/procedure_edit/key.py       ReKey: keys
basersi/procedure_edit/select.py    ReKey: line search of the step
basersi/procedure_edit/rounds.py    rounds of the inner loop
basersi/export/                     writes the edits into one checkpoint
basersi/config/                     configuration of the paper and the manifest of the development questions
figs/                               figures from the paper
```

## Requirements

- Python 3.12, PyTorch 2.14, Transformers 5.20, vLLM 0.33 (pinned in `pyproject.toml`)
- GPUs with 80 GB of memory (H100 in the paper): the servers of the outer loop, two cards for the reads of the
  acceptance tests, and one card for the Procedure Edit

All roles are the base model Qwen3.8-27B: it answers, proposes the entries and routes them. No external API is
called and no gradient is computed.

## Installation

```bash
git clone https://github.com/beita6969/BaseRSI.git BaseRSI
cd BaseRSI
pip install ".[gpu,sources]"
```

## Dataset

The paper evaluates ten visual reasoning benchmarks: five in-distribution (ID) benchmarks supply the development
questions and the ID tests, and five out-of-distribution (OOD) benchmarks are used for testing only. Every set has
128 questions per benchmark.

**ID**:

| Benchmark | Fitting | Validation | Test | Source | License |
| --- | --- | --- | --- | --- | --- |
| BabyVision | 54 | 74 | 128 | `UnipatAI/BabyVision` | MIT |
| BLINK | 54 | 74 | 128 | `BLINK-Benchmark/BLINK` | Apache-2.0 |
| CharXiv | 54 | 74 | 128 | `princeton-nlp/CharXiv` | CC BY-SA 4.0 |
| VisuLogic | 54 | 74 | 128 | `VisuLogic/VisuLogic` | Apache-2.0 |
| VLMs Are Biased | 54 | 74 | 128 | `anvo25/vlms-are-biased` | MIT |

**OOD**:

| Benchmark | Test | Source | License |
| --- | --- | --- | --- |
| ChartQAPro | 128 | `ahmed-masry/ChartQAPro` | MIT |
| HallusionBench | 128 | `lmms-lab/HallusionBench` | BSD-3-Clause |
| PerceptionBench | 128 | `moonshotai/PerceptionBench` | CC BY-NC 4.0 |
| SpatialViz | 128 | `PLM-Team/Spatial-Visualization-Benchmark` | MIT |
| VisualPuzzles | 128 | `neulab/VisualPuzzles` | MIT |

The 640 development questions (fitting and validation) are the only data the pipeline reads. The repository ships
their manifest, `basersi/config/development_manifest.jsonl` (identifiers, roles and hashes), and `basersi data`
rebuilds the questions from the official files pinned in `basersi/config/sources.yaml` and checks every row against
it. Entries are proposed and edits are fitted on the fitting questions only; the acceptance tests score on the
validation questions.

The test sets (640 ID and 640 OOD questions with their images) are in the
[BaseRSI-Dataset](https://huggingface.co/datasets/beita6969/BaseRSI-Dataset). They share no question, image or
near-duplicate image with the development questions. The repository holds the training-free pipeline only;
evaluation on the test sets is not part of it.

## Running

```bash
basersi pipeline --config my_config.yaml --only data                                # build the development questions
basersi serve server --model /path/to/Qwen3.8-27B --queue-dir runs/queue            # model server of the outer loop
basersi serve dispatcher --queue-dir runs/queue --rows data/development.jsonl       # sends the loop's requests to the servers
basersi pipeline --config my_config.yaml                                            # all steps
```

A minimal config:

```yaml
work_dir: runs/basersi
data_dir: data
model: /path/to/Qwen3.8-27B
data:
  sources: /path/to/official_benchmark_files
loop:
  queue_dir: runs/queue
```

`basersi pipeline` runs the steps in order: development questions, outer loop, acceptance of its entries, View
Edit, its acceptance, Procedure Edit, export, and the acceptance of the Procedure Edit. It stops at an acceptance
gate when an entry or an edit is not accepted, and the same command resumes a stopped run. `--list` prints the
steps, and `--only`, `--from` and `--to` run a part. Every command lists its inputs and options with `--help`.

Each step writes into `work_dir`: the state of the run (`pipeline_state.json`), the run folder of the loop, the
verdicts of the acceptance tests, the View Edit checkpoint, the files of the Procedure Edit, and the exported
checkpoint.

Settings of the paper (`basersi/config/`):

| Category | Setting |
| --- | --- |
| Base model | Qwen3.8-27B (64 blocks) with its original weights as executor, proposer and router |
| Acceptance test | alpha = 0.10 with the Holm correction over the candidates of a generation; one-sided sign-flip over picture groups, 20,000 flips |
| View Edit | patch size 16 to 8; 2,048 to 4,096 image tokens per image |
| Procedure Edit | all 64 blocks; rank r = 64, 65 neurons per block; J = 3 rounds |
| Ridge weight | {0.03, 0.3, 3} x mean diagonal of the input covariance, chosen on a held-out fold |
| Key | slope beta = 8, threshold 0.25 |
| Step | line search over {0.5, 0.75, 1, 1.25, 1.5, 1.8, 2, 2.6}; `--step 1.8` gives the step of the published model |
| Positions per response | 5 template-tail positions and up to 512 (round 1) or 256 (rounds 2 and 3) reasoning positions |

## Model

The edited checkpoint of the paper is [BaseRSI-27B](https://huggingface.co/beita6969/BaseRSI-27B). It loads as the
base model does and answers the original request.

## License

This repository is released for research use. Please also follow the licenses and terms of the upstream model,
datasets and benchmarks used with BaseRSI; the sources and licenses of the benchmark files and prompt templates are
listed in `basersi/config/sources.yaml`.
