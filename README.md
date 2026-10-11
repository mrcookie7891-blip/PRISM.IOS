# PRISM 0.2 — Native Learning Core

PRISM trains its own weights from random initialization on text you select.
There are no pretrained weights, imported model vocabularies, model downloads,
or hosted inference calls. NumPy and optional PyTorch provide numerical operations.

This milestone builds on the letter mapper and evolution engine from
`cursor/prism-self-evolving-ai-c8b3` (the `main` branch contains datasets only).
The original pattern learner remains available through `run_prism.py`.

## Train your native model

Requires Python 3.10+:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -v
python -m unittest discover -s tests -v
python run_native.py train data --output runs/prism/native.npz --steps 2000
python run_native.py export runs/prism/native.npz --output runs/prism/local-int8.npz --int8
python run_native.py generate runs/prism/local-int8.npz --prompt 'I am PRISM' --length 200
```

`train` starts a new model each time. It writes a checkpoint and adjacent
`.report.json` containing source and split hashes, settings, validation results,
test results, parameter count, and a unigram comparison. Use distinct output
paths for experiments. Checkpoints do not currently resume optimizer state.

## Powerful training, local inference

The network learns a character vocabulary from the training split only. It uses
learned character embeddings, a flattened fixed context, a tanh hidden layer,
and a next-character softmax. Unknown held-out characters have a separate token;
padding never appears as a prediction. Manual backpropagation, gradient clipping,
and Adam train the NumPy path.

For accelerator training, install PyTorch separately on the training machine
and use the same architecture and initial weights:

```sh
python run_native.py train data --device cuda --context 64 --embedding 32 --hidden 256 --steps 10000 --output runs/gpu/native.npz
```

`--device cpu` uses PyTorch on CPU, `mps` selects Apple's accelerator, and `numpy`
is the default. PyTorch is optional and never loads another model. Copy the
exported artifact to the inference machine; Python and NumPy suffice there.
The optional PyTorch backend is implemented but was not executed in this
workspace because PyTorch and accelerators were unavailable.

Int8 export uses symmetric per-tensor quantization and compressed NPZ files.
It dequantizes to float32 on load, so the smaller download/storage size does not
imply int8 arithmetic or reduced inference RAM. This release provides Python
local inference; a Swift/Metal or Core ML iOS runtime is a subsequent milestone.

## Evaluation and preserved pattern learning

Normalized unique lines are deduplicated before deterministic approximately
80/10/10 train/validation/test splitting. Contexts never cross line or split
boundaries. Vocabulary fitting and optimization see training text only.
Validation chooses between the initial and final trained checkpoint. The test
split is scored after that choice. Related passages and near duplicates can
still cross splits: these are corpus learning checks, not independent reasoning
benchmarks. A separate curated benchmark is needed before capability claims.

The evolution engine now reserves held-out text before training, rejects explicit
evaluation text contained in training, clones memory deeply, trains candidates
once, compares them with the incumbent on the same evaluation sample, and saves
exactly the winning evaluated state. Fitness is held-out log-likelihood; pattern
count and generation number no longer inflate it. Historical fields named
`intelligence_score` remain for compatibility but represent prediction quality
only. Reusing a previously contaminated state cannot undo past exposure; use a
fresh state when establishing an evaluation baseline. Changing corpora across
cycles also requires keeping evaluation data out of all historical training.

```sh
python run_prism.py data --state runs/pattern/mind.json --generations 5
python run_mapper.py data --letter t
```

A pre-existing experimental byte model (`prism/core.py`, `python -m prism`) is
also preserved. Its byte vocabulary, contiguous splits, SGD, and checkpoint
format differ from the character learner; their artifacts are not interchangeable.
Use `run_native.py` for the character learner and accelerator workflow above.

## Verified smoke run

NumPy, seed 42, 300 steps, context 16, embedding 8, hidden 64; trained on
`corpus3.txt`, `corpus4.txt`, `corpus6.txt`, and `corpus7.txt` (208 unique lines):

| Measurement | Result |
| --- | ---: |
| Trainable parameters | 13,220 |
| Initial validation cross-entropy | 4.2035 |
| Trained validation cross-entropy | 2.3670 |
| Held-out test cross-entropy | 2.3809 |
| Unigram test cross-entropy | 3.0033 |
| Int8 artifact test cross-entropy | 2.3811 |
| Int8 artifact size | About 14 KB |

Seventeen tests passed across both test suites, including numerical gradients,
learning, vocabulary handling, document boundaries, duplicate removal, clone
isolation, incumbent preservation, persistence, and export parity. These results
show a working trainable foundation; they do not establish conversational
competence, general reasoning, or superintelligence.

## Next milestones

1. Curate a separate task benchmark and measure data quality and contamination.
2. Run the accelerator path and scale data/context with measured training budgets.
3. Add optimizer checkpoint/resume and validation checks during long runs.
4. Compare a recurrent or causal attention network initialized from scratch.
5. Build and verify an iOS runtime against Python logits on real hardware.
6. Introduce concept-memory and architecture evolution experiments only when
   independent tests show improved accuracy within memory and latency budgets.
