# PRISM 0.2 — Native Learning Core

PRISM owns its trained weights. This implementation initializes a small neural
language model randomly and trains exclusively on a corpus you select. It uses
NumPy for numerical operations and never downloads pretrained models or weights.

The current repository originally contained only `data/corpus*.txt`; no prior
pattern learner, evolution engine, or iOS app was available to integrate.
All existing corpus files are preserved.

## Run

Requires Python 3.10+.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m prism train --corpus data/corpus.txt --out runs/native --steps 2000
python -m prism generate --model runs/native/model-int8.npz --prompt 'PRISM learns' --bytes 200
```

Training writes `model.npz`, `model-int8.npz`, and `metrics.json` locally.
These outputs are ignored by git. Run inference by copying only the model artifact
and the `prism` package to a machine with Python and NumPy; no server is required.
Use separate output directories for separate experiments.

## Architecture and evaluation

UTF-8 byte IDs (0–255) provide a transparent lossless encoding of input text,
without any imported vocabulary. A learned embedding, flattened context,
tanh hidden layer, and next-byte softmax form a 33,056-parameter default model.
Backpropagation and clipped stochastic gradient descent are implemented directly.
Generation samples bytes; malformed generated UTF-8 displays replacement characters.

Raw corpus bytes are divided contiguously 80/10/10 **before** windows are formed.
Training uses only the first split. Validation selects the best checkpoint,
including the initial checkpoint if training does not improve it. Test loss is
reported only after selection, for both full precision and quantized artifacts.
Evaluation samples up to 1,024 evenly spaced windows per split. Repeated text
across splits can still inflate results; this is not a contamination-proof benchmark.
Metrics record corpus SHA-256, seed, split sizes, parameter count, and losses.
Lower next-byte cross-entropy is better; it does not measure general reasoning.

Int8 export uses symmetric per-tensor quantization and compressed NPZ storage.
The NumPy runtime dequantizes to float32 at load time: this reduces artifact size,
not inference RAM or arithmetic precision. Default float32 weights occupy about
129 KiB, excluding Python, NumPy, activations, and archive overhead.

## Verified first run

On `data/corpus.txt`, seed 0, 300 steps, default dimensions:

| Metric | Result |
| --- | ---: |
| Initial validation loss | 5.5781 |
| Selected validation loss | 2.9783 |
| Test loss | 3.0645 |
| Int8 artifact test loss | 3.0651 |

Three tests verify numerical gradients, learning on a synthetic sequence,
full-precision round trips, quantization accuracy, deterministic generation,
and independent window boundaries. This is a functioning training foundation,
not a claim of conversational intelligence or superintelligence.

## Next milestones

1. Curate and deduplicate your data; establish an independent, versioned benchmark.
2. Compare context sizes and widths on validation, then test a frozen candidate.
3. Add a recurrent or causal attention architecture trained from scratch and an
   accelerator training backend with checkpoint/resume support.
4. Benchmark memory, speed, and quality before adopting compression changes.
5. Implement a native Swift/Metal or Core ML runtime and validate artifact parity
   on a real iOS device. The current runtime is Python, not an iOS integration.
6. Add evolution experiments only with fixed budgets and validation-based promotion;
   reserve an untouched benchmark for independent final evaluation.

Powerful-hardware acceleration, a learned tokenizer, concept learning, and automated
architecture evolution remain future work. No pretrained intelligence is needed
for these milestones.
