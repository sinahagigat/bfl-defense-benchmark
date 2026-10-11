# BFL defense benchmark

Code and results for the controlled comparative experiment in the review
"A Review of Anomaly Detection and Model Integrity Verification in
Blockchain-based Federated Learning".

The experiment runs one representative defense per detection family under identical
conditions (same non-IID data split, same malicious clients, same seeds), and measures the
gas cost of a minimal on-chain integrity layer.

## Setup

* Data: MNIST, 50 clients, non-IID split (Dirichlet, alpha = 0.5), 30 rounds,
  1 local epoch, SGD (lr 0.05, batch 32), small CNN.
* Attacks: label flipping, sign flipping, scaling (label-flipped update x10), Gaussian noise;
  10% and 30% malicious clients; plus a no-attack setting.
* Defenses: FedAvg (no defense), Krum, Multi-Krum, coordinate-wise median, trimmed mean,
  FLTrust (100-sample root set), cosine-similarity filter (mean - 1 std rule), and a
  committee-validation scheme with reputation weighting that emulates blockchain-native
  detection (5-member committee drawn each round; colluding members vote for malicious updates).
* 3 seeds per setting, 216 runs in total.

## Files

| File | Purpose |
|---|---|
| `bfl_bench.py` | Runs the experiment; appends one line per run to `results.csv` (resumable) |
| `summarize.py` | Builds the tables and figure in `summary/` from `results.csv` |
| `gas_bench.py` | Gas cost of the on-chain layer on a local in-memory Ethereum chain |
| `FLLedger.sol` | Smart contract measured by `gas_bench.py` (compiled bytecode is embedded) |
| `results.csv` | Results of the 216 runs reported in the paper |
| `gas_results.csv` | Gas measurements reported in the paper |
| `summary/` | Tables and figure produced by `summarize.py` |

## Reproduce

Python 3.10 or newer.

    pip install -r requirements.txt
    python bfl_bench.py --quick      # smoke test, about 2 minutes
    python bfl_bench.py              # full grid, several hours on a CPU
    python summarize.py

Gas measurement (on Windows the install needs the Microsoft C++ Build Tools;
it installs directly on Linux, macOS and Google Colab):

    pip install -r requirements-gas.txt
    python gas_bench.py

Gas is deterministic, so `gas_results.csv` does not depend on the machine.

## Limitations

This is a small, illustrative benchmark: one dataset, 50 clients, a small model, and a simplified
emulation of committee-based blockchain validation. It is meant to show how the defense families
compare under identical conditions, not to rank published systems.

## License

MIT
