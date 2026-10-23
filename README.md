# BFL defense benchmark

Code and results for the controlled comparative experiment in the review
"A Systematic Review of Anomaly Detection and Model Integrity Verification in
Blockchain-based Federated Learning".

The experiment compares statistical and similarity-based aggregation rules, a trust-based
filter (FLTrust), and a simplified emulation of committee-based blockchain detection under
identical conditions (same non-IID data split, same malicious clients, same seeds), and measures
the gas cost of a minimal on-chain integrity layer. Learning-based anomaly detectors are not
included.

## Setup

* Data: MNIST and Fashion-MNIST, 50 clients, non-IID split (Dirichlet, alpha = 0.5), 30 rounds,
  1 local epoch, SGD (lr 0.05, batch 32), small CNN (2 conv layers 16/32, FC 64).
* Attacks: label flipping, sign flipping, scaling (label-flipped update x10), Gaussian noise,
  and a non-omniscient form of the adaptive "a little is enough" attack (ALIE, Baruch et al.,
  2019), in which the attackers compute honest updates on their own data in every round and send
  mean - z * std of those updates (they do not see honest clients' updates), with
  z = Phi^-1((n - s) / n), s = floor(n / 2 + 1) - m, so z = 0.20 for 10% and 0.77 for 30% attackers.
  Variants in which the attackers also see the honest clients' updates can be stronger.
* Attacker shares: 10% and 30% malicious clients, plus a no-attack setting.
* Defenses: FedAvg (no defense), Krum, Multi-Krum, coordinate-wise median, trimmed mean,
  FLTrust (100-sample root set), cosine-similarity filter (mean - 1 std rule), and a
  committee-validation scheme with reputation weighting that emulates blockchain-native
  detection (5-member committee drawn each round; colluding members vote for malicious updates).
* 3 seeds (1, 2, 3) per setting: 264 runs per dataset, 528 runs in total.

## Detection metrics (Table 13)

For the rules that accept or reject individual updates, every run counts, over all 30 rounds, how many malicious and honest
updates were rejected (`bfl_bench.py`, variables `tp`, `fp`, `fn`, `tn`):

* `det_recall` = rejected malicious updates / all malicious updates
* `det_fpr` = rejected honest updates / all honest updates

Table 13 reports the mean of these two values over the three seeds. Rules that keep a fixed
number of updates give fixed rates by construction: Krum keeps one update per round, so its
false-positive rate is 44/45 (10% attackers) or 34/35 (30% attackers); Multi-Krum always
discards m updates, so when it accepts every ALIE update its false-positive rate is 5/45 or
15/35. Table 13 lists Multi-Krum, FLTrust, the cosine filter and the committee; Krum (fixed
rates, see above) is in `results.csv` but not in Table 13, and the median and the trimmed mean
do not accept or reject whole updates.

## Files

| File | Purpose |
|---|---|
| `bfl_bench.py` | Runs the experiment; appends one line per run to `results.csv` (resumable) |
| `summarize.py` | Builds the tables and figure in `summary/` from `results.csv` |
| `gas_bench.py` | Gas cost of the on-chain layer on a local in-memory Ethereum chain |
| `FLLedger.sol` | Smart contract measured by `gas_bench.py` (compiled bytecode is embedded) |
| `requirements-lock.txt` | Exact package versions used to produce `results.csv` |
| `results.csv` | Results of the 528 runs reported in the paper (Tables 12 and 13) |
| `gas_results.csv` | Gas measurements reported in the paper |
| `summary/` | Tables and figure produced by `summarize.py` |

## Reproduce

Python 3.10 or newer.

    pip install -r requirements.txt               # or requirements-lock.txt for the exact versions used
    python bfl_bench.py --quick                  # smoke test, about 2 minutes
    python bfl_bench.py --datasets mnist fmnist  # full grid (528 runs), about 14 hours on a laptop CPU
    python summarize.py                          # tables and figures in summary/

`summary/table_accuracy_<dataset>.csv` corresponds to Table 12 and
`summary/table_detection_<dataset>.csv` to Table 13 of the paper. All settings
(seeds, number of clients, rounds, learning rate, attack strengths) are command-line
options of `bfl_bench.py` with the defaults used in the paper; `python bfl_bench.py --help`
lists them. Runs are appended to `results.csv`, so an interrupted run can be resumed by
starting the same command again. Results were produced on CPU (PyTorch 2.14, Python 3.14);
small numerical differences can occur on other hardware.

Gas measurement (on Windows the install needs the Microsoft C++ Build Tools;
it installs directly on Linux, macOS and Google Colab):

    pip install -r requirements-gas.txt
    python gas_bench.py

Gas is deterministic, so `gas_results.csv` does not depend on the machine. The contract was
compiled with solc 0.8.24, and the measurement runs on the in-memory test chain of
`eth-tester` (py-evm backend); it covers only the contract in `FLLedger.sol`, not model storage,
identity management or production consensus.

## Limitations

This is a small, illustrative benchmark: two image datasets, 50 clients, a small model, and a
simplified emulation of committee-based blockchain validation. It is meant to show how the defense families
compare under identical conditions, not to rank published systems.

## License

MIT
