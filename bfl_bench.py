"""
Small, reproducible benchmark of poisoning defenses for (blockchain-based) federated learning.

One representative defense per detection family reviewed in the paper is run under identical
conditions (same data split, same attacks, same malicious clients, same seeds):

  fedavg      no defense (weighted federated averaging)
  krum        Krum (Blanchard et al., 2017)
  multikrum   Multi-Krum (Blanchard et al., 2017)
  median      coordinate-wise median (Yin et al., 2018)
  trimmed     coordinate-wise trimmed mean (Yin et al., 2018)
  fltrust     FLTrust with a small clean root dataset at the aggregator (Cao et al., 2021)
  cosine      statistical similarity filter: cosine to the coordinate-wise median, reject below
              mean - 1 std (the one-standard-deviation rule used by several reviewed studies)
  committee   blockchain-native emulation: a randomly drawn committee scores every update by its
              marginal effect on the validation loss of the committee member's own data (loss of the
              full average minus loss without that update), the median committee score decides
              acceptance (median + 2 MAD rule), and accepted
              clients are weighted by an on-chain style reputation (EMA of past acceptances)

Attacks (applied by a fixed set of malicious clients):
  none        all clients honest
  labelflip   malicious clients train on flipped labels (y -> 9 - y)
  signflip    malicious clients send the negated honest update
  scaling     label-flipped update boosted by a large factor (model-replacement style)
  gaussian    malicious clients send random Gaussian noise
  alie        "A little is enough" (Baruch et al., 2019): malicious clients estimate the mean and
              standard deviation of honest updates from their own honest updates and all send
              mean - z * std, with z chosen so the update stays inside the spread of honest updates

Results are appended to a CSV file, one line per run; finished runs are skipped on restart,
so the script can be stopped and resumed at any time.

Usage examples:
  python bfl_bench.py --quick                 # smoke test, about 1-2 minutes
  python bfl_bench.py                         # full MNIST grid
  python bfl_bench.py --datasets mnist fmnist # both datasets
"""
import argparse
import csv
import itertools
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFENSES = ["fedavg", "krum", "multikrum", "median", "trimmed", "fltrust", "cosine", "committee"]
ATTACKS = ["none", "labelflip", "signflip", "scaling", "gaussian", "alie"]
FILTERING = {"krum", "multikrum", "fltrust", "cosine", "committee"}  # defenses that flag clients


# ----------------------------------------------------------------------------- data
def load_dataset(name, data_dir, synthetic=False):
    if synthetic:  # only for testing the pipeline without internet access
        g = torch.Generator().manual_seed(0)
        protos = torch.randn(10, 1, 28, 28, generator=g)
        def make(n):
            y = torch.randint(0, 10, (n,), generator=g)
            x = protos[y] + 0.8 * torch.randn(n, 1, 28, 28, generator=g)
            return x, y
        xtr, ytr = make(12000)
        xte, yte = make(2000)
        return xtr, ytr, xte, yte
    from torchvision import datasets
    cls = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST}[name]
    tr = cls(data_dir, train=True, download=True)
    te = cls(data_dir, train=False, download=True)
    mean, std = (0.1307, 0.3081) if name == "mnist" else (0.2860, 0.3530)
    def prep(ds):
        x = ds.data.float().div(255.0).unsqueeze(1)
        return (x - mean) / std, ds.targets.clone().long()
    xtr, ytr = prep(tr)
    xte, yte = prep(te)
    return xtr, ytr, xte, yte


def dirichlet_split(y, n_clients, alpha, rng):
    """Non-IID split: for each class, client shares are drawn from Dirichlet(alpha)."""
    y = y.numpy()
    idx = [[] for _ in range(n_clients)]
    for c in range(10):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        p = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
        for k, part in enumerate(np.split(ids, cuts)):
            idx[k].extend(part.tolist())
    # make sure every client holds at least 20 samples
    pool = rng.permutation(len(y)).tolist()
    for k in range(n_clients):
        while len(idx[k]) < 20:
            idx[k].append(pool.pop())
    return [np.array(sorted(i)) for i in idx]


# ----------------------------------------------------------------------------- model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 5)
        self.c2 = nn.Conv2d(16, 32, 5)
        self.f1 = nn.Linear(32 * 4 * 4, 64)
        self.f2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


def get_vec(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_vec(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p))
        i += n


def local_train(model, gvec, x, y, epochs, bs, lr, gen):
    set_vec(model, gvec)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    model.train()
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            F.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
    return get_vec(model) - gvec  # the update (delta)


@torch.no_grad()
def evaluate(model, vec, x, y, bs=2000):
    set_vec(model, vec)
    model.eval()
    correct, flipped = 0, 0
    for s in range(0, len(y), bs):
        pred = model(x[s:s + bs]).argmax(1)
        correct += (pred == y[s:s + bs]).sum().item()
        flipped += (pred == 9 - y[s:s + bs]).sum().item()
    return correct / len(y), flipped / len(y)


# ----------------------------------------------------------------------------- aggregation rules
def pairwise_sq_dists(U):
    sq = (U * U).sum(1)
    return (sq[:, None] + sq[None, :] - 2 * U @ U.t()).clamp_min(0)


def krum_select(U, f, m):
    n = U.shape[0]
    k = max(n - f - 2, 1)
    D = pairwise_sq_dists(U)
    D.fill_diagonal_(float("inf"))
    scores = D.topk(k, dim=1, largest=False).values.sum(1)
    return scores.argsort()[:m]


def aggregate(defense, U, sizes, f, ctx):
    """Returns (aggregated update, boolean mask of clients that were accepted, or None)."""
    n = U.shape[0]
    w = sizes / sizes.sum()
    if defense == "fedavg":
        return (w[:, None] * U).sum(0), None
    if defense == "krum":
        sel = krum_select(U, f, 1)
        mask = torch.zeros(n, dtype=torch.bool); mask[sel] = True
        return U[sel].mean(0), mask
    if defense == "multikrum":
        sel = krum_select(U, f, n - f)
        mask = torch.zeros(n, dtype=torch.bool); mask[sel] = True
        return U[sel].mean(0), mask
    if defense == "median":
        return U.median(0).values, None
    if defense == "trimmed":
        b = min(f, (n - 1) // 2)
        S = U.sort(0).values
        return S[b:n - b].mean(0), None
    if defense == "fltrust":
        g0 = ctx["server_update"]()
        cos = F.cosine_similarity(U, g0[None, :], dim=1)
        ts = cos.clamp_min(0)
        if ts.sum() == 0:
            return torch.zeros_like(g0), torch.zeros(n, dtype=torch.bool)
        scaled = U * (g0.norm() / U.norm(dim=1).clamp_min(1e-12))[:, None]
        return (ts[:, None] * scaled).sum(0) / ts.sum(), ts > 0
    if defense == "cosine":
        ref = U.median(0).values
        cos = F.cosine_similarity(U, ref[None, :], dim=1)
        mask = cos >= cos.mean() - cos.std()
        norms = U.norm(dim=1)
        mask &= norms <= 3 * norms.median()
        if mask.sum() == 0:
            mask[:] = True
        ww = w[mask] / w[mask].sum()
        return (ww[:, None] * U[mask]).sum(0), mask
    if defense == "committee":
        return committee_aggregate(U, w, ctx)
    raise ValueError(defense)


def committee_aggregate(U, w, ctx):
    """Blockchain-native emulation: committee validation + reputation-weighted aggregation."""
    n = U.shape[0]
    rng, rep, malicious = ctx["rng"], ctx["reputation"], ctx["malicious"]
    members = rng.choice(n, size=ctx["committee_size"], replace=False)
    mal_t = torch.as_tensor(malicious)
    scores = torch.zeros(len(members), n)
    for r, m in enumerate(members):
        if malicious[m]:
            # a colluding committee member reports the best score for malicious updates
            # and the worst score for honest ones
            scores[r] = torch.where(mal_t, torch.tensor(-1e6), torch.tensor(1e6))
        else:
            # marginal contribution: loss of the full average minus loss of the average without
            # update i (positive = including i makes the model worse on this member's data)
            full = U.mean(0)
            l_full = ctx["val_loss"](m, ctx["gvec"] + full)
            scores[r] = torch.tensor([l_full - ctx["val_loss"](m, ctx["gvec"] + (full * n - U[i]) / (n - 1))
                                      for i in range(n)])
    score = scores.median(0).values               # robust to a minority of dishonest members
    med = score.median()
    mad = (score - med).abs().median() * 1.4826
    mask = score <= med + 2 * mad                  # reject updates whose marginal effect on loss is abnormally bad
    if mask.sum() == 0:
        mask[score.argmin()] = True
    rep.mul_(0.8).add_(0.2 * mask.float())       # reputation ledger (exponential moving average)
    weight = w * rep * mask.float()
    weight = weight / weight.sum()
    ctx["accept_log"].append(mask.clone())
    return (weight[:, None] * U).sum(0), mask


# ----------------------------------------------------------------------------- one run
def run_one(cfg, data, dataset, attack, ratio, defense, seed, device):
    xtr, ytr, xte, yte = data
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    n = cfg.clients
    parts = dirichlet_split(ytr, n, cfg.alpha, rng)
    # the aggregator's small clean root set (FLTrust) is drawn from the training pool
    root = torch.as_tensor(rng.choice(len(ytr), size=cfg.root_size, replace=False))
    n_mal = int(round(ratio * n)) if attack != "none" else 0
    mal_ids = rng.choice(n, size=n_mal, replace=False)
    malicious = np.zeros(n, dtype=bool); malicious[mal_ids] = True
    f = max(n_mal, 1)  # number of tolerated attackers assumed by Krum / trimmed mean
    # ALIE: z = Phi^-1((n - s) / n) with s = floor(n / 2 + 1) - n_mal supporters needed
    from statistics import NormalDist
    s_need = int(n / 2 + 1) - n_mal
    alie_z = NormalDist().inv_cdf((n - s_need) / n) if n_mal > 0 else 0.0

    cx = [xtr[p].to(device) for p in parts]
    cy = [ytr[p].to(device) for p in parts]
    cy_flip = [9 - y for y in cy]
    sizes = torch.tensor([len(p) for p in parts], dtype=torch.float32)
    xte, yte = xte.to(device), yte.to(device)
    rx, ry = xtr[root].to(device), ytr[root].to(device)

    model = CNN().to(device)
    gvec = get_vec(model).clone()
    rep = torch.ones(n)
    accept_log = []

    def server_update():
        return local_train(model, gvec, rx, ry, cfg.local_epochs, cfg.batch, cfg.lr, gen).cpu()

    def val_loss(member, vec):
        # each committee member validates on (up to) 200 of its own samples
        x, y = cx[member][:200], cy[member][:200]
        set_vec(model, vec.to(device))
        model.eval()
        with torch.no_grad():
            return F.cross_entropy(model(x), y).item()

    ctx = {"rng": rng, "reputation": rep, "malicious": malicious, "committee_size": cfg.committee,
           "val_loss": val_loss, "server_update": server_update, "accept_log": accept_log}

    tp = fp = fn = tn = 0
    agg_time = 0.0
    t0 = time.time()
    for rnd in range(cfg.rounds):
        ups = []
        for k in range(n):
            if malicious[k] and attack in ("labelflip", "scaling"):
                d = local_train(model, gvec, cx[k], cy_flip[k], cfg.local_epochs, cfg.batch, cfg.lr, gen)
                if attack == "scaling":
                    d = d * cfg.boost
            elif malicious[k] and attack == "gaussian":
                d = torch.randn(gvec.numel(), generator=gen).to(device) * cfg.noise_std
            else:
                d = local_train(model, gvec, cx[k], cy[k], cfg.local_epochs, cfg.batch, cfg.lr, gen)
                if malicious[k] and attack == "signflip":
                    d = -d
            ups.append(d.cpu())
        U = torch.stack(ups)
        if attack == "alie" and n_mal > 0:
            mal_t = torch.as_tensor(malicious)
            own = U[mal_t]                       # honest updates computed on the attackers' own data
            mu, sd = own.mean(0), own.std(0, unbiased=False)
            U[mal_t] = (mu - alie_z * sd)[None, :]
        ctx["gvec"] = gvec.cpu()
        ta = time.time()
        agg, mask = aggregate(defense, U, sizes, f, ctx)
        agg_time += time.time() - ta
        gvec = gvec + agg.to(device)
        if mask is not None and n_mal > 0:
            mal_t = torch.as_tensor(malicious)
            rejected = ~mask
            tp += (rejected & mal_t).sum().item()
            fp += (rejected & ~mal_t).sum().item()
            fn += (~rejected & mal_t).sum().item()
            tn += (~rejected & ~mal_t).sum().item()

    acc, flip_rate = evaluate(model, gvec, xte, yte)
    det_recall = tp / (tp + fn) if (tp + fn) else float("nan")
    det_fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "dataset": dataset, "attack": attack, "mal_ratio": ratio if attack != "none" else 0.0,
        "defense": defense, "seed": seed, "test_acc": round(acc, 4),
        "flip_rate": round(flip_rate, 4),
        "det_recall": round(det_recall, 4) if det_recall == det_recall else "",
        "det_fpr": round(det_fpr, 4) if det_fpr == det_fpr else "",
        "agg_ms_per_round": round(1000 * agg_time / cfg.rounds, 2),
        "run_s": round(time.time() - t0, 1),
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist"], choices=["mnist", "fmnist"])
    ap.add_argument("--attacks", nargs="+", default=ATTACKS, choices=ATTACKS)
    ap.add_argument("--defenses", nargs="+", default=DEFENSES, choices=DEFENSES)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID concentration")
    ap.add_argument("--root_size", type=int, default=100, help="FLTrust clean root set size")
    ap.add_argument("--committee", type=int, default=5)
    ap.add_argument("--boost", type=float, default=10.0, help="scaling attack factor")
    ap.add_argument("--noise_std", type=float, default=1.0, help="gaussian attack std")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--synthetic", action="store_true", help="synthetic data (pipeline test only)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    cfg = ap.parse_args()

    if cfg.quick:
        cfg.rounds, cfg.clients, cfg.seeds = 3, 20, [1]
        cfg.attacks, cfg.ratios = ["none", "labelflip"], [0.3]
        if cfg.out == "results.csv":
            cfg.out = "results_quick.csv"

    device = "cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu"
    print(f"device={device}  torch={torch.__version__}  python={platform.python_version()}")

    done = set()
    if os.path.exists(cfg.out):
        with open(cfg.out, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["dataset"], r["attack"], float(r["mal_ratio"]), r["defense"], int(r["seed"])))

    jobs = []
    for ds, att, ratio, dfn, seed in itertools.product(cfg.datasets, cfg.attacks, cfg.ratios,
                                                         cfg.defenses, cfg.seeds):
        r = 0.0 if att == "none" else ratio
        key = (ds, att, r, dfn, seed)
        if key in done or key in [j[:5] for j in jobs]:
            continue
        jobs.append(key)
    print(f"{len(jobs)} runs to do, {len(done)} already in {cfg.out}")

    fields = ["dataset", "attack", "mal_ratio", "defense", "seed", "test_acc", "flip_rate",
              "det_recall", "det_fpr", "agg_ms_per_round", "run_s"]
    new_file = not os.path.exists(cfg.out)
    cache = {}
    for i, (ds, att, r, dfn, seed) in enumerate(jobs, 1):
        if ds not in cache:
            cache = {ds: load_dataset(ds, cfg.data_dir, cfg.synthetic)}
        res = run_one(cfg, cache[ds], ds, att, r, dfn, seed, device)
        with open(cfg.out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            if new_file:
                wr.writeheader(); new_file = False
            wr.writerow(res)
        print(f"[{i}/{len(jobs)}] {ds} {att} r={r} {dfn} seed={seed} "
              f"acc={res['test_acc']} recall={res['det_recall']} fpr={res['det_fpr']} ({res['run_s']}s)",
              flush=True)


if __name__ == "__main__":
    main()
