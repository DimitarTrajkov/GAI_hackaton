"""
We train a one-shot generative model self-supervised within each style, 
and at inference combine it with a metric-aware, profile-targeted search that refines X.

We overlay chord-tone onsets placed on exact integer-beat cells while a searched fraction 
of overlay onsets are allowed off-chord. 
A trained one-shot generative model (FiLM-conditioned conv U-Net, conditioned
on the style example Z) proposes additional coherent overlay candidates.

AI-assistance disclosure: developed with the assistance of Claude (Anthropic);
all methods were reviewed and tested by the author, who is responsible for the
final submission.
"""

from __future__ import annotations
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utility import pianoroll as pr
from utility.data import HackathonDataset, load_roll_bundle
from utility.metric import (score_item, content_preservation, compute_histograms,
                            build_style_profile_from_bundles, HIST_NAMES, _cosine,
                            TIME_BINS, DURATION_BINS, VELOCITY_BINS)
from utility.submission import (bundle_to_rows, encode_notes, validate_submission,
                                SUBMISSION_COLUMNS)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NP, T = pr.NUM_PITCHES, pr.T_PER_FRAGMENT
CPB = pr.STEPS_PER_BEAT * pr.BEATS_PER_BAR          # cells per bar = 16
N_BARS, SPB = pr.BARS_PER_FRAGMENT, pr.STEPS_PER_BEAT
BB = TIME_BINS // N_BARS                             # time_pitch beat-bins per bar
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_RNG = np.random.default_rng(0)

_PC_NP = np.zeros((12, NP), np.float32)
for _p in range(NP):
    _PC_NP[_p % 12, _p] = 1.0
_PC = torch.from_numpy(_PC_NP).to(DEVICE)

# search grid: (overlay onset count, off-chord fraction)
_GRID = [(no, oc) for no in (48, 96, 160) for oc in (0.0, 0.25, 0.5, 0.8, 1.0)]


# --------------------------------------------------------------------------- #
# Roll helpers
# --------------------------------------------------------------------------- #
def combined(b):
    p = b["pitched"].max(0) if b["pitched"].size else np.zeros((NP, T), np.float32)
    return np.maximum(p, b["drum"])


def two_ch(b):
    p = b["pitched"].max(0) if b["pitched"].size else np.zeros((NP, T), np.float32)
    return np.stack([p.astype(np.float32), b["drum"].astype(np.float32)], 0)


def bar_pcs(cr):
    ch = _PC_NP @ cr
    return [set(np.where(ch[:, b*CPB:(b+1)*CPB].sum(1) > 1e-6)[0].tolist()) for b in range(N_BARS)]


def chord_skeleton(pitched_combined):
    ch = _PC_NP @ pitched_combined
    sk = np.zeros((NP, T), np.float32)
    for b in range(N_BARS):
        s, e = b*CPB, (b+1)*CPB
        for pc in np.where(ch[:, s:e].sum(1) > 1e-6)[0]:
            sk[pc:NP:12, s:e] = 1.0
    return sk


# --------------------------------------------------------------------------- #
# Generative model: style encoder + FiLM-conditioned conv U-Net
# --------------------------------------------------------------------------- #
class StyleEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 96, 4, 2, 1), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(96, dim)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))


class FiLM(nn.Module):
    def __init__(self, sdim, ch):
        super().__init__()
        self.f = nn.Linear(sdim, 2 * ch)

    def forward(self, h, s):
        g, b = self.f(s).chunk(2, -1)
        return h * (1 + g[..., None, None]) + b[..., None, None]


class Refiner(nn.Module):
    def __init__(self, sdim=64):
        super().__init__()
        self.in_conv = nn.Conv2d(1, 48, 3, 1, 1)
        self.d1, self.d2 = nn.Conv2d(48, 64, 4, 2, 1), nn.Conv2d(64, 96, 4, 2, 1)
        self.film1, self.film2 = FiLM(sdim, 64), FiLM(sdim, 96)
        self.mid = nn.Conv2d(96, 96, 3, 1, 1)
        self.u2 = nn.ConvTranspose2d(96, 64, 4, 2, 1)
        self.u1 = nn.ConvTranspose2d(128, 48, 4, 2, 1)
        self.out = nn.Conv2d(96, 2, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, skel, s):
        h0 = self.act(self.in_conv(skel))
        h1 = self.act(self.film1(self.d1(h0), s))
        h2 = self.act(self.film2(self.d2(h1), s))
        h2 = self.act(self.mid(h2))
        u2 = self.act(self.u2(h2))
        u1 = self.act(self.u1(torch.cat([u2, h1], 1)))
        return torch.sigmoid(self.out(torch.cat([u1, h0], 1)))


def _soft_chroma(pit):
    return torch.einsum("cp,bpt->bct", _PC, pit)


def _recon_loss(out, tgt, w_pos=120.0):
    w = torch.where(tgt > 1e-3, torch.tensor(w_pos, device=tgt.device),
                    torch.tensor(1.0, device=tgt.device))
    return (w * (out - tgt) ** 2).mean()


def _cp_loss(out_pit, tgt_pit):
    return 1.0 - F.cosine_similarity(_soft_chroma(out_pit), _soft_chroma(tgt_pit), dim=1).mean()


class SiblingPairs(torch.utils.data.Dataset):
    def __init__(self, ds, split="train"):
        self.ds = ds
        by = defaultdict(list)
        for t in ds.split(split):
            by[t.style_tgt].append(t)
        self.by = by
        self.items = [t for v in by.values() if len(v) >= 2 for t in v]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ta = self.items[i]
        sib = [t for t in self.by[ta.style_tgt] if t.item_id != ta.item_id]
        tb = sib[np.random.randint(len(sib))]
        Xa = two_ch(load_roll_bundle(str(self.ds.root / ta.X_path)))
        Xb = two_ch(load_roll_bundle(str(self.ds.root / tb.X_path)))
        return (torch.from_numpy(chord_skeleton(Xa[0])[None]),
                torch.from_numpy(Xb), torch.from_numpy(Xa))


def train_model(ds, epochs=8, bs=32, lam_cp=0.5, lr=2e-4):
    enc, ref = StyleEncoder().to(DEVICE), Refiner().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ref.parameters()), lr=lr)
    dl = torch.utils.data.DataLoader(SiblingPairs(ds, "train"), batch_size=bs,
                                     shuffle=True, num_workers=0, drop_last=True)
    for ep in range(epochs):
        enc.train(); ref.train(); tot = 0.0
        for skel, xb, tgt in dl:
            skel, xb, tgt = skel.to(DEVICE), xb.to(DEVICE), tgt.to(DEVICE)
            out = ref(skel, enc(xb))
            loss = _recon_loss(out, tgt) + lam_cp * _cp_loss(out[:, 0], tgt[:, 0])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1}/{epochs}  loss={tot/len(dl):.4f}")
    enc.eval(); ref.eval()
    return enc, ref


# --------------------------------------------------------------------------- #
# Candidate generators
# --------------------------------------------------------------------------- #
def sampler_overlay(Xb, prof, n_onsets, off_chord):
    """Overlay chord-tone onsets on X; durations/velocities sampled from the
    profile; a fraction allowed off-chord. CP-safe except for the off-chord share."""
    cr = combined(Xb); pcs = bar_pcs(cr)
    tp = prof["time_pitch"]; tps = tp.sum()
    if tps == 0:
        return None
    tpf = tp.ravel() / tps
    od = prof["onset_duration"]; odf = od.ravel()/od.sum() if od.sum() > 0 else None
    ov = prof["onset_velocity"]; ovf = ov.ravel()/ov.sum() if ov.sum() > 0 else None

    add = np.zeros((NP, T), np.float32)
    draw = _RNG.choice(tpf.size, size=n_onsets*4, p=tpf)
    tb, pitch = np.unravel_index(draw, tp.shape)
    coin = _RNG.random(tb.size); placed = 0
    for k in range(tb.size):
        if placed >= n_onsets:
            break
        beat_bin, p = int(tb[k]), int(pitch[k])
        bar, beat = beat_bin // BB, beat_bin % BB
        if (p % 12 not in pcs[bar]) and (coin[k] >= off_chord):
            continue
        dur = (int(np.unravel_index(_RNG.choice(odf.size, p=odf), od.shape)[1]) + 1) if odf is not None else 2
        if ovf is not None:
            vb = int(np.unravel_index(_RNG.choice(ovf.size, p=ovf), ov.shape)[1])
            vel = float(np.clip((vb + 0.5) * 128 / VELOCITY_BINS / 127.0, 0.1, 1.0))
        else:
            vel = 0.6
        t0 = bar*CPB + beat*SPB
        add[p, t0:min(t0+dur, T)] = max(add[p, t0], vel)
        placed += 1
    return _attach(Xb, add)


def model_overlays(Xb, prof, enc, ref, Zb):
    """Two CP-aware overlay candidates from the trained generative model."""
    if enc is None or ref is None or Zb is None:
        return
    with torch.no_grad():
        skel = torch.from_numpy(chord_skeleton(two_ch(Xb)[0])[None, None]).to(DEVICE)
        out = ref(skel, enc(torch.from_numpy(two_ch(Zb)[None]).to(DEVICE)))[0].cpu().numpy()
    cr = combined(Xb); pcs = bar_pcs(cr)
    mask = np.zeros((NP, T), np.float32)
    for b in range(N_BARS):
        for pc in pcs[b]:
            mask[pc:NP:12, b*CPB:(b+1)*CPB] = 1.0
    masked = (out[0] * mask).astype(np.float32)
    yield _attach(Xb, masked), "model_masked"
    yield _attach(Xb, np.where(out[0] > 0.5, out[0], masked).astype(np.float32)), "model_offchord"


def _attach(Xb, add):
    pit = Xb["pitched"] if Xb["pitched"].size else np.zeros((0, NP, T), np.float32)
    tids = list(Xb.get("track_ids", []))
    return {"pitched": np.concatenate([pit, add[None]], 0).astype(np.float32),
            "drum": Xb["drum"].astype(np.float32),
            "track_ids": np.array(tids + [(max(tids)+1 if tids else 0)], np.int32)}


def optimize_item(Xb, prof, enc=None, ref=None, Zb=None):
    cr = combined(Xb)
    best, bsc, btag = Xb, score_item(cr, Xb, prof)["score"], "copyX"
    for (no, oc) in _GRID:
        N_SEEDS = 4
        for seed in range(N_SEEDS):
            cand = sampler_overlay(Xb, prof, no, oc)
            if cand is None:
                continue
            s = score_item(cr, cand, prof)["score"]
            if s > bsc:
                best, bsc, btag = cand, s, f"n{no}oc{oc}s{seed}"
    for cand, tag in model_overlays(Xb, prof, enc, ref, Zb):
        s = score_item(cr, cand, prof)["score"]
        if s > bsc:
            best, bsc, btag = cand, s, tag
    return best, bsc, btag


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def true_profile(ds, style):
    p = ds.root / "style_profiles" / f"{style}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def run_split(ds, split, enc, ref, out_csv=None, n=None,
              use_true_profiles=False, loo_exclude_self=True):
    by_style = defaultdict(list)
    for t in ds.split(split):
        by_style[t.style_tgt].append(t)
    base, opt, rows, ids, wins = [], [], [], [], []
    ph = {k: [] for k in HIST_NAMES}; cnt = 0
    for style, items in by_style.items():
        bnd = {o.item_id: load_roll_bundle(str(ds.root / o.X_path)) for o in items}
        shared = true_profile(ds, style) if use_true_profiles else None
        for t in items:
            Xb = bnd[t.item_id]
            if shared is not None:
                prof = shared
            else:
                pool = [bnd[o.item_id] for o in items
                        if loo_exclude_self is False or o.item_id != t.item_id]
                prof = build_style_profile_from_bundles(pool)
            Zb = load_roll_bundle(str(ds.root / t.Z_path)) if enc is not None else None
            base.append(score_item(combined(Xb), Xb, prof)["score"])
            cand, sc, tag = optimize_item(Xb, prof, enc, ref, Zb)
            opt.append(sc); wins.append(tag); ids.append(t.item_id)
            rows.append({"ID": t.item_id, "notes": encode_notes(bundle_to_rows(t.item_id, cand))})
            h = compute_histograms(cand)
            for k in HIST_NAMES:
                ph[k].append(_cosine(h[k], prof[k]))
            cnt += 1
            if n and cnt >= n:
                break
        if n and cnt >= n:
            break
    mw = sum(v for k, v in Counter(wins).items() if k.startswith("model"))
    print(f"[{split}] copy-X {np.mean(base):.4f} -> optimised {np.mean(opt):.4f}"
          f"  lift {np.mean(opt)-np.mean(base):+.4f}  (n={cnt})")
    print("  per-hist:", {k: round(np.mean(ph[k]), 3) for k in HIST_NAMES})
    if out_csv:
        pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS)).to_csv(out_csv, index=False)
        validate_submission(out_csv, required_item_ids=ids)
        print(f"  wrote {out_csv}")
    return float(np.mean(opt))


def main(root, out_csv="submission.csv", val_n=300):
    ds = HackathonDataset(root)
    print("1) Training one-shot generative model...")
    enc, ref = train_model(ds)
    print("\n2) Validation check:")
    run_split(ds, "val", enc, ref, n=val_n, use_true_profiles=True)
    print("\n3) Building submission on test (reconstructed profiles):")
    run_split(ds, "test", enc, ref, out_csv=out_csv, loo_exclude_self=False)
    print("\nDone.")


if __name__ == "__main__":
    import sys
    main("/kaggle/input/competitions/sapienza-genai-hackathon/dataset")