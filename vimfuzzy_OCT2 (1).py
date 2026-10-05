#!/usr/bin/env python3
# =============================================================================
# ViM-Fuzzy: Bidirectional Vision Mamba tissue segmentation + Mamdani fuzzy
# chronicity / healing assessment.  SINGLE FILE, follows the manuscript's
# Section 3 (Eqs. 1-23) step by step.
#
# HOW TO RUN (Colab):
#   1) Runtime > GPU.   2) Paste this file in one cell (or upload and
#      `!python vim_fuzzy_pipeline.py`).   3) Edit the CONFIG block only.
#
# Every number in the paper's tables must come from the CSV files written to
# OUT_DIR by this script.  Nothing is hard-coded.
# =============================================================================
import os, re, json, math, random, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  (the only block you should edit)
# =============================================================================
CFG = dict(
    DATASET_ROOT="/content/drive/MyDrive/Colab Notebooks/Wound/Woundds",
    MASK_DIR=None,            # None -> DATASET_ROOT/Mask   (files: <case>_tissue.png, labels 0..3)
    IMG_DIR=None,             # None -> auto-search DATASET_ROOT for <case>.(jpg|jpeg|png)
    OUT_DIR=None,             # None -> DATASET_ROOT/ViM_Fuzzy_Results
    PATIENT_CSV=None,         # optional: columns Case,Patient  (to split by patient)
    CLINICAL_CSV=None,        # optional: columns Case,Age  [,RiskScore 0-100]

    # ---- data / split (manuscript Sec. 4.1) ----
    IMG_SIZE=256, NUM_CLASSES=4, SPLIT=(0.70, 0.15, 0.15), SEED=42,
    # ---- model (Sec. 3.2) ----
    PATCH=16, EMBED_DIM=192, DEPTH=8, EXPAND=1, D_STATE=16,
    # ---- training ----
    EPOCHS=60, BATCH=8, LR=3e-4, WD=0.05, WARMUP=3, CLIP=1.0, AMP=True,
    LOSS_W_CE=0.5, LOSS_W_DICE=0.5, PATIENCE=15, NUM_WORKERS=2,
    # ---- evaluation ----
    NSD_TAU=2.0,              # px, at IMG_SIZE
    CI_THRESH=(25.0, 50.0, 75.0),   # a-priori CI cut-offs: Low/Moderate/High/Critical
    SKIP_TRAIN=False,         # True -> load OUT_DIR/best_vim.pth
    MAX_CASES=None,           # debug: use only N cases
)
CLASS_NAMES = ["Background", "Granulation", "Slough", "Necrosis"]
TISSUE = CLASS_NAMES[1:]
HEALING = ["Bruises", "Abrasions", "Laceration", "Cut"]          # etiology-defined labels
CHRONIC = ["Pressure_Wounds", "Venous_Wounds", "Diabetic_Wounds"]  # (Burns excluded)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def etiology(case):
    return re.sub(r"_\d+$", "", case)


# =============================================================================
# STEP 1  DATA DISCOVERY AND SPLIT
# =============================================================================
def discover_cases(cfg):
    root = Path(cfg["DATASET_ROOT"])
    mask_dir = Path(cfg["MASK_DIR"] or root / "Mask")
    masks = {p.name[:-len("_tissue.png")]: p for p in mask_dir.glob("*_tissue.png")}
    imgs = {}
    search = Path(cfg["IMG_DIR"]) if cfg["IMG_DIR"] else root
    skip = {"mask", "stagewise", cfg["OUT_DIR_NAME"].lower()}
    for dp, dn, fn in os.walk(search):
        dn[:] = [d for d in dn if d.lower() not in skip]
        for f in fn:
            st, ext = os.path.splitext(f)
            if ext.lower() in (".jpg", ".jpeg", ".png") and st in masks and st not in imgs:
                imgs[st] = Path(dp) / f
    cases = sorted(set(masks) & set(imgs))
    print(f"  masks found: {len(masks)} | images matched: {len(imgs)} | usable cases: {len(cases)}")
    if not cases:
        raise RuntimeError("No (image, mask) pairs found. Set CFG['IMG_DIR'] to the folder holding the photos "
                           "named exactly like the cases (e.g. Abrasions_019.jpg).")
    return cases, masks, imgs


def make_split(cases, cfg):
    groups = {c: c for c in cases}
    if cfg["PATIENT_CSV"]:
        pm = pd.read_csv(cfg["PATIENT_CSV"]).set_index("Case")["Patient"].to_dict()
        groups = {c: str(pm.get(c, c)) for c in cases}
    else:
        print("  NOTE: no PATIENT_CSV -> every case is treated as an independent patient (state this in the paper).")
    g2c = {}
    for c, g in groups.items():
        g2c.setdefault(g, []).append(c)
    gl = sorted(g2c); strat = [etiology(g2c[g][0]) for g in gl]
    tr, te = train_test_split(gl, test_size=cfg["SPLIT"][1] + cfg["SPLIT"][2], stratify=strat, random_state=cfg["SEED"])
    va, te = train_test_split(te, test_size=cfg["SPLIT"][2] / (cfg["SPLIT"][1] + cfg["SPLIT"][2]),
                              stratify=[etiology(g2c[g][0]) for g in te], random_state=cfg["SEED"])
    sp = {k: sorted(c for g in v for c in g2c[g]) for k, v in dict(train=tr, val=va, test=te).items()}
    print("  split sizes:", {k: len(v) for k, v in sp.items()})
    return sp


class WoundDS(Dataset):
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, cases, masks, imgs, size, train):
        self.cases, self.masks, self.imgs, self.size, self.train = cases, masks, imgs, size, train

    def __len__(self): return len(self.cases)

    def __getitem__(self, i):
        c = self.cases[i]
        im = cv2.cvtColor(cv2.imread(str(self.imgs[c])), cv2.COLOR_BGR2RGB)
        mk = cv2.imread(str(self.masks[c]), cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mk = cv2.resize(mk, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random() < 0.5: im, mk = im[:, ::-1], mk[:, ::-1]
            if random.random() < 0.5: im, mk = im[::-1], mk[::-1]
            im = np.clip(im.astype(np.float32) * random.uniform(0.85, 1.15) + random.uniform(-15, 15), 0, 255)
        im = (im.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        return dict(img=torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))),
                    mask=torch.from_numpy(np.ascontiguousarray(mk)).long(), case=c)


def check_mask_labels(cases, masks, k, n=30):
    vals = set()
    for c in cases[:n]:
        vals |= set(np.unique(cv2.imread(str(masks[c]), cv2.IMREAD_GRAYSCALE)).tolist())
    print("  label values seen in masks:", sorted(vals))
    if not vals <= set(range(k)):
        raise ValueError(f"Mask values {sorted(vals)} are not in 0..{k-1}. Re-map them before training.")


# =============================================================================
# STEP 2  MODEL  (Sec. 3.2: patch embedding, selective SSM, bidirectional scan, fusion)
# =============================================================================
class SelectiveSSM(nn.Module):
    """Eqs. 3-6: input-dependent (Delta_t, B_t, C_t), ZOH discretisation (Eq. 4), recurrence (Eq. 5)."""

    def __init__(self, d_inner, d_state=16):
        super().__init__()
        self.D_in, self.S = d_inner, d_state
        self.R = max(1, math.ceil(d_inner / 16))
        self.x_proj = nn.Linear(d_inner, self.R + 2 * d_state, bias=False)   # s_Delta (low rank), s_B, s_C
        self.dt_proj = nn.Linear(self.R, d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))
        self.Dskip = nn.Parameter(torch.ones(d_inner))
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))      # softplus^-1(dt)

    def forward(self, u):                                  # u: (B, N, D)
        u = u.float()
        Bsz, N, D = u.shape
        A = -torch.exp(self.A_log.float())                 # (D, S) diagonal state matrix
        dt, Bt, Ct = torch.split(self.x_proj(u), [self.R, self.S, self.S], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                  # (B, N, D)  Delta_t > 0
        Abar = torch.exp(dt.unsqueeze(-1) * A)             # (B, N, D, S)   exp(Delta A)
        Bbar = (Abar - 1.0) / A * Bt.unsqueeze(2)          # (Delta A)^-1 (exp(Delta A) - I) Delta B  (diagonal A)
        BU = Bbar * u.unsqueeze(-1)
        h = u.new_zeros(Bsz, D, self.S)
        ys = []
        for t in range(N):                                 # sequential scan (linear in N)
            h = Abar[:, t] * h + BU[:, t]
            ys.append((h * Ct[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, 1) + u * self.Dskip


class BiSSMBlock(nn.Module):
    """Forward scan (Eq. 7), backward scan (Eq. 8), fusion y = phi(Wf yf + Wb yb + b) (Eq. 9),
    followed by Mamba-style z-gating and a residual connection."""

    def __init__(self, d, expand=1, d_state=16):
        super().__init__()
        inner = d * expand
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * inner)
        self.conv_f = nn.Conv1d(inner, inner, 3, padding=1, groups=inner)
        self.conv_b = nn.Conv1d(inner, inner, 3, padding=1, groups=inner)
        self.ssm_f, self.ssm_b = SelectiveSSM(inner, d_state), SelectiveSSM(inner, d_state)
        self.Wf, self.Wb = nn.Linear(inner, inner, bias=False), nn.Linear(inner, inner, bias=True)
        self.out_proj = nn.Linear(inner, d)

    def forward(self, x):
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        uf = F.silu(self.conv_f(u.transpose(1, 2)).transpose(1, 2))
        ub = F.silu(self.conv_b(u.flip(1).transpose(1, 2)).transpose(1, 2))
        yf = self.ssm_f(uf)                                # t = 1 -> N
        yb = self.ssm_b(ub).flip(1)                        # t = N -> 1
        y = F.gelu(self.Wf(yf) + self.Wb(yb))              # Eq. 9
        return x + self.out_proj(y * F.silu(z))


class VisionMambaSeg(nn.Module):
    def __init__(self, k=4, patch=16, d=192, depth=8, expand=1, d_state=16, img=256):
        super().__init__()
        self.patch, self.d, self.g = patch, d, img // patch
        self.embed = nn.Conv2d(3, d, patch, patch)         # == flatten patch + linear E  (Eqs. 1-2)
        self.pos = nn.Parameter(torch.zeros(1, self.g * self.g, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([BiSSMBlock(d, expand, d_state) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        n_up = int(math.log2(patch)); ch = [d] + [max(16, 128 // 2 ** i) for i in range(n_up)]
        L = []
        for i in range(n_up):                              # progressive decoder (Eq. 10, repeated)
            L += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                  nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.GroupNorm(8, ch[i + 1]), nn.GELU()]
        self.decoder = nn.Sequential(*L); self.head = nn.Conv2d(ch[-1], k, 1)

    def forward(self, x):
        t = self.embed(x).flatten(2).transpose(1, 2) + self.pos          # (B, N, d)
        for b in self.blocks: t = b(t)
        t = self.norm(t)
        wfv = t.mean(1)                                                  # deep visual embedding WFV (B, d)
        fm = t.transpose(1, 2).reshape(x.size(0), self.d, self.g, self.g)  # tokens -> 2-D map
        return self.head(self.decoder(fm)), wfv


# =============================================================================
# STEP 3  LOSS (Eqs. 13-15) AND TRAINING
# =============================================================================
def seg_loss(logits, y, k, wce, wdice, eps=1e-6):
    ce = F.cross_entropy(logits, y)
    p = F.softmax(logits, 1); oh = F.one_hot(y, k).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); card = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = 1 - ((2 * inter + eps) / (card + eps)).mean()             # per-class Dice, averaged
    return wce * ce + wdice * dice


@torch.no_grad()
def val_foreground_dice(model, loader, k):
    model.eval(); inter = np.zeros(k); card = np.zeros(k)
    for b in loader:
        pr = model(b["img"].to(DEVICE))[0].argmax(1).cpu().numpy(); gt = b["mask"].numpy()
        for c in range(k):
            inter[c] += ((pr == c) & (gt == c)).sum(); card[c] += (pr == c).sum() + (gt == c).sum()
    d = 2 * inter / np.maximum(card, 1)
    return float(d[1:].mean())


def train(model, dl_tr, dl_va, cfg, out):
    k = cfg["NUM_CLASSES"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    total = cfg["EPOCHS"] * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (cfg["WARMUP"] * len(dl_tr))) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["AMP"] and DEVICE.type == "cuda")
    hist, best, bad = [], -1, 0
    for ep in range(1, cfg["EPOCHS"] + 1):
        model.train(); tl = 0
        for b in dl_tr:
            x, y = b["img"].to(DEVICE), b["mask"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, enabled=cfg["AMP"] and DEVICE.type == "cuda"):
                lg, _ = model(x)
            loss = seg_loss(lg.float(), y, k, cfg["LOSS_W_CE"], cfg["LOSS_W_DICE"])
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["CLIP"])
            scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        model.eval(); vl = 0
        with torch.no_grad():
            for b in dl_va:
                lg, _ = model(b["img"].to(DEVICE))
                vl += seg_loss(lg.float(), b["mask"].to(DEVICE), k, cfg["LOSS_W_CE"], cfg["LOSS_W_DICE"]).item()
        vd = val_foreground_dice(model, dl_va, k)
        hist.append(dict(epoch=ep, train_loss=tl / len(dl_tr), val_loss=vl / len(dl_va), val_fg_dice=vd))
        print(f"  ep {ep:03d} | train {hist[-1]['train_loss']:.4f} | val {hist[-1]['val_loss']:.4f} | val fgDice {vd:.4f}")
        if vd > best:
            best, bad = vd, 0; torch.save(model.state_dict(), out / "best_vim.pth")
        else:
            bad += 1
            if bad >= cfg["PATIENCE"]: print("  early stopping"); break
    h = pd.DataFrame(hist); h.to_csv(out / "training_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_fg_dice); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation foreground Dice")
    plt.tight_layout(); plt.savefig(out / "fig_training_curves.png", dpi=200); plt.close()
    return best


# =============================================================================
# STEP 4  SEGMENTATION METRICS (Table 1 of the manuscript)
# =============================================================================
# --- BEGIN PURE-NUMPY SECTION (metrics, WTP, fuzzy) ---
def dice_iou(gt, pr, k):
    d, i = [], []
    for c in range(k):
        g, p = gt == c, pr == c
        if not g.any() and not p.any(): d.append(np.nan); i.append(np.nan); continue
        inter = (g & p).sum(); d.append(2 * inter / (g.sum() + p.sum())); i.append(inter / (g | p).sum())
    return d, i


def _surface(m): return m & ~ndi.binary_erosion(m, border_value=0)


def hd95_nsd(g, p, tau):
    if not g.any() or not p.any(): return np.nan, np.nan
    sg, sp = _surface(g), _surface(p)
    dg, dp = ndi.distance_transform_edt(~sg), ndi.distance_transform_edt(~sp)
    a, b = dg[sp], dp[sg]
    return max(np.percentile(a, 95), np.percentile(b, 95)), ((a <= tau).sum() + (b <= tau).sum()) / (len(a) + len(b))


def pooled_prf(cm):
    tot = cm.sum(); rows = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = tot - tp - fn - fp
        rows.append(dict(Precision=tp / max(tp + fp, 1), Recall=tp / max(tp + fn, 1), Specificity=tn / max(tn + fp, 1)))
    return rows


def wtp(mask, k=4):
    """Image-level P_c (Eq. 17a, includes background) and wound-level R_c (Eq. 17b, tissue only)."""
    cnt = np.array([(mask == c).sum() for c in range(k)], float)
    p = 100 * cnt / mask.size
    aw = cnt[1:].sum(); r = 100 * cnt[1:] / aw if aw > 0 else np.full(k - 1, np.nan)
    return p, r


def agreement(gt, pr):
    gt, pr = np.asarray(gt, float), np.asarray(pr, float); ok = ~(np.isnan(gt) | np.isnan(pr)); gt, pr = gt[ok], pr[ok]
    e = pr - gt; sd = e.std(ddof=1)
    r = pearsonr(gt, pr)[0] if gt.std() > 0 and pr.std() > 0 else np.nan
    ccc = 2 * np.cov(gt, pr, ddof=0)[0, 1] / (gt.var() + pr.var() + (gt.mean() - pr.mean()) ** 2)
    return dict(n=int(ok.sum()), GT_mean=gt.mean(), Pred_mean=pr.mean(), Bias=e.mean(), MAE=np.abs(e).mean(),
                RMSE=np.sqrt((e ** 2).mean()), Pearson_r=r, Lin_CCC=ccc, LoA_low=e.mean() - 1.96 * sd, LoA_high=e.mean() + 1.96 * sd)


# =============================================================================
# STEP 6  MAMDANI FUZZY INFERENCE (Sec. 3.4, Eqs. 18-23)
# =============================================================================
def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    l = np.ones_like(x) if b == a else np.clip((x - a) / (b - a), 0, 1)
    r = np.ones_like(x) if d == c else np.clip((d - x) / (d - c), 0, 1)
    return np.minimum(l, r)


def trimf(x, a, b, c): return trapmf(x, a, b, b, c)


# !!! ASSUMED membership parameters: they are NOT in the manuscript.  Replace with your own,
# !!! justify them (literature / clinicians) and report them in a table + figure.
MF_IN = {
    "nec": {"L": ("trap", 0, 0, 10, 30), "M": ("tri", 15, 35, 55), "H": ("trap", 40, 65, 100, 100)},
    "slo": {"L": ("trap", 0, 0, 10, 30), "M": ("tri", 15, 35, 55), "H": ("trap", 40, 65, 100, 100)},
    "gra": {"P": ("trap", 0, 0, 15, 35), "M": ("tri", 20, 45, 70), "O": ("trap", 55, 75, 100, 100)},
    "ctx": {"LR": ("trap", 0, 0, 25, 45), "MR": ("tri", 30, 50, 70), "HR": ("trap", 55, 75, 100, 100)},
}
MF_OUT = {
    "risk": {"L": ("trap", 0, 0, 15, 35), "M": ("tri", 25, 45, 65), "H": ("tri", 50, 70, 85), "C": ("trap", 75, 90, 100, 100)},
    "heal": {"S": ("trap", 0, 0, 15, 35), "D": ("tri", 25, 50, 75), "F": ("trap", 65, 85, 100, 100)},
}
UNI = np.linspace(0, 100, 501)
# (id, R_nec, R_slough, R_gran, C, risk, healing) -- Table 2 of the manuscript
RULES = [("R01", "H", "H", "P", "HR", "C", "S"), ("R02", "H", "M", "P", "MR", "H", "S"),
         ("R03", "H", "L", "P", "LR", "H", "S"), ("R04", "M", "H", "P", "HR", "C", "S"),
         ("R05", "M", "H", "M", "MR", "H", "D"), ("R06", "M", "M", "M", "MR", "M", "D"),
         ("R07", "M", "L", "M", "LR", "M", "D"), ("R08", "L", "H", "P", "HR", "H", "S"),
         ("R09", "L", "H", "M", "MR", "M", "D"), ("R10", "L", "M", "M", "LR", "M", "D"),
         ("R11", "L", "M", "O", "MR", "L", "F"), ("R12", "L", "L", "P", "HR", "M", "D"),
         ("R13", "L", "L", "M", "LR", "L", "F"), ("R14", "L", "L", "O", "LR", "L", "F"),
         ("R15", "L", "L", "O", "HR", "L", "F")]
RULESETS = {4: ["R04", "R08", "R11", "R15"],
            8: ["R01", "R04", "R05", "R07", "R08", "R11", "R13", "R15"],
            12: ["R01", "R02", "R04", "R05", "R06", "R07", "R08", "R09", "R11", "R13", "R14", "R15"],
            15: [r[0] for r in RULES]}
CATS = ["Low", "Moderate", "High", "Critical"]; RISK_LABEL = dict(L="Low", M="Moderate", H="High", C="Critical")


def mf(spec, x):
    return trapmf(x, *spec[1:]) if spec[0] == "trap" else trimf(x, *spec[1:])


def fuzzy_infer(nec, slo, gra, ctx, rule_ids):
    """Returns CI, HS (centroid, Eq. 23), firing strengths (min t-norm, Eq. 20); NaN CI/HS if no rule fires."""
    mu = {"nec": {s: float(mf(v, nec)) for s, v in MF_IN["nec"].items()},
          "slo": {s: float(mf(v, slo)) for s, v in MF_IN["slo"].items()},
          "gra": {s: float(mf(v, gra)) for s, v in MF_IN["gra"].items()},
          "ctx": {s: float(mf(v, ctx)) for s, v in MF_IN["ctx"].items()}}
    agg_r, agg_h, alphas = np.zeros_like(UNI), np.zeros_like(UNI), {}
    for rid, n, s, g, c, rk, hl in RULES:
        if rid not in rule_ids: continue
        a = min(mu["nec"][n], mu["slo"][s], mu["gra"][g], mu["ctx"][c]); alphas[rid] = a
        if a > 0:
            agg_r = np.maximum(agg_r, np.minimum(a, mf(MF_OUT["risk"][rk], UNI)))   # Eq. 21-22
            agg_h = np.maximum(agg_h, np.minimum(a, mf(MF_OUT["heal"][hl], UNI)))
    if agg_r.sum() == 0: return np.nan, np.nan, alphas
    return float((UNI * agg_r).sum() / agg_r.sum()), float((UNI * agg_h).sum() / agg_h.sum()), alphas


def ci_category(ci, th):
    if np.isnan(ci): return "Fallback"
    return CATS[int(np.searchsorted(th, ci, side="right"))]


def context_score(case, clin):
    """ASSUMPTION: clinical risk context C in [0,100]. Uses RiskScore, else age (30 y -> 0, 80 y -> 100), else 50."""
    if clin is not None and case in clin.index:
        row = clin.loc[case]
        if "RiskScore" in row and not pd.isna(row["RiskScore"]): return float(row["RiskScore"])
        if "Age" in row and not pd.isna(row["Age"]): return float(np.clip((row["Age"] - 30) / 50 * 100, 0, 100))
    return 50.0


def run_fuzzy(R, ctx, rule_ids):
    out = [fuzzy_infer(r[2], r[1], r[0], c, rule_ids) for r, c in zip(R, ctx)]   # R columns: gran, slough, nec
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]), [o[2] for o in out])
# --- END PURE-NUMPY SECTION ---


def evaluate_segmentation(model, ds_test, cfg, out):
    k, tau = cfg["NUM_CLASSES"], cfg["NSD_TAU"]
    dl = DataLoader(ds_test, batch_size=1, shuffle=False)
    cm = np.zeros((k, k), np.int64); rows, wrows = [], []
    pred_dir = out / "pred_masks"; pred_dir.mkdir(exist_ok=True); (out / "embeddings").mkdir(exist_ok=True)
    model.eval(); t_inf = []
    with torch.no_grad():
        for b in dl:
            case = b["case"][0]; x = b["img"].to(DEVICE)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.time(); lg, wfv = model(x)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t_inf.append(time.time() - t0)
            pr = lg.argmax(1)[0].cpu().numpy().astype(np.uint8); gt = b["mask"][0].numpy().astype(np.uint8)
            cv2.imwrite(str(pred_dir / f"{case}_pred.png"), pr); np.save(out / "embeddings" / f"{case}_wfv.npy", wfv[0].cpu().numpy())
            cm += np.bincount((gt.astype(int) * k + pr).ravel(), minlength=k * k).reshape(k, k)
            d, i = dice_iou(gt, pr, k); rec = dict(Case=case, Etiology=etiology(case))
            for c, n in enumerate(CLASS_NAMES):
                h, s = hd95_nsd(gt == c, pr == c, tau)
                rec.update({f"Dice_{n}": d[c], f"IoU_{n}": i[c], f"HD95_{n}": h, f"NSD_{n}": s})
            for key in ("Dice", "IoU", "HD95", "NSD"):
                rec[f"{key}_Foreground"] = np.nanmean([rec[f"{key}_{n}"] for n in TISSUE]) if not all(np.isnan(rec[f"{key}_{n}"]) for n in TISSUE) else np.nan
            rows.append(rec)
            pg, rg = wtp(gt, k); pp, rp = wtp(pr, k)
            w = dict(Case=case, Etiology=etiology(case))
            w.update({f"P_GT_{n}": pg[c] for c, n in enumerate(CLASS_NAMES)}); w.update({f"P_Pred_{n}": pp[c] for c, n in enumerate(CLASS_NAMES)})
            w.update({f"R_GT_{n}": rg[c] for c, n in enumerate(TISSUE)}); w.update({f"R_Pred_{n}": rp[c] for c, n in enumerate(TISSUE)})
            wrows.append(w)
    df, dw = pd.DataFrame(rows), pd.DataFrame(wrows)
    df.to_csv(out / "per_image_metrics.csv", index=False); dw.to_csv(out / "per_image_wtp.csv", index=False)
    np.savetxt(out / "confusion_matrix_counts.csv", cm, fmt="%d", delimiter=",")
    prf = pooled_prf(cm); T3 = []
    for c, n in enumerate(CLASS_NAMES):
        r = dict(Class=n)
        for key in ("Dice", "IoU", "HD95", "NSD"):
            v = df[f"{key}_{n}"].dropna(); r[key] = f"{v.mean():.3f} ± {v.std():.3f}"; r[f"n_{key}"] = len(v)
        r.update({m: round(prf[c][m], 3) for m in ("Precision", "Recall", "Specificity")}); T3.append(r)
    fg = dict(Class="Foreground (mean of 3 tissues)")
    for key in ("Dice", "IoU", "HD95", "NSD"):
        v = df[f"{key}_Foreground"].dropna(); fg[key] = f"{v.mean():.3f} ± {v.std():.3f}"; fg[f"n_{key}"] = len(v)
    T3.append(fg); T3 = pd.DataFrame(T3); T3.to_csv(out / "Table_segmentation.csv", index=False)
    cmn = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for a in range(k):
        for b_ in range(k): ax.text(b_, a, f"{cmn[a, b_]:.2f}", ha="center", va="center", color="white" if cmn[a, b_] > .5 else "black")
    ax.set_xticks(range(k)); ax.set_yticks(range(k)); ax.set_xticklabels(CLASS_NAMES, rotation=30); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.set_title("Row-normalised confusion matrix (pooled)")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(out / "fig_confusion_matrix.png", dpi=200); plt.close()
    fig, ax = plt.subplots(figsize=(7, 3.8)); ax.boxplot([df[f"Dice_{n}"].dropna() for n in CLASS_NAMES], labels=CLASS_NAMES)
    ax.set_ylabel("Dice (per image)"); plt.tight_layout(); plt.savefig(out / "fig_dice_boxplot.png", dpi=200); plt.close()
    info = dict(ms_per_image=1000 * float(np.median(t_inf[1:] or t_inf)), n_test=len(df))
    return df, dw, cm, T3, info


def wtp_tables(dw, out):
    rows = []
    for n in CLASS_NAMES:
        rows.append(dict(Level="image-level (incl. background)", Class=n, **agreement(dw[f"P_GT_{n}"], dw[f"P_Pred_{n}"])))
    for n in TISSUE:
        rows.append(dict(Level="wound-level (tissue only)", Class=n, **agreement(dw[f"R_GT_{n}"], dw[f"R_Pred_{n}"])))
    T = pd.DataFrame(rows).round(3); T.to_csv(out / "Table_WTP_agreement.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, n in zip(ax, TISSUE):
        g, p = dw[f"R_GT_{n}"], dw[f"R_Pred_{n}"]; m = (g + p) / 2; e = p - g
        a.scatter(m, e, s=8, alpha=.5); a.axhline(e.mean(), color="r"); a.axhline(e.mean() + 1.96 * e.std(), ls="--", color="gray"); a.axhline(e.mean() - 1.96 * e.std(), ls="--", color="gray")
        a.set_title(f"Bland-Altman: {n}"); a.set_xlabel("mean of GT and prediction (%)"); a.set_ylabel("Pred - GT (pp)")
    plt.tight_layout(); plt.savefig(out / "fig_bland_altman.png", dpi=200); plt.close()
    g = dw.groupby("Etiology"); E = []
    for et, d in g:
        r = dict(Etiology=et, n=len(d))
        for n in TISSUE: r[f"Pred_{n}_%image"] = f"{d[f'P_Pred_{n}'].mean():.2f} ± {d[f'P_Pred_{n}'].std():.2f}"
        for n in TISSUE: r[f"Pred_{n}_%wound"] = f"{d[f'R_Pred_{n}'].mean():.2f} ± {d[f'R_Pred_{n}'].std():.2f}"
        r["Empty_wound_pred_n"] = int(d["R_Pred_Granulation"].isna().sum()); E.append(r)
    pd.DataFrame(E).to_csv(out / "Table_WTP_by_etiology.csv", index=False)
    return T


def plot_mfs(out):
    fig, ax = plt.subplots(2, 3, figsize=(13, 6)); ax = ax.ravel(); i = 0
    for grp, nm in ((MF_IN, "Input"), (MF_OUT, "Output")):
        for var, sets in grp.items():
            for s, spec in sets.items(): ax[i].plot(UNI, mf(spec, UNI), label=s)
            ax[i].set_title(f"{nm}: {var}"); ax[i].legend(); i += 1
    plt.tight_layout(); plt.savefig(out / "fig_membership_functions.png", dpi=200); plt.close()
    rows = [dict(Variable=v, Set=s, Type=sp[0], Parameters=sp[1:]) for g in (MF_IN, MF_OUT) for v, st in g.items() for s, sp in st.items()]
    pd.DataFrame(rows).to_csv(out / "Table_membership_parameters.csv", index=False)


def evaluate_fuzzy(dw, cfg, out):
    clin = pd.read_csv(cfg["CLINICAL_CSV"]).set_index("Case") if cfg["CLINICAL_CSV"] else None
    if clin is None: print("  NOTE: no CLINICAL_CSV -> context C = 50 (Moderate) for every case; fuzzy results are tissue-driven only.")
    ctx = np.array([context_score(c, clin) for c in dw.Case]); th = cfg["CI_THRESH"]
    Rp = dw[[f"R_Pred_{n}" for n in TISSUE]].values; Rg = dw[[f"R_GT_{n}" for n in TISSUE]].values
    valid = ~(np.isnan(Rp).any(1) | np.isnan(Rg).any(1))
    print(f"  cases with an empty predicted or GT wound region (excluded from fuzzy stage): {(~valid).sum()}")
    res = {}; abl = []; dom_tab = []
    for size, ids in RULESETS.items():
        ci, hs, al = run_fuzzy(Rp[valid], ctx[valid], ids); cat = np.array([ci_category(c, th) for c in ci])
        res[size] = (ci, hs, al, cat)
        row = dict(RuleSet=f"{size}-rule", n=int(valid.sum()), CI=f"{np.nanmean(ci):.2f} ± {np.nanstd(ci, ddof=1):.2f}", HS=f"{np.nanmean(hs):.2f}",
                   Fallback_n=int((cat == "Fallback").sum()), Fallback_pct=round(100 * (cat == "Fallback").mean(), 1))
        row.update({c: int((cat == c).sum()) for c in CATS}); abl.append(row)
        for rid in ids:
            fired = np.array([a.get(rid, 0) > 0 for a in al])
            dom = np.array([max(a, key=a.get) == rid if a and max(a.values()) > 0 else False for a in al])
            intended = RISK_LABEL[[r for r in RULES if r[0] == rid][0][5]]
            dom_tab.append(dict(RuleSet=f"{size}-rule", Rule=rid, Fired_pct=round(100 * fired.mean(), 1), Dominant_n=int(dom.sum()),
                                Dominant_pct=round(100 * dom.mean(), 1),
                                Purity_pct=round(100 * (cat[dom] == intended).mean(), 1) if dom.sum() else np.nan))
    pd.DataFrame(abl).to_csv(out / "Table_rule_ablation.csv", index=False)
    pd.DataFrame(dom_tab).to_csv(out / "Table_rule_firing_dominance.csv", index=False)
    ci, hs, al, cat = res[15]
    # error propagation: fuzzy output from GT tissue vs predicted tissue
    cg, hg, _ = run_fuzzy(Rg[valid], ctx[valid], RULESETS[15]); catg = np.array([ci_category(c, th) for c in cg])
    ok = ~(np.isnan(ci) | np.isnan(cg))
    prop = dict(n=int(ok.sum()), CI_MAE=float(np.abs(ci[ok] - cg[ok]).mean()), HS_MAE=float(np.abs(hs[ok] - hg[ok]).mean()),
                CI_pearson_r=float(pearsonr(ci[ok], cg[ok])[0]), CategoryAgreement_pct=float(100 * (cat[ok] == catg[ok]).mean()))
    pd.DataFrame([prop]).round(3).to_csv(out / "Table_error_propagation.csv", index=False)
    d = dw[valid].copy(); d["CI"], d["HS"], d["Category"] = ci, hs, cat
    d["DominantRule"] = [max(a, key=a.get) if a and max(a.values()) > 0 else "none" for a in al]
    d["CI_fromGT"], d["HS_fromGT"] = cg, hg
    d.to_csv(out / "per_image_fuzzy.csv", index=False)
    E = d.groupby("Etiology").agg(n=("CI", "size"), CI_mean=("CI", "mean"), CI_sd=("CI", "std"), HS_mean=("HS", "mean"), HS_sd=("HS", "std")).round(2)
    E.to_csv(out / "Table_fuzzy_by_etiology.csv")
    # etiology-defined healing vs chronic (threshold-free AUC, bootstrap CI)
    sub = d[d.Etiology.isin(HEALING + CHRONIC)].copy(); y = sub.Etiology.isin(CHRONIC).astype(int).values   # 1 = chronic
    feats = {"Granulation % (lower -> chronic)": -sub["R_Pred_Granulation"], "Slough %": sub["R_Pred_Slough"],
             "Necrosis %": sub["R_Pred_Necrosis"], "Fuzzy CI": sub["CI"], "Fuzzy HS (lower -> chronic)": -sub["HS"]}
    rng = np.random.default_rng(cfg["SEED"]); A = []
    for nme, v in feats.items():
        v = v.values; m = ~np.isnan(v); a = roc_auc_score(y[m], v[m]); bs = []
        for _ in range(1000):
            ix = rng.integers(0, m.sum(), m.sum()); yy = y[m][ix]
            if yy.min() != yy.max(): bs.append(roc_auc_score(yy, v[m][ix]))
        A.append(dict(Metric=nme, AUC=round(a, 3), CI95=f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]", n_healing=int((y[m] == 0).sum()), n_chronic=int((y[m] == 1).sum())))
    pd.DataFrame(A).to_csv(out / "Table_healing_vs_chronic_AUC.csv", index=False)
    return pd.DataFrame(abl), prop, pd.DataFrame(A), E


def qualitative(model, ds, out, n=6):
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int); pal = np.array([[0, 0, 0], [220, 40, 40], [240, 200, 40], [40, 40, 40]], np.uint8)
    pal[3] = [60, 160, 60]; fig, ax = plt.subplots(3, len(idx), figsize=(2.6 * len(idx), 7.5), squeeze=False); model.eval()
    for j, i in enumerate(idx):
        b = ds[i]
        with torch.no_grad(): pr = model(b["img"][None].to(DEVICE))[0].argmax(1)[0].cpu().numpy()
        im = (b["img"].numpy().transpose(1, 2, 0) * WoundDS.STD + WoundDS.MEAN).clip(0, 1)
        ax[0, j].imshow(im); ax[0, j].set_title(b["case"], fontsize=8); ax[1, j].imshow(pal[b["mask"].numpy()]); ax[2, j].imshow(pal[pr])
        for r in range(3): ax[r, j].axis("off")
    ax[0, 0].text(-0.05, .5, "image", transform=ax[0, 0].transAxes, rotation=90, ha="right"); plt.tight_layout()
    plt.savefig(out / "fig_qualitative.png", dpi=200); plt.close()


# =============================================================================
# MAIN: runs every step in order
# =============================================================================
def main(cfg=CFG):
    cfg = dict(cfg); root = Path(cfg["DATASET_ROOT"]); cfg["OUT_DIR_NAME"] = Path(cfg["OUT_DIR"]).name if cfg["OUT_DIR"] else "ViM_Fuzzy_Results"
    out = Path(cfg["OUT_DIR"] or root / "ViM_Fuzzy_Results"); out.mkdir(parents=True, exist_ok=True); seed_all(cfg["SEED"])
    print("STEP 1  data discovery and stratified split"); cases, masks, imgs = discover_cases(cfg)
    if cfg["MAX_CASES"]: cases = cases[:cfg["MAX_CASES"]]
    check_mask_labels(cases, masks, cfg["NUM_CLASSES"]); sp = make_split(cases, cfg)
    pd.DataFrame([(c, s) for s, v in sp.items() for c in v], columns=["Case", "Split"]).to_csv(out / "split.csv", index=False)
    print(pd.DataFrame([(etiology(c), s) for s, v in sp.items() for c in v], columns=["E", "S"]).value_counts().unstack(fill_value=0))
    ds = {s: WoundDS(sp[s], masks, imgs, cfg["IMG_SIZE"], s == "train") for s in sp}
    dl = lambda s, sh: DataLoader(ds[s], batch_size=cfg["BATCH"], shuffle=sh, num_workers=cfg["NUM_WORKERS"], drop_last=(s == "train" and len(ds[s]) > cfg["BATCH"]))
    print("STEP 2  build model"); model = VisionMambaSeg(cfg["NUM_CLASSES"], cfg["PATCH"], cfg["EMBED_DIM"], cfg["DEPTH"], cfg["EXPAND"], cfg["D_STATE"], cfg["IMG_SIZE"]).to(DEVICE)
    npar = sum(p.numel() for p in model.parameters()); N = (cfg["IMG_SIZE"] // cfg["PATCH"]) ** 2
    print(f"  tokens N={N} | d={cfg['EMBED_DIM']} | depth={cfg['DEPTH']} | parameters={npar/1e6:.2f} M | device={DEVICE}")
    ck = out / "best_vim.pth"
    if cfg["SKIP_TRAIN"] and ck.exists(): print("STEP 3  training skipped, loading checkpoint")
    else: print("STEP 3  training"); print(f"  best validation foreground Dice: {train(model, dl('train', True), dl('val', False), cfg, out):.4f}")
    model.load_state_dict(torch.load(ck, map_location=DEVICE))
    print("STEP 4  segmentation metrics on the held-out test set"); df, dw, cm, T3, info = evaluate_segmentation(model, ds["test"], cfg, out)
    print(T3.to_string(index=False)); print(f"  inference: {info['ms_per_image']:.1f} ms/image (median, batch 1, {DEVICE})")
    print("STEP 5  WTP agreement (Eq. 16-17) and WFV export"); T4 = wtp_tables(dw, out); print(T4[["Level", "Class", "n", "GT_mean", "Pred_mean", "Bias", "MAE", "RMSE", "Pearson_r", "Lin_CCC"]].to_string(index=False))
    print("STEP 6  Mamdani fuzzy inference, rule ablation, error propagation"); plot_mfs(out)
    abl, prop, auc, E = evaluate_fuzzy(dw, cfg, out); print(abl.to_string(index=False)); print("  error propagation:", prop); print(auc.to_string(index=False))
    qualitative(model, ds["test"], out)
    json.dump(dict(config={k: v for k, v in cfg.items()}, parameters_M=npar / 1e6, tokens=N, **info), open(out / "run_info.json", "w"), indent=2, default=str)
    print(f"\nDONE. All tables/figures are in: {out}")


if __name__ == "__main__":
    main()
