import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from src.data import load_cache, normalize, make_masks, apply_mask
from src.models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["s2only", "concat", "ma"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--max_rate", type=float, default=0.5)    # training missing-rate ceiling
ap.add_argument("--moddrop", type=float, default=0.1)     # prob. a whole sensor is dropped
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--cache", default="data/processed/cache_seed42_cell1.npz")
ap.add_argument("--out", default="runs")
ap.add_argument("--val_mask_seed", type=int, default=1234)  # SET TO THE SEED USED FOR YOUR LR VAL MASKS
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)

z = load_cache(args.cache)
Xn = normalize(z["X"], z["mean"], z["std"])
y, split = z["y"], z["split"]
tr, va = split == "train", split == "val"        # the test split is never touched here
Xtr = torch.from_numpy(Xn[tr]).to(dev); ytr = torch.from_numpy(y[tr]).to(dev)
Xva_n, yva = Xn[va], y[va]

counts = np.bincount(y[tr], minlength=10)
w = torch.tensor(len(ytr) / (10 * counts), dtype=torch.float32, device=dev)
crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

model = build_model(args.model).to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

def apply_mask_t(x, m):
    x = x.clone()
    x[:, :, 0:2] *= m[:, :, 0:1]
    x[:, :, 2:13] *= m[:, :, 1:2]
    return x

@torch.no_grad()
def predict(Xnp, M):
    model.eval()
    Xm = torch.from_numpy(apply_mask(Xnp, M)).to(dev)
    Mt = torch.from_numpy(M).to(dev)
    out = [model(Xm[i:i + 1024], Mt[i:i + 1024]).argmax(1).cpu() for i in range(0, len(Xm), 1024)]
    return torch.cat(out).numpy()

def metrics(yt, yp):
    return {"acc": accuracy_score(yt, yp), "bal_acc": balanced_accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro")}

sel_masks = make_masks(len(yva), "random", 0.3, args.val_mask_seed)   # checkpoint-selection condition
best, best_ep, best_state, bad = -1.0, -1, None, 0
for ep in range(args.epochs):
    model.train()
    g = torch.Generator(device=dev).manual_seed(args.seed * 1000 + ep)
    perm = torch.randperm(len(ytr), device=dev, generator=g)
    for i in range(0, len(perm), args.batch):
        idx = perm[i:i + args.batch]
        xb, yb = Xtr[idx], ytr[idx]
        B = xb.shape[0]
        rate = torch.rand(B, 1, 1, device=dev, generator=g) * args.max_rate
        mb = torch.rand(B, 12, 2, device=dev, generator=g) >= rate
        d1 = torch.rand(B, device=dev, generator=g) < args.moddrop
        d2 = (torch.rand(B, device=dev, generator=g) < args.moddrop) & ~d1
        mb[d1, :, 0] = False; mb[d2, :, 1] = False
        loss = crit(model(apply_mask_t(xb, mb), mb), yb)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()
    f1 = metrics(yva, predict(Xva_n, sel_masks))["macro_f1"]
    print(f"ep {ep:02d} loss {loss.item():.3f} val macro-F1 {f1:.4f}")
    if f1 > best:
        best, best_ep, bad = f1, ep, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= args.patience:
            break

model.load_state_dict(best_state)
conds = [("random", 0.0), ("random", 0.3), ("random", 0.5), ("random", 0.7),
         ("block", 0.5), ("no_s1", 0.0), ("no_s2", 0.0)]
grid = {}
for kind, rate in conds:
    M = make_masks(len(yva), kind, rate, args.val_mask_seed)
    grid[f"{kind}_{rate}"] = metrics(yva, predict(Xva_n, M))
    print(kind, rate, {k: round(v, 4) for k, v in grid[f"{kind}_{rate}"].items()})

os.makedirs(args.out, exist_ok=True)
tag = f"{args.model}_seed{args.seed}"
res = {"args": vars(args), "params": n_params, "best_epoch": best_ep, "device": dev,
       "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
       "torch": torch.__version__, "val": grid}
json.dump(res, open(f"{args.out}/{tag}.json", "w"), indent=2)
torch.save(best_state, f"{args.out}/{tag}.pt")
print("saved", tag, "| params", n_params)