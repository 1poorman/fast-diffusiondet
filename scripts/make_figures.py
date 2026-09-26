# 终版图表：AP-NFE 曲线 / 训练 LOSS 曲线对比 / latency-AP 权衡
# 产出：results/figures/*.png（300 dpi，论文可用）
import csv, json, os, re, sys
from collections import defaultdict
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 300})

# ============ 图 1：AP-NFE 曲线（E0-CTRL bs8 终版矩阵） ============
ap = defaultdict(list)
for r in csv.DictReader(open(f"{ROOT}/results/raw/m4_e0ctrl_full.csv")):
    ap[(r["solver"], int(r["actual_nfe"]))].append(float(r["AP"]))

offsets = {"ddim": -0.12, "euler": 0.0, "heun": 0.12, "dpm_v3": 0.24}
colors = {"ddim": "#888888", "euler": "#1f77b4", "heun": "#2ca02c", "dpm_v3": "#d62728"}
labels = {"ddim": "DDIM (eta=1)", "euler": "Euler", "heun": "Heun", "dpm_v3": "DPM-Solver++ (DPP)"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for solver in ["ddim", "euler", "heun", "dpm_v3"]:
    pts = sorted((n, st.mean(v), st.stdev(v)) for (s, n), v in ap.items() if s == solver)
    xs = [n + offsets[solver] for n, m, sd in pts]
    ys = [m for _, m, _ in pts]
    es = [sd for _, _, sd in pts]
    ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=3, lw=1.6,
                color=colors[solver], label=labels[solver])
ax.set_xlabel("NFE (head forward count)")
ax.set_ylabel("COCO AP")
ax.set_title("AP vs NFE  (SDD, E0-CTRL D4-bs8 model)")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# ============ 图 2：latency-AP 权衡 ============
lat = defaultdict(list)
for r in csv.DictReader(open(f"{ROOT}/results/raw/m4_e0ctrl_full.csv")):
    lat[(r["solver"], int(r["actual_nfe"]))].append(
        (float(r["AP"]), float(r["latency_ms_median"])))
ax = axes[1]
for solver in ["ddim", "euler", "heun", "dpm_v3"]:
    pts = sorted((n, st.mean([a for a, _ in v]), st.median([l for _, l in v]))
                 for (s, n), v in lat.items() if s == solver)
    xs = [l for _, _, l in pts]
    ys = [m for _, m, _ in pts]
    ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=colors[solver], label=labels[solver])
    for n, m, l in pts:
        ax.annotate(f"N{n}", (l, m), fontsize=7, xytext=(3, 4), textcoords="offset points")
ax.set_xlabel("latency (ms, median, eval bs=32)")
ax.set_ylabel("COCO AP")
ax.set_title("AP vs latency trade-off")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_ap_nfe_curves.png", bbox_inches="tight")
print("fig1 saved")

# ============ 图 3：训练 LOSS 曲线对比（D4 策略三个数据集 + bs2 对照） ============
def load_metrics(path):
    """metrics.json 每行一个 JSON 记录；time>0 过滤 eval 行。"""
    its, losses, evals = [], [], []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "total_loss" in d and d.get("time", 0) > 0:
                its.append(d["iteration"]); losses.append(d["total_loss"])
            if "bbox/AP" in d:
                evals.append((d["iteration"], d["bbox/AP"]))
    return its, losses, evals


def smooth(v, k=20):
    return [sum(v[max(0, i - k):i + 1]) / len(v[max(0, i - k):i + 1]) for i in range(len(v))]


fig, axes = plt.subplots(1, 3, figsize=(14, 4))
specs = [
    ("SDD (7 cls, D4 bs8, 1750 it)", f"{ROOT}/output/lab/sdd.res50.bs8/log.txt"),
    ("PLS (16 cls, D4 bs8, 3000 it)", f"{ROOT}/output/lab/pls.res50.bs8/log.txt"),
    ("WHEAT (12 cls, D4 bs8, 1400 it)", f"{ROOT}/output/lab/wheat.res50.bs8/log.txt"),
]
for ax, (title, p) in zip(axes, specs):
    if not os.path.exists(p):
        ax.set_title(title + "\n(missing)"); continue
    its, losses, evals = load_metrics(p)
    ax.plot(its, losses, alpha=0.25, lw=0.8, color="#1f77b4")
    ax.plot(its, smooth(losses), lw=1.8, color="#1f77b4", label="total_loss (smoothed)")
    for it, vap in evals:
        ax.axvline(it, color="#d62728", ls=":", lw=0.8, alpha=0.6)
        ax.annotate(f"AP {vap:.1f}", (it, ax.get_ylim()[1]), fontsize=7,
                    rotation=90, va="top", color="#d62728")
    ax.set_title(title); ax.set_xlabel("iteration"); ax.set_ylabel("total loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_training_loss.png", bbox_inches="tight")
print("fig2 saved")

# ============ 图 4：bs2(v1.0) vs bs8(D4) 训练对照（SDD） ============
p_old = f"{ROOT}/output/lab/sdd.res50/log.txt"
p_new = f"{ROOT}/output/lab/sdd.res50.bs8/log.txt"
if os.path.exists(p_old) and os.path.exists(p_new):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    i1, l1, e1 = load_metrics(p_old)
    i2, l2, e2 = load_metrics(p_new)
    ax.plot([i * 2 for i in i1], l1, alpha=0.2, lw=0.8, color="#888")
    ax.plot([i * 2 for i in i1], smooth(l1, 40), lw=1.6, color="#888",
            label="v1.0 bs2/LR2.5e-5 (3500 it, AP 61.83)")
    ax.plot(i2, l2, alpha=0.2, lw=0.8, color="#d62728")
    ax.plot(i2, smooth(l2), lw=1.6, color="#d62728",
            label="D4 bs8/LR1e-4 (1750 it, AP 67.16)")
    for it, vap in e2:
        ax.annotate(f"{vap:.1f}", (it, vap), fontsize=8, color="#d62728",
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("images seen (thousands) = iter x bs")
    ax.set_ylabel("total loss")
    ax.set_title("D4 training strategy: bs2 vs bs8 (SDD)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_d4_strategy.png", bbox_inches="tight")
    print("fig3 saved")
else:
    print("fig3 skipped (missing metrics)")
