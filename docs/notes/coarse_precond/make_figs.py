# Figures for coarse_precond.tex, from the outputs in data/.
# Usage: python3 make_figs.py   (writes figs/*.pdf)
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
data = lambda f: os.path.join(here, "data", f)
fig = lambda f: os.path.join(here, "figs", f)
os.makedirs(os.path.join(here, "figs"), exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

def rows(path):
    out = []
    for line in open(path):
        s = line.split()
        if s and re.fullmatch(r"\d+", s[0]) and len(s) >= 5:
            out.append(s)
    return out

# ---------------------------------------------------------------- spectra
lab = {"regular": "regular ordering", "ashtari": "Ashtari ordering",
       "precond": r"Ashtari, $A M^{-1}$, $k = 8$"}
ev = {}
for line in open(data("spectrum.txt")):
    re_, im_, l = line.split()
    ev.setdefault(l, []).append(complex(float(re_), float(im_)))
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
for a, l in zip(ax, ("regular", "ashtari", "precond")):
    z = np.array(ev[l])
    a.plot(z.real, z.imag, ".", ms=2.5, color="C0")
    a.plot([0], [0], "+", color="C3", ms=8, mew=1.5)
    a.set_title(lab[l]); a.set_aspect("equal"); a.set_xlabel(r"Re $\lambda$")
    th = np.linspace(0, 2*np.pi, 200)
    c = -1 if l == "ashtari" else (1 if l == "precond" else None)
    if c is not None:
        a.plot(c + np.cos(th), np.sin(th), "--", lw=0.6, color="0.5")
ax[0].set_ylabel(r"Im $\lambda$")
f.savefig(fig("spectrum.pdf"))

# ---------------------------------------------- GMRES vs T, trajectory operators
k0 = [(float(r[1]), int(r[3])) for r in rows(data("trajectory_k0.txt")) if r[2] == "0"]
co = rows(data("trajectory_coarse.txt")) + rows(data("trajectory_k567.txt"))
kc = {4: "C1", 5: "C3", 6: "C4", 7: "C5", 8: "C0", 12: "C2"}
f, a = plt.subplots(figsize=(5.2, 3.2))
T, m = zip(*k0); a.loglog(T, m, "o-", label="no preconditioner", color="k", ms=3)
for k, c in kc.items():
    d = sorted((float(r[1]), int(r[3])) for r in co if r[2] == str(k))
    T, m = zip(*d); a.loglog(T, m, "s-", label=f"coarse, $k={k}$", color=c, ms=3)
d = [(float(r[1]), int(r[3])) for r in rows(data("sweeps_20.txt"))]
T, m = zip(*d); a.loglog(T, m, "^", color="C1", mfc="none", ms=6, label="$k=4$, 20 sweeps")
Tr = np.array([16, 2048]); a.loglog(Tr, 3*Tr**1 / 1, ":", color="0.5", lw=0.8)
a.text(300, 1300, r"$\propto T$", color="0.4")
a.set_xlabel("period $T$"); a.set_ylabel("GMRES iterations (rtol $10^{-6}$)")
a.set_ylim(1.5, 3000); a.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
f.savefig(fig("gmres_trajectory.pdf"))

# ------------------------------------------------- Newton searches, long orbits
U = []
for line in open(data("upo_newton.txt")):
    s = line.split()
    if s and re.fullmatch(r"\d+", s[0]):
        its = [int(x) for x in line.split("(")[0].split()[6:]]
        U.append(dict(M=int(s[0]), N=int(s[1]), T=float(s[2]), k=int(s[3]), its=its))
def work(u, sweeps=5):
    # segment integrations of a whole search: GMRES matvecs, one initial
    # residual per solve, and (coarse) k per step plus `sweeps*k` at the first step
    n = len(u["its"])
    per = sum(u["its"]) + n + (u["k"]*(n + sweeps) if u["k"] > 0 else 0)
    return per*u["N"]
f, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for k, c, l in ((0, "k", "no preconditioner"), (8, "C0", "coarse, $k=8$")):
    d = [u for u in U if u["k"] == k]
    T = [u["T"] for u in d]
    ax[0].loglog(T, [np.mean(u["its"]) for u in d], "o-", color=c, ms=3, label=l)
    ax[1].loglog(T, [work(u) for u in d], "o-", color=c, ms=3, label=l)
ax[0].set_xlabel("period $T$"); ax[0].set_ylabel("GMRES iterations per Newton step")
ax[1].set_xlabel("period $T$"); ax[1].set_ylabel("segment integrations per search")
Tr = np.array([53, 425])
ax[1].loglog(Tr, 2.2e3*(Tr/53)**2, ":", color="0.5", lw=0.8); ax[1].text(150, 6e4, r"$\propto T^2$", color="0.4")
ax[1].loglog(Tr, 1.2e3*(Tr/53), "--", color="0.5", lw=0.8); ax[1].text(200, 3.2e3, r"$\propto T$", color="0.4")
ax[0].legend(fontsize=7); ax[0].set_ylim(1.5, 800)
for a in ax:
    a.set_xticks([53, 106, 213, 425]); a.set_xticklabels(["53", "106", "213", "425"]); a.minorticks_off()
f.savefig(fig("newton_upo.pdf"))

# table of the Newton work, printed for the note
for u in U:
    print(u["M"], u["N"], u["T"], u["k"], u["its"], work(u))

# ------------------------------------------------ segment length τ = 2, 4, 6
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for a, t in zip(ax, (2, 4, 6)):
    R = rows(data(f"tau{t}.txt"))
    for k, c, l in ((0, "k", "no prec."), (2, "C3", "$k=2$"), (4, "C1", "$k=4$"), (8, "C0", "$k=8$")):
        d = sorted((float(r[1]), int(r[3])) for r in R if r[2] == str(k))
        T, m = zip(*d); a.loglog(T, m, "o-" if k == 0 else "s-", color=c, ms=3, label=l)
    a.set_title(rf"$\tau = {t}$"); a.set_xlabel("period $T$")
    a.set_xticks([48, 192, 768]); a.set_xticklabels(["48", "192", "768"]); a.minorticks_off()
ax[0].set_ylabel("GMRES iterations (rtol $10^{-6}$)"); ax[0].set_ylim(2, 1200)
ax[2].legend(fontsize=7, loc="upper left")
f.savefig(fig("tau.pdf"))
