"""
doe_queue.py
============
Design of Experiments on the single-station queue (uses queue_sim.py).

Two factors, 2 levels each -> 2x2 = 4 configurations, 3 replicates = 12 runs.
  Factor A = arrival rate (load):   low  lambda = 0.5   /   high lambda = 0.9
  Factor B = number of servers:     1                  /   2
  Fixed:     service rate mu = 1.0,  M/M/c, FIFO, infinite capacity
  Response Y = Wq  (mean waiting time in queue)

We tabulate the runs, run a two-way ANOVA (effects of A, B and A:B on Wq),
and draw an interaction plot (colored matrix + line plot).
"""
import itertools
import statistics
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Importiamo i componenti corretti dal modulo del simulatore
from queue_sim import Simulator, Station, Distribution, Source, LCG

MU = 1.0
A_LEVELS = {"low": 0.5, "high": 0.9}     # arrival rate lambda
B_LEVELS = {1: 1, 2: 2}                   # number of servers c
N_REP = 3
HORIZON = 20000.0


def run_config(lam: float, c: int, seed: int) -> float:
    """Run one M/M/c replicate and return Wq (mean queue wait)."""
    # 1. Inizializzazione del generatore PRNG (coerente con R) per la replica corrente
    rng = LCG(seed=seed)
    
    # 2. Configurazione del motore a eventi discreti
    sim = Simulator()
    sim.trace_enabled = False  # Silenziamo il tracciamento degli eventi per velocizzare il DoE
    
    # 3. Definizione del processo di servizio -> Esponenziale (M) con media 1/mu
    service_dist = Distribution(code="M", mean=1.0 / MU, rng=rng)
    
    # 4. Creazione della stazione di servizio (Station) con i parametri corretti
    node = Station(
        name="Q",
        sim=sim,
        service=service_dist,
        c=c,
        capacity=float("inf"),
        discipline="FIFO"
    )
    
    # 5. Definizione del processo di arrivo -> Esponenziale (M) con media 1/lambda
    arrival_dist = Distribution(code="M", mean=1.0 / lam, rng=rng)
    
    # 6. Creazione e attivazione della sorgente degli arrivi collegata alla stazione
    source = Source(name="SRC", sim=sim, arrival=arrival_dist, target=node)
    source.start()
    
    # 7. Esecuzione della simulazione fino all'orizzonte impostato
    sim.run(until=HORIZON)
    
    # 8. Consolidamento dei report integrati nel tempo e lettura di Wq
    node.finalize()
    stats = node.report(horizon=HORIZON)
    
    return stats["Wq"]


def mmc_wq(lam, mu, c):
    """Closed-form Wq for M/M/c (sanity reference)."""
    import math
    a = lam / mu
    rho = a / c
    if rho >= 1:
        return float("nan")
    s = sum(a ** n / math.factorial(n) for n in range(c))
    last = a ** c / (math.factorial(c) * (1 - rho))
    P0 = 1 / (s + last)
    Lq = P0 * a ** c * rho / (math.factorial(c) * (1 - rho) ** 2)
    return Lq / lam


def run_design():
    rows = []
    seed = 100
    for (aname, lam), (bname, c) in itertools.product(A_LEVELS.items(), B_LEVELS.items()):
        for rep in range(1, N_REP + 1):
            wq = run_config(lam, c, seed)
            seed += 1
            rows.append({"A": aname, "B": bname, "lambda": lam, "c": c,
                         "rep": rep, "Wq": wq})
    return pd.DataFrame(rows)


def main():
    df = run_design()

    # ---- results table -------------------------------------------------
    print("=" * 70)
    print("2x2 FACTORIAL  ·  response = Wq (mean queue wait)  ·  mu = 1.0")
    print("  A = arrival rate (low 0.5 / high 0.9)   B = servers (1 / 2)")
    print(f"  {N_REP} replicates per cell, horizon = {HORIZON:.0f} time units")
    print("=" * 70)
    wide = df.pivot_table(index=["A", "lambda"], columns=["B", "c"],
                          values="Wq", aggfunc=list)
    # readable per-run table
    print(f"\n{'A':<6}{'lambda':>7}{'B(servers)':>12}{'rep1':>10}{'rep2':>10}{'rep3':>10}{'mean':>10}")
    for (aname, lam), (bname, c) in itertools.product(A_LEVELS.items(), B_LEVELS.items()):
        sub = df[(df.A == aname) & (df.B == bname)].sort_values("rep")
        ys = sub.Wq.tolist()
        print(f"{aname:<6}{lam:>7}{c:>12}{ys[0]:>10.4f}{ys[1]:>10.4f}{ys[2]:>10.4f}{statistics.mean(ys):>10.4f}")

    # theory reference
    print("\nclosed-form M/M/c Wq (reference):")
    for (aname, lam), (bname, c) in itertools.product(A_LEVELS.items(), B_LEVELS.items()):
        print(f"  A={aname:<5} B={c}: Wq_theory = {mmc_wq(lam, MU, c):.4f}")

    # ---- two-way ANOVA -------------------------------------------------
    model = ols("Wq ~ C(A) * C(B)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    anova["sig"] = ["***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                    for p in anova["PR(>F)"]]
    print("\n" + "=" * 70)
    print("TWO-WAY ANOVA  (H0: factor / interaction has no effect on Wq)")
    print("=" * 70)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(anova.rename(columns={"PR(>F)": "p-value"}))
    print("\nLegend: C(A)=arrival rate, C(B)=servers, C(A):C(B)=interaction")

    make_plot(df)
    return df, anova


def make_plot(df, path="doe_interaction.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    means = df.groupby(["A", "B"])["Wq"].mean().unstack()      # rows A, cols B
    means = means.reindex(index=["low", "high"], columns=[1, 2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # ---- colored matrix (heatmap) ----
    M = means.values
    im = ax1.imshow(M, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["1 server", "2 servers"])
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["λ = 0.5 (low)", "λ = 0.9 (high)"])
    ax1.set_xlabel("B — servers"); ax1.set_ylabel("A — arrival rate")
    ax1.set_title("Mean Wq by configuration (colored matrix)")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                     color="black" if M[i, j] < (M.max() * 0.6) else "white",
                     fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax1, label="mean Wq")

    # ---- interaction line plot ----
    x = [0, 1]
    ax2.plot(x, means[1].values, "-o", color="#D85A30", lw=2, label="1 server")
    ax2.plot(x, means[2].values, "-o", color="#1D9E75", lw=2, label="2 servers")
    ax2.set_xticks(x); ax2.set_xticklabels(["λ = 0.5 (low)", "λ = 0.9 (high)"])
    ax2.set_xlabel("A — arrival rate"); ax2.set_ylabel("mean Wq")
    ax2.set_title("Interaction plot")
    ax2.legend(title="B (servers)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nsaved interaction plot -> {path}")


if __name__ == "__main__":
    main()