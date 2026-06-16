"""
doe_mesh.py
===========
Design of Experiments on the 3-station MESH network (uses queue_sim.py,
the mesh simulator). The reference topology is the heterogeneous Example-2
mesh, kept FIXED; three global factors are varied around it.

Reference mesh (fixed):
  S1: M/M/c1, mu=1.0, FIFO, reachable {2,3}
  S2: M/D/c2, mu=1.2, LIFO, reachable {1,3}
  S3: D/M/c3, mu=0.9, SIRO, reachable {1,2}     (S3 is the bottleneck)

2^3 full factorial (8 configurations x 3 replicates = 24 runs):
  Factor A = load      : external lambda per station   low 0.3 / high 0.4
  Factor B = servers   : base (c = 1,2,1) / plus1 (c = 2,3,2)
  Factor C = capacity  : infinite / finite (N = 5 at every station)

Responses:
  W    = mean end-to-end time of served entities (primary)
  loss = blocked entities / total arrivals       (secondary; >0 only if N finite)

A three-way ANOVA is run on W; cell means of W and loss are tabulated; and an
interaction plot (W vs load, by servers, paneled by capacity) is saved.

Note: the mesh has no closed-form solution (superposed external + routed
arrivals are not exactly Poisson, and routing has no product-form here), so
this experiment is EXPLORATORY -- it complements, and does not replace, the
single-station DoE that is validated against the M/M/c formulas.
"""
import random
import itertools
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from queue_sim import (Simulation, QueueNode,
                       DistributionArrivalTimes as A,
                       DistributionServiceTimes as S,
                       QueueDiscipline as D)

LOAD    = {"low": 0.3, "high": 0.4}        # external lambda per station
SERVERS = {"base": (1, 2, 1), "plus1": (2, 3, 2)}
CAP     = {"inf": None, "finite": 5}
N_REP   = 3
HORIZON = 20000.0


def build_mesh(lam, servers, cap):
    c1, c2, c3 = servers
    s1 = QueueNode(id=1, name="S1", arrival_rate=lam, service_rate=1.0,
                   arrival_distribution=A.M, service_distribution=S.M,
                   num_servers=c1, sys_capacity=cap,
                   queue_discipline=D.FIFO, reachable=[2, 3])
    s2 = QueueNode(id=2, name="S2", arrival_rate=lam, service_rate=1.2,
                   arrival_distribution=A.M, service_distribution=S.D,
                   num_servers=c2, sys_capacity=cap,
                   queue_discipline=D.LIFO, reachable=[1, 3])
    s3 = QueueNode(id=3, name="S3", arrival_rate=lam, service_rate=0.9,
                   arrival_distribution=A.D, service_distribution=S.M,
                   num_servers=c3, sys_capacity=cap,
                   queue_discipline=D.SIRO, reachable=[1, 2])
    return [s1, s2, s3]


def run_cell(lam, servers, cap, seed):
    random.seed(seed)
    nodes = build_mesh(lam, servers, cap)
    sim = Simulation(nodes=nodes)
    sim.run(until=HORIZON)
    st = sim.final_statistics()
    W = st["overall"]["W"]
    arrivals = sum(n.total_arrivals for n in nodes)
    lost = sum(n.total_lost for n in nodes)
    return W, (lost / arrivals if arrivals else 0.0)


def run_design():
    rows, seed = [], 200
    for (an, lam), (bn, sv), (cn, cap) in itertools.product(
            LOAD.items(), SERVERS.items(), CAP.items()):
        for rep in range(1, N_REP + 1):
            W, loss = run_cell(lam, sv, cap, seed)
            seed += 1
            rows.append({"load": an, "servers": bn, "capacity": cn, "rep": rep,
                         "W": W, "loss": loss})
    return pd.DataFrame(rows)


def main():
    df = run_design()

    print("=" * 72)
    print("2^3 FACTORIAL on the MESH  ·  3 replicates per cell, horizon =",
          int(HORIZON))
    print("  A = load (lambda 0.3/0.4)   B = servers (base/plus1)   C = capacity (inf/finite N=5)")
    print("=" * 72)
    cells = df.groupby(["load", "servers", "capacity"])[["W", "loss"]].mean().round(4)
    print("\nCell means (W = end-to-end time, loss = blocked fraction):")
    print(cells)

    model = ols("W ~ C(load) * C(servers) * C(capacity)", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    aov["sig"] = ["***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                  for p in aov["PR(>F)"]]
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 72)
    print("THREE-WAY ANOVA on W   (H0: term has no effect on end-to-end time)")
    print("=" * 72)
    print(aov.rename(columns={"PR(>F)": "p-value"}))

    make_plot(df)
    return df, aov


def make_plot(df, path="doe_mesh_interaction.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    order_load = ["low", "high"]
    for ax, capn in zip(axes, ["inf", "finite"]):
        sub = df[df.capacity == capn]
        for svn, color in [("base", "#D85A30"), ("plus1", "#1D9E75")]:
            m = (sub[sub.servers == svn].groupby("load")["W"].mean().reindex(order_load))
            ax.plot([0, 1], m.values, "-o", color=color, lw=2, label=svn)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["low λ=0.3", "high λ=0.4"])
        ax.set_xlabel("A — load"); ax.set_title(f"capacity = {capn}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean end-to-end time W")
    axes[1].legend(title="B (servers)")
    fig.suptitle("Mesh DoE — interaction of load and servers on W (by capacity)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nsaved mesh interaction plot -> {path}")


if __name__ == "__main__":
    main()
