r"""
ed_trace_analysis.py
====================
Produces the three validation artifacts required to align the project with the
VV&A framework (Fonseca i Casas, 2023):

 1. EXECUTION TESTING  -> a time-series plot of a state variable (number of
    patients in the ED) for an understaffed vs a well-staffed scenario. Seeing
    the system fill up and stabilise at the expected level is the visual
    verification analogue of the reference report's inventory/loss traces.

 2. PATH ANALYSIS      -> the full event-by-event journey of one admitted
    patient, exposing the causal chain
    (triage -> all doctors busy -> queue -> treatment -> admission -> exit).

 3. RNG NUMERIC CHECK  -> quick statistical confirmation of the LCG in Python
    (the full randtoolbox battery lives in prng_validation.R, Appendix B).

Run:  python3 ed_trace_analysis.py
"""

from __future__ import annotations
import re
import math
import statistics

from ed_simulation import LCG, Simulator, EDNetwork, BASELINE


# ---------------------------------------------------------------------------
# 1. EXECUTION TESTING — number in system over time
# ---------------------------------------------------------------------------
def run_with_monitor(params, seed, horizon, dt=0.5):
    rng = LCG(seed=seed)
    sim = Simulator(trace=False)
    net = EDNetwork(params, rng, sim)
    net.source.start()
    ts, n_sys = [], []

    def sample():
        ts.append(sim.clock)
        n_sys.append(sum(s.n for s in net.stations))
        sim.schedule(dt, sample)

    sim.schedule(dt, sample)
    sim.run(horizon)
    return ts, n_sys


def execution_testing(path="ed_execution_testing.png", horizon=240.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    understaffed = dict(BASELINE)                 # c1=1, c2=3  (scenario "(1)")
    staffed = dict(BASELINE); staffed["c1"] = 2; staffed["c2"] = 4   # scenario "ab"

    t1, n1 = run_with_monitor(understaffed, seed=2024, horizon=horizon)
    t2, n2 = run_with_monitor(staffed, seed=2024, horizon=horizon)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(t1, n1, color="#D85A30", lw=1.0,
            label="understaffed  (1 nurse, 3 doctors)")
    ax.plot(t2, n2, color="#1D9E75", lw=1.0,
            label="well-staffed  (2 nurses, 4 doctors)")
    ax.axhline(statistics.mean(n1[40:]), color="#993C1D", ls="--", lw=0.8)
    ax.axhline(statistics.mean(n2[40:]), color="#0F6E56", ls="--", lw=0.8)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("patients in the ED")
    ax.set_title("Execution testing: patients in system over time")
    ax.legend(loc="upper right", fontsize=9)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved -> {path}")
    print(f"  understaffed mean (after warm-up) = {statistics.mean(n1[40:]):.2f}")
    print(f"  well-staffed mean (after warm-up) = {statistics.mean(n2[40:]):.2f}")
    return path


# ---------------------------------------------------------------------------
# 2. PATH ANALYSIS — one admitted patient's full journey
# ---------------------------------------------------------------------------
def path_analysis(seed=7, max_time=12.0):
    rng = LCG(seed=seed)
    sim = Simulator(trace=True)
    net = EDNetwork(BASELINE, rng, sim)
    net.source.start()

    # silence stdout printing from emit by capturing only log_lines
    sim.trace = True
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.run(max_time)

    admit_ids = []
    for line in sim.log_lines:
        m = re.search(r"route p(\d+)\s+Treatment -> Admission", line)
        if m:
            admit_ids.append(int(m.group(1)))
    if not admit_ids:
        print("(no admitted patient in window; increase max_time)")
        return None, []

    # prefer an admitted patient that actually waited in a queue (richer path)
    pid = admit_ids[0]
    for cand in admit_ids:
        jr = [l for l in sim.log_lines if re.search(rf"\bp{cand}\b", l)]
        if any("QUEUE" in l for l in jr):
            pid = cand
            break
    journey = [l for l in sim.log_lines if re.search(rf"\bp{pid}\b", l)]
    print(f"\nPATH ANALYSIS — patient p{pid} (admitted)\n" + "-" * 60)
    for l in journey:
        print(l)
    return pid, journey


# ---------------------------------------------------------------------------
# 3. RNG NUMERIC CHECK (Python cross-check of the LCG)
# ---------------------------------------------------------------------------
def rng_check(n=10000, seed=42, bins=20):
    from scipy import stats

    g = LCG(seed=seed)
    u = [g.random() for _ in range(n)]

    mean, var = statistics.mean(u), statistics.pvariance(u)

    # frequency / chi-square uniformity
    counts = [0] * bins
    for x in u:
        counts[min(int(x * bins), bins - 1)] += 1
    chi2, p_chi = stats.chisquare(counts)

    # Kolmogorov-Smirnov vs Uniform(0,1)
    ks, p_ks = stats.kstest(u, "uniform")

    # runs test (up/down)
    s = [1 if u[i + 1] > u[i] else -1 for i in range(n - 1)]
    runs = 1 + sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])
    mu_r = (2 * n - 1) / 3
    var_r = (16 * n - 29) / 90
    z = (runs - mu_r) / math.sqrt(var_r)
    p_runs = 2 * (1 - stats.norm.cdf(abs(z)))

    # lag-1 autocorrelation
    m = mean
    num = sum((u[i] - m) * (u[i + 1] - m) for i in range(n - 1))
    den = sum((x - m) ** 2 for x in u)
    r1 = num / den

    print("\nRNG NUMERIC CHECK (LCG, Python)\n" + "-" * 48)
    print(f"  draws                : {n}")
    print(f"  mean                 : {mean:.4f}   (target 0.5)")
    print(f"  variance             : {var:.4f}   (target 0.0833)")
    print(f"  chi-square uniformity: p = {p_chi:.3f}")
    print(f"  Kolmogorov-Smirnov   : p = {p_ks:.3f}")
    print(f"  runs (up/down)       : p = {p_runs:.3f}")
    print(f"  lag-1 autocorrelation: {r1:+.4f}   (target ~0)")
    return dict(mean=mean, var=var, p_chi=p_chi, p_ks=p_ks,
                p_runs=p_runs, r1=r1)


if __name__ == "__main__":
    execution_testing()
    path_analysis()
    rng_check()
