r"""
ed_simulation.py
================
Discrete-event (event-scheduling) simulation of a hospital Emergency
Department modelled as an OPEN JACKSON QUEUEING NETWORK.

Topology (3 stages, one probabilistic branch):

      lambda          all              p_admit
   ----------> [Triage] ----> [Treatment] ----> [Admission] ----> out
                c1 nurses      c2 doctors  \      c3 clerks
                                            \  1 - p_admit
                                             ------------------> out (discharged)

Why this design
---------------
* Multi-stage  : required by the assignment ("as long as it is multi stage").
* Jackson-form : every station is M/M/c with Poisson external input and
                 exponential service, FIFO, infinite capacity. Jackson's
                 theorem then gives a CLOSED-FORM steady state per station, so
                 the simulation can be validated ANALYTICALLY (stronger than a
                 GPSS-only check). The GPSS model (ed_model.gps) is the second,
                 independent validation that the assignment asks for explicitly.

The pseudo-random numbers come from the SAME LCG validated in
prng_validation.R, so RNG validation (mandatory) and the simulation are one
coherent pipeline.

Units: time in HOURS.
"""

from __future__ import annotations
import heapq
import math
import itertools
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Optional, List, Dict


# ===========================================================================
# 1. RNG  — Linear Congruential Generator (identical to prng_validation.R)
# ===========================================================================
class LCG:
    def __init__(self, seed: int = 42, m: int = 2 ** 31,
                 a: int = 1103515245, c: int = 12345):
        self.m, self.a, self.c = m, a, c
        self.state = seed % m

    def random(self) -> float:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m


# ===========================================================================
# 2. Distribution layer (Kendall letter -> sampler)
# ===========================================================================
class Distribution:
    def __init__(self, code: str, mean: float, rng: LCG):
        self.code, self.mean, self.rng = code.upper(), mean, rng

    def sample(self) -> float:
        if self.code == "M":
            u = max(self.rng.random(), 1e-12)
            return -self.mean * math.log(u)
        if self.code == "D":
            return self.mean
        raise ValueError(f"unsupported code {self.code!r}")


# ===========================================================================
# 3. Event-scheduling engine (Future Event List = binary heap)
# ===========================================================================
@dataclass(order=True)
class _Ev:
    time: float
    seq: int
    cb: Callable = field(compare=False)
    args: tuple = field(compare=False, default=())


class Simulator:
    def __init__(self, trace: bool = False):
        self.clock = 0.0
        self._fel: List[_Ev] = []
        self._ctr = itertools.count()
        self.trace = trace
        self.log_lines: List[str] = []

    def schedule(self, delay: float, cb: Callable, *args) -> None:
        heapq.heappush(self._fel, _Ev(self.clock + delay, next(self._ctr), cb, args))

    def emit(self, msg: str) -> None:
        if self.trace:
            line = f"[t={self.clock:8.3f}] {msg}"
            self.log_lines.append(line)
            print(line)

    def run(self, until: float) -> None:
        while self._fel and self._fel[0].time <= until:
            ev = heapq.heappop(self._fel)
            self.clock = ev.time           # event-driven advance
            ev.cb(*ev.args)


# ===========================================================================
# 4. Patient entity
# ===========================================================================
@dataclass
class Patient:
    eid: int
    t_sys_in: float                 # entry to the whole network
    admitted: bool = False
    t_stn_in: float = 0.0           # entry to current station
    t_enq: float = 0.0
    t_svc: float = 0.0


# ===========================================================================
# 5. Station  =  M/M/c node (FIFO, infinite capacity)
# ===========================================================================
class Station:
    def __init__(self, name: str, sim: Simulator, service: Distribution,
                 c: int, router: Optional[Callable[[Patient], "Station"]] = None):
        self.name, self.sim, self.service, self.c = name, sim, service, c
        self.router = router
        self.network: Optional["EDNetwork"] = None
        self.queue: deque[Patient] = deque()
        self.busy = 0
        self.reset_stats(0.0)

    def reset_stats(self, t: float) -> None:
        self.arrivals = self.served = 0
        self._t0 = t
        self._last = t
        self._area_L = self._area_Lq = self._busy_time = 0.0
        self._sum_W = self._sum_Wq = 0.0

    @property
    def n(self) -> int:
        return len(self.queue) + self.busy

    def _acc(self) -> None:
        dt = self.sim.clock - self._last
        if dt > 0:
            self._area_L += self.n * dt
            self._area_Lq += len(self.queue) * dt
            self._busy_time += self.busy * dt
        self._last = self.sim.clock

    def arrive(self, p: Patient) -> None:
        self._acc()
        self.arrivals += 1
        p.t_stn_in = self.sim.clock
        if self.busy < self.c:
            self.busy += 1
            p.t_enq = p.t_svc = self.sim.clock
            st = self.service.sample()
            self.sim.schedule(st, self.depart, p)
            self.sim.emit(f"ARRIVE p{p.eid} -> {self.name} SERVICE busy={self.busy}/{self.c} q={len(self.queue)}")
        else:
            p.t_enq = self.sim.clock
            self.queue.append(p)
            self.sim.emit(f"ARRIVE p{p.eid} -> {self.name} QUEUE  busy={self.busy}/{self.c} q={len(self.queue)}")

    def depart(self, p: Patient) -> None:
        self._acc()
        self.busy -= 1
        self.served += 1
        self._sum_W += self.sim.clock - p.t_stn_in
        self._sum_Wq += p.t_svc - p.t_enq
        self.sim.emit(f"DEPART p{p.eid} <- {self.name}  busy={self.busy}/{self.c} q={len(self.queue)}")
        # pull next waiting patient
        if self.queue:
            nxt = self.queue.popleft()
            self.busy += 1
            nxt.t_svc = self.sim.clock
            st = self.service.sample()
            self.sim.schedule(st, self.depart, nxt)
        # route onward (modular hand-off) or exit the network
        target = self.router(p) if self.router else None
        if target is not None:
            self.sim.emit(f"  route p{p.eid}  {self.name} -> {target.name}")
            target.arrive(p)
        elif self.network is not None:
            self.network.record_exit(p)

    def finalize(self) -> None:
        self._acc()

    def report(self) -> dict:
        T = self.sim.clock - self._t0
        return dict(
            L=self._area_L / T, Lq=self._area_Lq / T,
            rho=self._busy_time / (self.c * T),
            W=self._sum_W / self.served if self.served else float("nan"),
            Wq=self._sum_Wq / self.served if self.served else float("nan"),
            arrivals=self.arrivals, served=self.served)


# ===========================================================================
# 6. Source (external Poisson arrivals)
# ===========================================================================
class Source:
    def __init__(self, sim: Simulator, arrival: Distribution, target: Station):
        self.sim, self.arrival, self.target = sim, arrival, target
        self._n = 0

    def start(self) -> None:
        self._next()

    def _next(self) -> None:
        self.sim.schedule(self.arrival.sample(), self._gen)

    def _gen(self) -> None:
        self._n += 1
        self.target.arrive(Patient(eid=self._n, t_sys_in=self.sim.clock))
        self._next()


# ===========================================================================
# 7. The Emergency-Department network
# ===========================================================================
class EDNetwork:
    """Wires Triage -> Treatment -> (Admission | discharge) and tracks LOS."""

    def __init__(self, params: dict, rng: LCG, sim: Simulator):
        self.sim = sim
        self.rng = rng
        self.params = params
        p_admit = params["p_admit"]

        # stations (built downstream-first so routers can reference targets)
        self.admission = Station("Admission", sim,
                                 Distribution("M", 1.0 / params["mu3"], rng),
                                 c=params["c3"], router=lambda p: None)
        self.treatment = Station("Treatment", sim,
                                 Distribution("M", 1.0 / params["mu2"], rng),
                                 c=params["c2"], router=self._after_treatment)
        self.triage = Station("Triage", sim,
                              Distribution("M", 1.0 / params["mu1"], rng),
                              c=params["c1"], router=lambda p: self.treatment)
        self.stations = [self.triage, self.treatment, self.admission]
        for s in self.stations:
            s.network = self

        self.source = Source(sim, Distribution("M", 1.0 / params["lam"], rng),
                             self.triage)

        # network-level LOS accumulators
        self.exits = 0
        self._sum_los = 0.0
        self._collecting = True

    def _after_treatment(self, p: Patient) -> Optional[Station]:
        if self.rng.random() < self.params["p_admit"]:
            p.admitted = True
            return self.admission
        return None                     # discharged -> leaves network

    def record_exit(self, p: Patient) -> None:
        if self._collecting:
            self.exits += 1
            self._sum_los += self.sim.clock - p.t_sys_in

    def reset_stats(self) -> None:
        for s in self.stations:
            s.reset_stats(self.sim.clock)
        self.exits = 0
        self._sum_los = 0.0
        self._collecting = True

    def finalize(self) -> None:
        for s in self.stations:
            s.finalize()

    def los_mean(self) -> float:
        return self._sum_los / self.exits if self.exits else float("nan")


# ===========================================================================
# 8. Jackson-network closed-form theory  (VALIDATION TARGET)
# ===========================================================================
def mmc(lam: float, mu: float, c: int) -> Optional[dict]:
    """Erlang-C M/M/c metrics; None if unstable."""
    a = lam / mu                       # offered load (Erlangs)
    rho = a / c
    if rho >= 1.0:
        return None
    s = sum(a ** n / math.factorial(n) for n in range(c))
    last = a ** c / (math.factorial(c) * (1 - rho))
    P0 = 1.0 / (s + last)
    Lq = P0 * (a ** c) * rho / (math.factorial(c) * (1 - rho) ** 2)
    Wq = Lq / lam
    W = Wq + 1.0 / mu
    L = lam * W                        # = Lq + a
    return dict(L=L, Lq=Lq, W=W, Wq=Wq, rho=rho)


def ed_theory(params: dict) -> dict:
    """Per-station Jackson metrics + network mean LOS (Little's law)."""
    lam = params["lam"]
    # traffic equations
    lam1 = lam
    lam2 = lam
    lam3 = params["p_admit"] * lam
    st = {
        "Triage":    mmc(lam1, params["mu1"], params["c1"]),
        "Treatment": mmc(lam2, params["mu2"], params["c2"]),
        "Admission": mmc(lam3, params["mu3"], params["c3"]),
    }
    if any(v is None for v in st.values()):
        return {"stable": False, "stations": st}
    L_total = sum(v["L"] for v in st.values())
    Lq_total = sum(v["Lq"] for v in st.values())
    los = L_total / lam                # Little's law over the whole network
    return {"stable": True, "stations": st, "lam": {"Triage": lam1, "Treatment": lam2, "Admission": lam3},
            "L_total": L_total, "Lq_total": Lq_total, "LOS": los}


# ===========================================================================
# 9. Cost model  (the DoE response)
#    daily cost = staffing + waiting penalty (per patient-hour queued)
# ===========================================================================
def ed_cost(net: EDNetwork, params: dict) -> dict:
    rep = {s.name: s.report() for s in net.stations}
    Lq_total = sum(rep[n]["Lq"] for n in rep)
    staffing_per_h = (params["c1"] * params["R_nurse"]
                      + params["c2"] * params["R_doctor"]
                      + params["c3"] * params["R_clerk"])
    waiting_per_h = params["P_wait"] * Lq_total
    cost_per_h = staffing_per_h + waiting_per_h
    return dict(cost_day=24 * cost_per_h,
                staffing_day=24 * staffing_per_h,
                waiting_day=24 * waiting_per_h,
                Lq_total=Lq_total,
                los=net.los_mean())


# ===========================================================================
# 10. Runners
# ===========================================================================
def run_ed(params: dict, seed: int, horizon: float, warmup: float = 0.0,
           trace: bool = False, trace_events: int = 0) -> dict:
    rng = LCG(seed=seed)
    sim = Simulator(trace=trace)
    net = EDNetwork(params, rng, sim)
    net.source.start()

    if trace and trace_events > 0:
        # trace only the first `trace_events` events, then continue silently
        sim.run(until=horizon)         # tracing handled inside emit; we cap below
    if warmup > 0:
        sim.run(until=warmup)
        net.reset_stats()
    sim.run(until=horizon)
    net.finalize()

    rep = {s.name: s.report() for s in net.stations}
    cost = ed_cost(net, params)
    return dict(stations=rep, cost=cost, los=net.los_mean(), exits=net.exits)


def run_replicas(params: dict, seeds: List[int], horizon: float,
                 warmup: float) -> List[dict]:
    return [run_ed(params, s, horizon, warmup) for s in seeds]


# ===========================================================================
# 11. Baseline scenario + analytical validation + trace sample
# ===========================================================================
BASELINE = dict(
    lam=5.0,            # 5 patients / hour
    mu1=6.0, c1=1,      # triage: 10 min/patient, 1 nurse   (rho ~ 0.83)
    mu2=2.0, c2=3,      # treatment: 30 min/patient, 3 doctors (rho ~ 0.83)
    mu3=4.0, c3=1,      # admission: 15 min/patient, 1 clerk (rho ~ 0.31)
    p_admit=0.25,       # 25% of treated patients are admitted
    # cost parameters ($/hour)
    R_nurse=40.0, R_doctor=150.0, R_clerk=30.0,
    P_wait=100.0,       # penalty per patient-hour spent waiting in any queue
)


def _banner(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def demo_baseline():
    _banner("BASELINE  lambda=5/h  Triage(1x6) Treatment(3x2) Admission(1x4)  p_admit=0.25")

    # ---- short traced run so the TRACE can be validated ------------------
    print("\n--- TRACE (first ~24 events) ---")
    rng = LCG(seed=1)
    sim = Simulator(trace=True)
    net = EDNetwork(BASELINE, rng, sim)
    net.source.start()
    sim.run(until=4.0)                 # a few hours is enough for ~20+ events
    sim.trace = False
    print(f"... (trace continues; {len(sim.log_lines)} events logged) ...")

    # ---- long run for tight comparison to Jackson theory -----------------
    res = run_ed(BASELINE, seed=12345, horizon=60000.0, warmup=500.0)
    th = ed_theory(BASELINE)

    print("\n--- PER-STATION:  simulation vs Jackson theory ---")
    hdr = f"{'station':<11}{'metric':<5}{'sim':>10}{'theory':>10}{'err%':>8}"
    for name in ("Triage", "Treatment", "Admission"):
        print(f"\n[{name}]  (lambda_eff = {th['lam'][name]:.3f} /h)")
        print(hdr)
        sv, tv = res["stations"][name], th["stations"][name]
        for k in ("rho", "L", "Lq", "W", "Wq"):
            e = 100 * abs(sv[k] - tv[k]) / tv[k] if tv[k] else 0.0
            print(f"{'':<11}{k:<5}{sv[k]:>10.4f}{tv[k]:>10.4f}{e:>8.2f}")

    print("\n--- NETWORK ---")
    los_sim = res["los"]
    print(f"mean Length-of-Stay   sim={los_sim:7.4f} h   theory={th['LOS']:7.4f} h"
          f"   err={100*abs(los_sim-th['LOS'])/th['LOS']:.2f}%")
    print(f"total in system  L    sim={sum(res['stations'][n]['L'] for n in res['stations']):7.4f}"
          f"     theory={th['L_total']:7.4f}")

    print("\n--- DAILY COST (baseline) ---")
    c = res["cost"]
    print(f"staffing = ${c['staffing_day']:,.0f}/day   "
          f"waiting = ${c['waiting_day']:,.0f}/day   "
          f"TOTAL = ${c['cost_day']:,.0f}/day")


if __name__ == "__main__":
    demo_baseline()
