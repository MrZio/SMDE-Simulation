"""
queue_sim.py
============
Event-scheduling simulation of queueing systems described in Kendall's
notation  A / S / c / N / D.

Design goals (from the practical spec)
--------------------------------------
* Flexible parametrization : any valid (A, S, c, N, D).
* Modularity              : Stations are connectable -> multi-stage / tandem
                            networks (next session) work with zero changes.
* Event-driven            : a Future Event List (FEL) drives the clock.
                            Time advances *to the next event*, never in fixed
                            steps.
* Validation              : closed-form theory for the Markovian cases, plus a
                            TRACE that can be diffed against the GPSS model.

The pseudo-random numbers come from the SAME Linear Congruential Generator that
is validated in `prng_validation.R`, so the Python and R sides are coherent.

Author : <your name>
Course : Statistics / Simulation
"""

from __future__ import annotations

import heapq
import math
import itertools
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Optional, List


# ---------------------------------------------------------------------------
# 1.  Pseudo-Random Number Generator  (same LCG as the R notebook)
#     X(n+1) = (a * X(n) + c) mod m          U = X / m  in [0, 1)
# ---------------------------------------------------------------------------
class LCG:
    """Linear Congruential Generator. Defaults match prng_validation.R."""

    def __init__(self, seed: int = 42, m: int = 2 ** 31,
                 a: int = 1103515245, c: int = 12345):
        self.m, self.a, self.c = m, a, c
        self.state = seed % m

    def random(self) -> float:
        """Uniform U(0,1)."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m


# ---------------------------------------------------------------------------
# 2.  Distribution layer  -> turns a Kendall letter into a sampler
#     M : Markovian / exponential      D : deterministic      G : general
# ---------------------------------------------------------------------------
class Distribution:
    """
    Maps a Kendall code to an inter-event time sampler.

    code  : 'M', 'D' or 'G'
    mean  : mean inter-event time (= 1 / rate)
    rng   : an LCG instance (shared so the whole model is reproducible)
    sampler : for 'G', a callable (rng) -> float. If omitted, 'G' defaults to
              an Erlang-2 with the requested mean (a simple non-exponential,
              non-deterministic example).
    """

    def __init__(self, code: str, mean: float, rng: LCG,
                 sampler: Optional[Callable[[LCG], float]] = None):
        self.code = code.upper()
        self.mean = mean
        self.rng = rng
        self._custom = sampler

    def sample(self) -> float:
        if self.code == "M":                       # exponential
            u = self.rng.random()
            # guard against log(0)
            u = u if u > 0.0 else 1e-12
            return -self.mean * math.log(u)
        if self.code == "D":                       # deterministic
            return self.mean
        if self.code == "G":                       # general
            if self._custom is not None:
                return self._custom(self.rng)
            # default G: Erlang-2 (sum of two exponentials, same total mean)
            u1 = max(self.rng.random(), 1e-12)
            u2 = max(self.rng.random(), 1e-12)
            return -(self.mean / 2.0) * (math.log(u1) + math.log(u2))
        raise ValueError(f"Unknown distribution code: {self.code!r}")


# ---------------------------------------------------------------------------
# 3.  Event-scheduling engine
#     The FEL is a binary heap keyed on (time, insertion-order).
#     insertion-order breaks ties deterministically (FIFO among equal times).
# ---------------------------------------------------------------------------
@dataclass(order=True)
class _ScheduledEvent:
    time: float
    seq: int
    callback: Callable = field(compare=False)
    args: tuple = field(compare=False, default=())


class Simulator:
    """Minimal next-event time-advance engine."""

    def __init__(self):
        self.clock: float = 0.0
        self._fel: List[_ScheduledEvent] = []
        self._counter = itertools.count()
        self.trace_enabled = True
        self._trace: List[str] = []

    def schedule(self, delay: float, callback: Callable, *args) -> None:
        ev = _ScheduledEvent(self.clock + delay, next(self._counter),
                             callback, args)
        heapq.heappush(self._fel, ev)

    def log(self, msg: str) -> None:
        line = f"[t={self.clock:9.4f}] {msg}"
        self._trace.append(line)
        if self.trace_enabled:
            print(line)

    def run(self, until: float) -> None:
        while self._fel and self._fel[0].time <= until:
            ev = heapq.heappop(self._fel)
            self.clock = ev.time          # <-- event-driven time advance
            ev.callback(*ev.args)

    @property
    def trace(self) -> List[str]:
        return self._trace


# ---------------------------------------------------------------------------
# 4.  Entity flowing through the network
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    eid: int
    t_arrival_system: float
    t_arrival_station: float = 0.0   # reset on each station entry (per-node W)
    t_enter_queue: float = 0.0
    t_start_service: float = 0.0


# ---------------------------------------------------------------------------
# 5.  Station  =  one  A/S/c/N/D  node.  Stations are CONNECTABLE.
# ---------------------------------------------------------------------------
class Station:
    """
    A single queueing node.

      service   : Distribution            (the 'S' of Kendall)
      c         : number of parallel servers
      capacity  : 'N' (max in system incl. in service); math.inf if open
      discipline: 'FIFO' | 'LIFO' | 'SIRO'
      router    : callable(entity) -> next Station or None.
                  This single hook is what makes the model modular: returning
                  another Station turns a single queue into a multi-stage net.
    """

    def __init__(self, name: str, sim: Simulator, service: Distribution,
                 c: int = 1, capacity: float = math.inf,
                 discipline: str = "FIFO",
                 router: Optional[Callable[[Entity], "Station"]] = None):
        self.name = name
        self.sim = sim
        self.service = service
        self.c = c
        self.capacity = capacity
        self.discipline = discipline.upper()
        self.router = router

        self.queue: deque[Entity] = deque()
        self.busy_servers = 0

        # ---- statistics accumulators -----------------------------------
        self.arrivals = 0
        self.served = 0
        self.rejected = 0
        self._last_t = 0.0
        self._area_L = 0.0          # time-integral of number in system
        self._area_Lq = 0.0         # time-integral of number in queue
        self._busy_time = 0.0       # server-busy-time integral (for rho)
        self._sum_W = 0.0           # sum of sojourn times
        self._sum_Wq = 0.0          # sum of waiting (queue-only) times

    # -- number currently in the node (queue + in service) ----------------
    @property
    def n_in_system(self) -> int:
        return len(self.queue) + self.busy_servers

    # -- keep the time-weighted integrals up to date ----------------------
    def _accumulate(self) -> None:
        dt = self.sim.clock - self._last_t
        if dt > 0:
            self._area_L += self.n_in_system * dt
            self._area_Lq += len(self.queue) * dt
            self._busy_time += self.busy_servers * dt
        self._last_t = self.sim.clock

    # -- the queue discipline: which waiting entity to serve next ---------
    def _pop_next(self) -> Entity:
        if self.discipline == "FIFO":
            return self.queue.popleft()
        if self.discipline == "LIFO":
            return self.queue.pop()
        if self.discipline == "SIRO":
            # service in random order -> pick a uniform index via the LCG
            idx = int(self.service.rng.random() * len(self.queue))
            idx = min(idx, len(self.queue) - 1)
            ent = self.queue[idx]
            del self.queue[idx]
            return ent
        raise ValueError(f"Unknown discipline {self.discipline!r}")

    # -- EVENT: an entity arrives at this station -------------------------
    def arrive(self, ent: Entity) -> None:
        self._accumulate()
        self.arrivals += 1
        ent.t_arrival_station = self.sim.clock        # local arrival (per-node W)

        if self.n_in_system >= self.capacity:        # blocked / lost
            self.rejected += 1
            self.sim.log(f"REJECT  e{ent.eid:<4d} @ {self.name} "
                         f"(system full N={self.capacity})")
            return

        if self.busy_servers < self.c:               # a server is free
            self.busy_servers += 1
            ent.t_enter_queue = self.sim.clock        # waited 0 in queue
            ent.t_start_service = self.sim.clock
            st = self.service.sample()
            self.sim.schedule(st, self.depart, ent)
            self.sim.log(f"ARRIVE  e{ent.eid:<4d} @ {self.name}  "
                         f"-> SERVICE ({st:.3f})  busy={self.busy_servers}/"
                         f"{self.c} q={len(self.queue)}")
        else:                                         # all servers busy -> wait
            ent.t_enter_queue = self.sim.clock
            self.queue.append(ent)
            self.sim.log(f"ARRIVE  e{ent.eid:<4d} @ {self.name}  "
                         f"-> QUEUE  busy={self.busy_servers}/{self.c} "
                         f"q={len(self.queue)}")

    # -- EVENT: a service completes ---------------------------------------
    def depart(self, ent: Entity) -> None:
        self._accumulate()
        self.busy_servers -= 1
        self.served += 1

        sojourn = self.sim.clock - ent.t_arrival_station
        self._sum_W += sojourn
        self._sum_Wq += ent.t_start_service - ent.t_enter_queue

        self.sim.log(f"DEPART  e{ent.eid:<4d} @ {self.name}  "
                     f"sojourn={sojourn:.3f}  busy={self.busy_servers}/{self.c}"
                     f" q={len(self.queue)}")

        # pull the next waiting entity into the freed server
        if self.queue:
            nxt = self._pop_next()
            self.busy_servers += 1
            nxt.t_start_service = self.sim.clock
            st = self.service.sample()
            self.sim.schedule(st, self.depart, nxt)
            self.sim.log(f"  start  e{nxt.eid:<4d} @ {self.name}  "
                         f"SERVICE ({st:.3f})  q={len(self.queue)}")

        # ---- routing : modular hand-off to the next stage --------------
        if self.router is not None:
            target = self.router(ent)
            if target is not None:
                self.sim.log(f"  route  e{ent.eid:<4d}  {self.name} "
                             f"-> {target.name}")
                target.arrive(ent)

    # -- close the integrals at end of run --------------------------------
    def finalize(self) -> None:
        self._accumulate()

    # -- empirical performance measures -----------------------------------
    def report(self, horizon: float) -> dict:
        L = self._area_L / horizon
        Lq = self._area_Lq / horizon
        rho = self._busy_time / (self.c * horizon)
        W = self._sum_W / self.served if self.served else float("nan")
        Wq = self._sum_Wq / self.served if self.served else float("nan")
        p_loss = self.rejected / self.arrivals if self.arrivals else 0.0
        return dict(L=L, Lq=Lq, rho=rho, W=W, Wq=Wq,
                    arrivals=self.arrivals, served=self.served,
                    rejected=self.rejected, p_loss=p_loss)


# ---------------------------------------------------------------------------
# 6.  Source  =  external arrival stream (the 'A' of Kendall).
#     Kept separate from Station so that downstream stations simply receive
#     their arrivals through routing.  This is the heart of the modularity.
# ---------------------------------------------------------------------------
class Source:
    def __init__(self, name: str, sim: Simulator, arrival: Distribution,
                 target: Station, max_entities: float = math.inf):
        self.name = name
        self.sim = sim
        self.arrival = arrival
        self.target = target
        self.max_entities = max_entities
        self._n = 0

    def start(self) -> None:
        self._schedule_next()

    def _schedule_next(self) -> None:
        if self._n >= self.max_entities:
            return
        self.sim.schedule(self.arrival.sample(), self._generate)

    def _generate(self) -> None:
        self._n += 1
        ent = Entity(eid=self._n, t_arrival_system=self.sim.clock)
        self.target.arrive(ent)
        self._schedule_next()


# ---------------------------------------------------------------------------
# 7.  Closed-form theory  (for VALIDATION of the simulation)
# ---------------------------------------------------------------------------
def theory(A: str, S: str, c: int, lam: float, mu: float,
           N: float = math.inf) -> Optional[dict]:
    """
    Returns analytic L, Lq, W, Wq, rho for the Markovian cases that have a
    closed form.  Returns None when no simple formula applies (then trust the
    GPSS cross-check instead).
    """
    A, S = A.upper(), S.upper()
    rho = lam / (c * mu)

    # ---- M/M/1 (infinite) -------------------------------------------------
    if A == "M" and S == "M" and c == 1 and math.isinf(N):
        if rho >= 1:
            return None
        L = rho / (1 - rho)
        Lq = rho ** 2 / (1 - rho)
        W = 1.0 / (mu - lam)
        Wq = rho / (mu - lam)
        return dict(L=L, Lq=Lq, W=W, Wq=Wq, rho=rho)

    # ---- M/M/1/N (finite capacity) ---------------------------------------
    if A == "M" and S == "M" and c == 1 and not math.isinf(N):
        r = lam / mu
        if abs(r - 1.0) < 1e-9:
            P0 = 1.0 / (N + 1)
            L = N / 2.0
        else:
            P0 = (1 - r) / (1 - r ** (N + 1))
            L = r * (1 - (N + 1) * r ** N + N * r ** (N + 1)) / \
                ((1 - r) * (1 - r ** (N + 1)))
        PN = (r ** N) * P0
        lam_eff = lam * (1 - PN)
        W = L / lam_eff
        Wq = W - 1.0 / mu
        Lq = lam_eff * Wq
        return dict(L=L, Lq=Lq, W=W, Wq=Wq, rho=lam_eff / mu, p_block=PN)

    # ---- M/M/c (infinite) -------------------------------------------------
    if A == "M" and S == "M" and c >= 1 and math.isinf(N):
        if rho >= 1:
            return None
        a = lam / mu
        s = sum(a ** n / math.factorial(n) for n in range(c))
        last = a ** c / (math.factorial(c) * (1 - rho))
        P0 = 1.0 / (s + last)
        Lq = P0 * (a ** c) * rho / (math.factorial(c) * (1 - rho) ** 2)
        Wq = Lq / lam
        W = Wq + 1.0 / mu
        L = lam * W
        return dict(L=L, Lq=Lq, W=W, Wq=Wq, rho=rho)

    return None


# ---------------------------------------------------------------------------
# 8.  Demonstrations
# ---------------------------------------------------------------------------
def banner(txt: str) -> None:
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


def demo_single(A="M", S="M", c=1, N=math.inf, D="FIFO",
                lam=1.0, mu=1.25, horizon=20000.0,
                trace_first_n=25, seed=42):
    """Run a single A/S/c/N/D station and validate it against theory."""
    label = f"{A}/{S}/{c}" + ("" if math.isinf(N) else f"/{int(N)}") + f"  ({D})"
    banner(f"SINGLE STATION   {label}   lambda={lam}  mu={mu}")

    rng = LCG(seed=seed)
    sim = Simulator()
    sim.trace_enabled = True

    service = Distribution(S, mean=1.0 / mu, rng=rng)
    station = Station("Q1", sim, service, c=c, capacity=N, discipline=D)
    arrival = Distribution(A, mean=1.0 / lam, rng=rng)
    source = Source("SRC", sim, arrival, station)

    # ---- TRACE (first events only, full trace kept in sim.trace) --------
    print(f"\n--- TRACE (first {trace_first_n} events) ---")
    _printed = {"n": 0}
    _orig_log = sim.log

    def capped_log(msg):
        if _printed["n"] < trace_first_n:
            _orig_log(msg)
            _printed["n"] += 1
        else:
            sim._trace.append(f"[t={sim.clock:9.4f}] {msg}")
    sim.log = capped_log

    source.start()
    sim.run(until=horizon)
    station.finalize()

    sim.log = _orig_log
    print(f"... ({len(sim.trace)} events total, trace truncated) ...")

    # ---- results vs theory ----------------------------------------------
    emp = station.report(horizon)
    th = theory(A, S, c, lam, mu, N)

    print("\n--- RESULTS  (simulation  vs  theory) ---")
    print(f"{'metric':<8}{'simulated':>14}{'theoretical':>16}{'rel.err %':>12}")
    for k in ("rho", "L", "Lq", "W", "Wq"):
        sv = emp.get(k, float('nan'))
        tv = th.get(k) if th else None
        if tv is None:
            print(f"{k:<8}{sv:>14.4f}{'   (no formula)':>16}{'':>12}")
        else:
            err = 100 * abs(sv - tv) / tv if tv else 0.0
            print(f"{k:<8}{sv:>14.4f}{tv:>16.4f}{err:>12.2f}")
    print(f"\narrivals={emp['arrivals']}  served={emp['served']}  "
          f"rejected={emp['rejected']}  P(loss)={emp['p_loss']:.4f}")
    return emp, th


def demo_tandem(lam=1.0, mu1=1.5, mu2=1.25, horizon=20000.0, seed=7):
    """
    Two stations in series (M/M/1 -> M/M/1).  Demonstrates MODULARITY:
    station 1 routes every finished entity into station 2 with one callback.
    """
    banner(f"TANDEM NETWORK   M/M/1 -> M/M/1   "
           f"lambda={lam}  mu1={mu1}  mu2={mu2}")
    rng = LCG(seed=seed)
    sim = Simulator()
    sim.trace_enabled = False        # network run is long; keep trace silent

    s2 = Station("Q2", sim, Distribution("M", 1.0 / mu2, rng), c=1)
    s1 = Station("Q1", sim, Distribution("M", 1.0 / mu1, rng), c=1,
                 router=lambda ent: s2)          # <-- connect Q1 to Q2
    src = Source("SRC", sim, Distribution("M", 1.0 / lam, rng), s1)

    src.start()
    sim.run(until=horizon)
    s1.finalize(); s2.finalize()

    for st, mu in ((s1, mu1), (s2, mu2)):
        emp = st.report(horizon)
        th = theory("M", "M", 1, lam, mu)
        print(f"\n[{st.name}]  mu={mu}")
        print(f"  rho  sim={emp['rho']:.4f}  th={th['rho']:.4f}")
        print(f"  L    sim={emp['L']:.4f}  th={th['L']:.4f}")
        print(f"  W    sim={emp['W']:.4f}  th={th['W']:.4f}")


if __name__ == "__main__":
    # Reference scenario, shared with the GPSS model:
    #   lambda = 1.0   mu = 1.25   ->  rho = 0.8
    #   theory: L=4, Lq=3.2, W=4, Wq=3.2
    demo_single(A="M", S="M", c=1, lam=1.0, mu=1.25)

    # finite capacity, multi-server, and a non-FIFO discipline
    demo_single(A="M", S="M", c=1, N=5, lam=1.0, mu=1.25, trace_first_n=0)
    demo_single(A="M", S="M", c=2, lam=1.5, mu=1.0, trace_first_n=0)
    demo_single(A="M", S="M", c=1, D="LIFO", lam=1.0, mu=1.25, trace_first_n=0)

    # modular multi-stage network
    demo_tandem()
