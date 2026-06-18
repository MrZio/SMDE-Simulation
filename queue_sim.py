"""
queue_sim3_extended.py
======================
Discrete-event simulator esteso da queue_sim3.py.

Aggiunge rispetto alla versione precedente:
  - Supporto a N nodi arbitrari in topologia qualsiasi (tandem, mesh, fork...)
  - Scenario 1: validazione M/M/1 singola stazione (confronto teorico Erlang)
  - Scenario 2: tandem puro M/M/1 -> M/M/1  (confronto con Burke e GPSS)
  - Scenario 3: rete mesh 3 stazioni (modello PR-01, pronto per DoE)
  - print_comparison() con valori teorici M/M/c
  - print_replications() per tabelle multi-run (identica a sim3)
  - Seed per riproducibilita' (allineato a GPSS RMULT 123459)
"""

from pydantic import BaseModel, ConfigDict
from enum import Enum
import heapq, random, math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DistributionArrivalTimes(Enum):
    M = 'Markovian'
    D = 'Deterministic'
    G = 'General'

class DistributionServiceTimes(Enum):
    M = 'Exponential'
    D = 'Deterministic'
    G = 'General'

class QueueDiscipline(Enum):
    FIFO = 'First In First Out'
    LIFO = 'Last In First Out'
    SIRO = 'Service In Random Order'

class EventType(Enum):
    ARRIVAL   = 'Arrival'
    DEPARTURE = 'Departure'


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    id: int
    path: list[int] = []
    path_index: int = 0
    arrival_times:   list[float] = []
    service_starts:  list[float] = []
    departure_times: list[float] = []


class QueueNode(BaseModel):
    id:   int
    name: str

    arrival_rate:          float                            # lambda (0 = solo routed)
    service_rate:          float                            # mu
    arrival_distribution:  DistributionArrivalTimes         # A
    service_distribution:  DistributionServiceTimes         # S
    num_servers:           int                              # c
    sys_capacity:          int | None                       # N  (None = inf)
    queue_discipline:      QueueDiscipline                  # D

    # topologia: lista degli id nodo verso cui questa stazione instrada
    # [] = nodo terminale (le entita' escono dal sistema dopo il servizio)
    reachable: list[int] = []

    # stato run-time
    waiting_queue: list[Entity] = []
    busy_servers:  int = 0

    # area-under-curve
    area_queue_length:  float = 0.0
    area_system_length: float = 0.0
    last_event_time:    float = 0.0

    # contatori
    total_arrivals: int = 0
    total_lost:     int = 0
    total_served:   int = 0


@dataclass(order=True)
class Event:
    time:       float
    event_type: EventType = field(compare=False)
    node:       QueueNode = field(compare=False)
    entity:     Entity    = field(compare=False, default=None)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

class Simulation(BaseModel):
    clock:          float      = 0.0
    event_list:     list[Event] = []
    nodes:          list[QueueNode] = []
    entity_served:  list[Entity]    = []
    entity_counter: int             = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- helpers ------------------------------------------------------------

    def _node_by_id(self, node_id: int) -> QueueNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"No node with id {node_id}")

    def _sample_path(self, entry_node_id: int) -> list[int]:
        """
        Routing deterministico tandem:
        segue la catena reachable[0] -> reachable[0] -> ... fino al nodo terminale.
        Se reachable e' vuoto, il nodo e' terminale e il percorso si ferma.
        """
        path = [entry_node_id]
        current = self._node_by_id(entry_node_id)
        while current.reachable:
            next_id = current.reachable[0]
            path.append(next_id)
            current = self._node_by_id(next_id)
        return path

    # --- scheduling ---------------------------------------------------------

    def schedule(self, event: Event):
        heapq.heappush(self.event_list, event)

    def schedule_arrival(self, node: QueueNode):
        t    = self.clock + self._arrival_time(node)
        path = self._sample_path(node.id)
        ent  = Entity(id=self.entity_counter, path=path,
                      path_index=0, arrival_times=[t])
        self.entity_counter += 1
        self.schedule(Event(time=t, event_type=EventType.ARRIVAL,
                            node=node, entity=ent))

    # --- main loop ----------------------------------------------------------

    def run(self, until: float):
        for node in self.nodes:
            if node.arrival_rate > 0.0:
                self.schedule_arrival(node)

        while self.event_list:
            ev = heapq.heappop(self.event_list)
            if ev.time > until:
                break
            self._update_area_stats(ev.node, ev.time)
            self.clock = ev.time
            self._process_event(ev)

    # --- event handlers -----------------------------------------------------

    def _process_event(self, event: Event):
        if event.event_type == EventType.ARRIVAL:
            self._handle_arrival(event)
        else:
            self._handle_departure(event)

    def _update_area_stats(self, node: QueueNode, now: float):
        dt = now - node.last_event_time
        if dt > 0:
            node.area_queue_length  += len(node.waiting_queue) * dt
            node.area_system_length += (len(node.waiting_queue) + node.busy_servers) * dt
        node.last_event_time = now

    def _handle_arrival(self, event: Event):
        node   = event.node
        entity = event.entity
        node.total_arrivals += 1

        in_system = len(node.waiting_queue) + node.busy_servers
        if node.sys_capacity is not None and in_system >= node.sys_capacity:
            node.total_lost += 1                        # entita' persa (blocking)
        elif node.busy_servers < node.num_servers:      # server libero
            node.busy_servers += 1
            entity.service_starts.append(self.clock)
            self.schedule(Event(
                time       = self.clock + self._service_time(node),
                event_type = EventType.DEPARTURE,
                node       = node,
                entity     = entity,
            ))
        else:
            node.waiting_queue.append(entity)           # entra in coda

        # re-arma il processo Poisson solo per arrivi esterni
        if entity.path_index == 0 and entity.path and entity.path[0] == node.id:
            self.schedule_arrival(node)

    def _handle_departure(self, event: Event):
        node   = event.node
        entity = event.entity
        entity.departure_times.append(self.clock)
        node.total_served += 1

        if node.waiting_queue:
            next_ent = self._pick_next(node)
            next_ent.service_starts.append(self.clock)
            self.schedule(Event(
                time       = self.clock + self._service_time(node),
                event_type = EventType.DEPARTURE,
                node       = node,
                entity     = next_ent,
            ))
        else:
            node.busy_servers -= 1

        entity.path_index += 1
        if entity.path_index < len(entity.path):
            next_node = self._node_by_id(entity.path[entity.path_index])
            self._route_to(entity, next_node)
        else:
            self.entity_served.append(entity)

    def _route_to(self, entity: Entity, next_node: QueueNode, delay: float = 0.0):
        t = self.clock + delay
        entity.arrival_times.append(t)
        self.schedule(Event(time=t, event_type=EventType.ARRIVAL,
                            node=next_node, entity=entity))

    # --- statistics ---------------------------------------------------------

    def final_statistics(self) -> dict:
        for node in self.nodes:
            self._update_area_stats(node, self.clock)

        results = {}
        for node in self.nodes:
            Lq  = node.area_queue_length  / self.clock if self.clock > 0 else 0.0
            L   = node.area_system_length / self.clock if self.clock > 0 else 0.0
            rho = (node.area_system_length - node.area_queue_length) \
                  / (self.clock * node.num_servers) if self.clock > 0 else 0.0

            wq_node, w_node = [], []
            for e in self.entity_served:
                if node.id in e.path:
                    i = e.path.index(node.id)
                    if i < len(e.arrival_times) and i < len(e.departure_times) \
                            and i < len(e.service_starts):
                        wq_node.append(e.service_starts[i]  - e.arrival_times[i])
                        w_node.append( e.departure_times[i] - e.arrival_times[i])

            results[node.name] = {
                "Lq": Lq, "L": L, "rho": rho,
                "sys_capacity":   node.sys_capacity,
                "num_servers":    node.num_servers,
                "total_arrivals": node.total_arrivals,
                "total_served":   node.total_served,
                "total_lost":     node.total_lost,
                "Wq": sum(wq_node)/len(wq_node) if wq_node else 0.0,
                "W":  sum(w_node) /len(w_node)  if w_node  else 0.0,
            }

        wq_all, w_all = [], []
        for e in self.entity_served:
            if e.arrival_times and e.departure_times and e.service_starts:
                hops = min(len(e.arrival_times), len(e.service_starts))
                wq_all.append(sum(e.service_starts[i] - e.arrival_times[i]
                                  for i in range(hops)))
                w_all.append(e.departure_times[-1] - e.arrival_times[0])

        results["overall"] = {
            "n_entities_served": len(self.entity_served),
            "Wq": sum(wq_all)/len(wq_all) if wq_all else 0.0,
            "W":  sum(w_all) /len(w_all)  if w_all  else 0.0,
            "simulation_time": self.clock,
        }
        return results

    # --- sampling -----------------------------------------------------------

    def _pick_next(self, node: QueueNode) -> Entity:
        if node.queue_discipline == QueueDiscipline.FIFO:
            return node.waiting_queue.pop(0)
        elif node.queue_discipline == QueueDiscipline.LIFO:
            return node.waiting_queue.pop(-1)
        else:  # SIRO
            e = random.choice(node.waiting_queue)
            node.waiting_queue.remove(e)
            return e

    def _arrival_time(self, node: QueueNode) -> float:
        if node.arrival_distribution == DistributionArrivalTimes.M:
            return random.expovariate(node.arrival_rate)
        elif node.arrival_distribution == DistributionArrivalTimes.D:
            return 1.0 / node.arrival_rate
        return random.uniform(0.5, 1.5)

    def _service_time(self, node: QueueNode) -> float:
        if node.service_distribution == DistributionServiceTimes.M:
            return random.expovariate(node.service_rate)
        elif node.service_distribution == DistributionServiceTimes.D:
            return 1.0 / node.service_rate
        return random.uniform(0.5, 1.5)


# ---------------------------------------------------------------------------
# Theoretical formulas (M/M/c, M/M/1/N)
# ---------------------------------------------------------------------------

def mm1_theory(lam: float, mu: float) -> dict:
    """Formule M/M/1 in forma chiusa."""
    rho = lam / mu
    if rho >= 1.0:
        return {"rho": rho, "Lq": float("inf"), "L": float("inf"),
                "Wq": float("inf"), "W": float("inf")}
    Lq = rho**2 / (1 - rho)
    Wq = Lq / lam
    W  = Wq + 1.0/mu
    L  = lam * W
    return {"rho": rho, "Lq": Lq, "L": L, "Wq": Wq, "W": W}


def mmc_theory(lam: float, mu: float, c: int) -> dict:
    """Formule M/M/c (Erlang-C) in forma chiusa."""
    rho = lam / (c * mu)
    if rho >= 1.0:
        return {"rho": rho, "Lq": float("inf"), "L": float("inf"),
                "Wq": float("inf"), "W": float("inf")}
    a  = lam / mu
    s  = sum(a**n / math.factorial(n) for n in range(c))
    s += a**c / (math.factorial(c) * (1 - rho))
    P0 = 1.0 / s
    Lq = P0 * a**c * rho / (math.factorial(c) * (1 - rho)**2)
    Wq = Lq / lam
    W  = Wq + 1.0/mu
    L  = lam * W
    return {"rho": rho, "Lq": Lq, "L": L, "Wq": Wq, "W": W}


def mm1N_theory(lam: float, mu: float, N: int) -> dict:
    """Formule M/M/1/N (capacita' finita) in forma chiusa."""
    rho = lam / mu
    if abs(rho - 1.0) < 1e-9:
        P0 = 1.0 / (N + 1)
        Pn = [P0] * (N + 1)
    else:
        P0 = (1 - rho) / (1 - rho**(N+1))
        Pn = [P0 * rho**n for n in range(N+1)]
    PN   = Pn[N]                       # prob di sistema pieno (blocking)
    lam_eff = lam * (1 - PN)           # throughput effettivo
    L    = sum(n * Pn[n] for n in range(N+1))
    Lq   = sum((n-1) * Pn[n] for n in range(1, N+1))
    W    = L  / lam_eff if lam_eff > 0 else float("inf")
    Wq   = Lq / lam_eff if lam_eff > 0 else float("inf")
    return {"rho": lam_eff/mu, "Lq": Lq, "L": L, "Wq": Wq, "W": W,
            "P_block": PN, "lam_eff": lam_eff}


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_comparison(stats: dict, node_name: str,
                     theoretical: dict | None = None, title: str = ""):
    """Tabella sim vs teoria per un singolo nodo."""
    s = stats[node_name]
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    fmt = "{:<22}{:>14}{:>16}{:>14}"
    print(fmt.format("Metric", "Simulated", "Theoretical", "Abs.error"))
    print("-" * 66)

    def row(label, sim_v, theo_v=None, as_int=False):
        sf = str(int(sim_v))  if as_int else f"{sim_v:.4f}"
        tf = str(int(theo_v)) if (as_int and theo_v is not None) \
             else (f"{theo_v:.4f}" if theo_v is not None else "-")
        ef = f"{abs(sim_v-theo_v):.4f}" if theo_v is not None and not as_int else "-"
        print(fmt.format(label, sf, tf, ef))

    t = theoretical or {}
    print(f"\n[Node: {node_name}]")
    row("c (servers)",    s["num_servers"], as_int=True)
    row("N (capacity)",   s["sys_capacity"] if s["sys_capacity"] else 999999, as_int=True)
    row("rho",            s["rho"],  t.get("rho"))
    row("Lq",             s["Lq"],   t.get("Lq"))
    row("L",              s["L"],    t.get("L"))
    row("Wq",             s["Wq"],   t.get("Wq"))
    row("W",              s["W"],    t.get("W"))
    row("Arrivals",       s["total_arrivals"], as_int=True)
    row("Served",         s["total_served"],   as_int=True)
    row("Lost/blocked",   s["total_lost"],     as_int=True)


def print_replications(all_stats: list[dict], node_names: list[str],
                       title: str = ""):
    """Tabella multi-run: righe=run, colonne=metriche, ultima riga=media±std."""
    COLS     = ["Wq",  "W",  "Lq",  "L",  "rho",
                "Arrivals",        "Served",       "Lost"]
    KEYS     = ["Wq",  "W",  "Lq",  "L",  "rho",
                "total_arrivals",  "total_served", "total_lost"]
    INT_KEYS = {"total_arrivals", "total_served", "total_lost"}
    COL_W, RUN_W = 11, 6

    if title:
        print(f"\n{'='*(RUN_W+1+len(COLS)*(COL_W+1))}")
        print(f" {title}")
        print(f"{'='*(RUN_W+1+len(COLS)*(COL_W+1))}")

    for node_name in node_names:
        hdr = f"{'Run':<{RUN_W}} " + " ".join(f"{c:>{COL_W}}" for c in COLS)
        sep = "-" * (RUN_W+1+len(COLS)*(COL_W+1))
        print(f"\n  Node: {node_name}")
        print(f"  {hdr}")
        print(f"  {sep}")

        collected = {k: [] for k in KEYS}
        for run_i, stats in enumerate(all_stats, 1):
            s = stats[node_name]
            vals = []
            for k in KEYS:
                v = s[k]; collected[k].append(v)
                vals.append(f"{int(v):>{COL_W}}" if k in INT_KEYS
                            else f"{v:>{COL_W}.4f}")
            print(f"  {run_i:<{RUN_W}} {' '.join(vals)}")

        print(f"  {sep}")
        n = len(all_stats)
        means = {k: sum(collected[k])/n for k in KEYS}
        stds  = {k: (sum((v-means[k])**2 for v in collected[k])/(n-1))**0.5
                 if n > 1 else 0.0 for k in KEYS}

        mv = [f"{means[k]:>{COL_W}.1f}" if k in INT_KEYS
              else f"{means[k]:>{COL_W}.4f}" for k in KEYS]
        sv = [f"{stds[k]:>{COL_W}.1f}"  if k in INT_KEYS
              else f"{stds[k]:>{COL_W}.4f}"  for k in KEYS]
        print(f"  {'Mean':<{RUN_W}} {' '.join(mv)}")
        print(f"  {'Std':<{RUN_W}} {' '.join(sv)}")


# ---------------------------------------------------------------------------
# Node factory (evita copia/incolla)
# ---------------------------------------------------------------------------

def make_node(id, name, lam, mu, c=1, N=None,
              arr=DistributionArrivalTimes.M,
              svc=DistributionServiceTimes.M,
              disc=QueueDiscipline.FIFO,
              reachable=None) -> QueueNode:
    return QueueNode(
        id=id, name=name,
        arrival_rate=lam, service_rate=mu,
        arrival_distribution=arr, service_distribution=svc,
        num_servers=c, sys_capacity=N,
        queue_discipline=disc,
        reachable=reachable or [],
    )


# ===========================================================================
# SCENARIO 1 – Validazione M/M/1 singola stazione
# ===========================================================================

def run_scenario1(n_runs=10, sim_time=50_000, seed=123459):
    """
    M/M/1: lambda=0.8, mu=1.0  ->  rho_teoria=0.8
    Confronto diretto con formule di Erlang e con gpssNewModel.gps.
    """
    print("\n" + "="*70)
    print(" SCENARIO 1 — Validazione M/M/1 (singola stazione)")
    print("="*70)
    print(f" lambda=0.8  mu=1.0  rho_teoria=0.800  |  {n_runs} run x {sim_time} t.u.")

    random.seed(seed)
    all_stats = []
    for _ in range(n_runs):
        node = make_node(0, "Station_A", lam=0.8, mu=1.0)
        sim  = Simulation(nodes=[node])
        sim.run(until=sim_time)
        all_stats.append(sim.final_statistics())

    # tabella multi-run
    print_replications(all_stats, ["Station_A"],
                       title=f"M/M/1  lambda=0.8 mu=1.0  ({n_runs} run x {sim_time} t.u.)")

    # confronto teorico sull'ultima run
    theory = mm1_theory(0.8, 1.0)
    print_comparison(all_stats[-1], "Station_A", theoretical=theory,
                     title="\nConfronto ultima run vs formula M/M/1")
    print(f"\n  Teorico rho={theory['rho']:.4f}  Lq={theory['Lq']:.4f}"
          f"  Wq={theory['Wq']:.4f}  W={theory['W']:.4f}  L={theory['L']:.4f}")


# ===========================================================================
# SCENARIO 2 – Validazione tandem M/M/1 -> M/M/1
# ===========================================================================

def run_scenario2(n_runs=10, sim_time=50_000, seed=123459):
    """
    Tandem puro: Station_A (M/M/1) -> Station_B (M/M/1)
    Per il teorema di Burke, l'output di M/M/1 stabile e' ancora Poisson(lambda)
    -> Station_B ha gli stessi parametri di Station_A: rho_B = rho_A = 0.8.
    Allineato a tandem_MM1_MM1.gps (RMULT 123459).
    """
    print("\n" + "="*70)
    print(" SCENARIO 2 — Validazione Tandem M/M/1 -> M/M/1")
    print("="*70)
    print(f" lambda=0.8  mu=1.0  rho_teoria=0.8 per entrambi  |  {n_runs} run x {sim_time} t.u.")

    random.seed(seed)
    all_stats = []
    for _ in range(n_runs):
        nA = make_node(0, "Station_A", lam=0.8, mu=1.0, reachable=[1])
        nB = make_node(1, "Station_B", lam=0.0, mu=1.0, reachable=[])
        sim = Simulation(nodes=[nA, nB])
        sim.run(until=sim_time)
        all_stats.append(sim.final_statistics())

    print_replications(all_stats, ["Station_A", "Station_B"],
                       title=f"Tandem M/M/1->M/M/1  ({n_runs} run x {sim_time} t.u.)")

    theory = mm1_theory(0.8, 1.0)
    print("\n  Valori teorici (Burke): entrambe le stazioni identiche")
    print(f"  rho={theory['rho']:.4f}  Lq={theory['Lq']:.4f}"
          f"  Wq={theory['Wq']:.4f}  W={theory['W']:.4f}  L={theory['L']:.4f}")


# ===========================================================================
# SCENARIO 3 – Rete mesh 3 stazioni (modello PR-01)
# ===========================================================================

def run_scenario3(n_runs=10, sim_time=50_000, seed=123459,
                  lam=0.3, mu_A=1.0, mu_B=1.2, mu_C=0.8,
                  cap_B=None, cap_C=None):
    """
    Mesh 3 stazioni: A -> B -> C  (topologia lineare di default)
    Parametri pensati per stabilita': lam=0.3 (basso per evitare instabilita' mesh).

    Puoi chiamare questa funzione con parametri diversi per il DoE:
        run_scenario3(lam=0.3, mu_A=1.0, mu_B=1.2, mu_C=0.8)
        run_scenario3(lam=0.5, mu_A=1.5, mu_B=1.0, mu_C=1.0, cap_B=5)
    """
    print("\n" + "="*70)
    print(" SCENARIO 3 — Rete Mesh 3 Stazioni (A -> B -> C)")
    print("="*70)
    print(f" lam={lam}  mu_A={mu_A}  mu_B={mu_B}  mu_C={mu_C}"
          f"  cap_B={cap_B}  cap_C={cap_C}")
    print(f" rho_A={lam/mu_A:.3f}  rho_B={lam/mu_B:.3f}  rho_C={lam/mu_C:.3f}")
    print(f" {n_runs} run x {sim_time} t.u.")

    random.seed(seed)
    all_stats = []
    for _ in range(n_runs):
        nA = make_node(0, "Station_A", lam=lam,  mu=mu_A, N=None,  reachable=[1])
        nB = make_node(1, "Station_B", lam=0.0,  mu=mu_B, N=cap_B, reachable=[2])
        nC = make_node(2, "Station_C", lam=0.0,  mu=mu_C, N=cap_C, reachable=[])
        sim = Simulation(nodes=[nA, nB, nC])
        sim.run(until=sim_time)
        all_stats.append(sim.final_statistics())

    print_replications(all_stats, ["Station_A", "Station_B", "Station_C"],
                       title=f"Mesh 3-nodi A->B->C  ({n_runs} run x {sim_time} t.u.)")

    return all_stats   # utile per DoE esterno


# ===========================================================================
# SCENARIO 4 – DoE 2^3  (pronto per il report)
# ===========================================================================

def run_doe(n_runs=5, sim_time=20_000, seed=123459):
    """
    Design of Experiments 2^3:
      Fattore 1: lambda  (livello -: 0.3,  livello +: 0.5)
      Fattore 2: mu_B    (livello -: 0.8,  livello +: 1.2)
      Fattore 3: cap_C   (livello -: None, livello +: 5  )

    Risposta: Wq_overall (attesa media nel sistema intero),
              rho_A, rho_B, rho_C, total_lost_C
    """
    print("\n" + "="*70)
    print(" SCENARIO 4 — DoE 2^3")
    print("="*70)
    print(f" Fattori: lambda (-:0.3 / +:0.5) | mu_B (-:0.8 / +:1.2)"
          f" | cap_C (-:inf / +:5)")
    print(f" {n_runs} run x {sim_time} t.u. per combinazione\n")

    levels = {
        "lam":   (0.3, 0.5),
        "mu_B":  (0.8, 1.2),
        "cap_C": (None, 5),
    }
    header = (f"{'Run#':>5}  {'lam':>5}  {'mu_B':>5}  {'cap_C':>6}  "
              f"{'Wq_tot':>8}  {'rho_A':>7}  {'rho_B':>7}  {'rho_C':>7}  {'Lost_C':>7}")
    print(header)
    print("-" * len(header))

    results_doe = []
    run_num = 0
    for lam in levels["lam"]:
        for mu_B in levels["mu_B"]:
            for cap_C in levels["cap_C"]:
                run_num += 1
                random.seed(seed + run_num)
                run_stats = []
                for _ in range(n_runs):
                    nA = make_node(0, "Station_A", lam=lam,  mu=1.0,  reachable=[1])
                    nB = make_node(1, "Station_B", lam=0.0,  mu=mu_B, reachable=[2])
                    nC = make_node(2, "Station_C", lam=0.0,  mu=0.8,
                                   N=cap_C, reachable=[])
                    sim = Simulation(nodes=[nA, nB, nC])
                    sim.run(until=sim_time)
                    run_stats.append(sim.final_statistics())

                Wq_tot = sum(s["overall"]["Wq"]      for s in run_stats) / n_runs
                rho_A  = sum(s["Station_A"]["rho"]   for s in run_stats) / n_runs
                rho_B  = sum(s["Station_B"]["rho"]   for s in run_stats) / n_runs
                rho_C  = sum(s["Station_C"]["rho"]   for s in run_stats) / n_runs
                lost_C = sum(s["Station_C"]["total_lost"] for s in run_stats) / n_runs
                cap_str = str(cap_C) if cap_C else "inf"
                print(f"  {run_num:>3}  {lam:>5.1f}  {mu_B:>5.1f}  {cap_str:>6}  "
                      f"{Wq_tot:>8.4f}  {rho_A:>7.4f}  {rho_B:>7.4f}"
                      f"  {rho_C:>7.4f}  {lost_C:>7.1f}")
                results_doe.append({
                    "run": run_num, "lam": lam, "mu_B": mu_B, "cap_C": cap_C,
                    "Wq_tot": Wq_tot, "rho_A": rho_A, "rho_B": rho_B,
                    "rho_C": rho_C, "lost_C": lost_C,
                })

    print()
    return results_doe

def run_multiseed_validation():
    seeds = [111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 999999, 123457]
    sim_time = 50000
    
    print("\n=== PYTHON VALIDATION: M/M/1 ===")
    all_stats_mm1 = []
    for s in seeds:
        random.seed(s)
        node = make_node(0, "Station_A", lam=0.8, mu=1.0)
        sim = Simulation(nodes=[node])
        sim.run(until=sim_time)
        all_stats_mm1.append(sim.final_statistics())
    print_replications(all_stats_mm1, ["Station_A"], title="M/M/1 (10 seeds)")

    print("\n=== PYTHON VALIDATION: TANDEM ===")
    all_stats_tandem = []
    for s in seeds:
        random.seed(s)
        nA = make_node(0, "Station_A", lam=0.8, mu=1.0, reachable=[1])
        nB = make_node(1, "Station_B", lam=0.0, mu=1.0, reachable=[])
        sim = Simulation(nodes=[nA, nB])
        sim.run(until=sim_time)
        all_stats_tandem.append(sim.final_statistics())
    print_replications(all_stats_tandem, ["Station_A", "Station_B"], title="Tandem (10 seeds)")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    run_multiseed_validation()
    
    # Scenario 1: M/M/1 singola stazione
    run_scenario1(n_runs=10, sim_time=50_000)

    # Scenario 2: tandem M/M/1 -> M/M/1
    run_scenario2(n_runs=10, sim_time=50_000)

    # Scenario 3: mesh 3 nodi (parametri stabili)
    run_scenario3(n_runs=10, sim_time=50_000,
                  lam=0.3, mu_A=1.0, mu_B=1.2, mu_C=0.8)
