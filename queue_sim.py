"""
queue_sim_ed.py
===============
Modello di simulazione a eventi discreti per un Pronto Soccorso (PS).
Basato su queue_sim3_extended.py.

Motore originale (classi Entity, QueueNode, Event, Simulation) NON modificato.
Il routing probabilistico e' implementato tramite:
  1. Campo opzionale  routing_weights: list[float]  aggiunto a QueueNode
     (lista di pesi parallela a reachable; None = routing deterministico tandem)
  2. Sottoclasse      SimulationED(Simulation)
     che sovrascrive solo _sample_path per gestire il fork probabilistico
     al nodo Triage, lasciando invariato tutto il resto del motore.

Topologia ED (7 nodi):

  [0] Segreteria      M/M/2/50    FIFO  <- unico ingresso esterno (lam)
        |
  [1] Triage          M/M/c_t/50  FIFO  <- c_t = num_triage_nurses (DoE)
        |
      fork PROBABILISTICO (routing_weights sul nodo Triage):
        |-- p=0.40 --> [2] Corsia_Bianca  M/D/1/inf  FIFO
        |-- p=0.20 --> [3] Corsia_Gialla  M/D/1/5    FIFO
        |-- p=0.30 --> [4] Corsia_Verde   M/D/1/5    FIFO
        |-- p=0.10 --> [5] Corsia_Rossa   M/D/3/5    FIFO
                             |
                      [6] Hospitalizzazione  M/M/3/cap_h  FIFO  (cap_h DoE)

Parametri fissi (da specifica clinica):
  - Segreteria:       mu=2.0/min  (30 s/pz),   c=2,  N=50
  - Triage:           mu=1.0/min  (1 min/pz),   c=num_triage_nurses, N=50
  - Corsia Bianca:    mu=0.5/min  (2 min/pz),   c=1,  N=inf, svc=D
  - Corsia Gialla:    mu=0.25/min (4 min/pz),   c=1,  N=5,   svc=D
  - Corsia Verde:     mu=0.25/min (4 min/pz),   c=1,  N=5,   svc=D
  - Corsia Rossa:     mu=0.1/min  (10 min/pz),  c=3,  N=5,   svc=D
  - Hospitalizzazione:mu=0.033/min(30 min/pz),  c=3,  N=cap_h, svc=M

Probabilita' di assegnazione corsia dal triage (configurabili):
  Bianca=0.40  Gialla=0.20  Verde=0.30  Rossa=0.10

Parametri DoE (variabili):
  - lam              : arrival rate (pazienti/min)  es. [4/60, 6/60, 8/60]
  - num_triage_nurses: infermieri al triage          es. [1, 2, 3]
  - cap_h            : capacita' hospitalizzazione   es. [5, 10, None]

Scenari disponibili:
  - run_scenario1/2/3  : invariati (validazione modello originale)
  - run_ed_scenario    : singola run PS con routing probabilistico
  - run_ed_doe         : griglia multi-scenario (DoE PS)
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

    # pesi per routing probabilistico (parallelo a reachable).
    # None  -> routing deterministico: viene usato sempre reachable[0]
    #          (comportamento originale, invariato per tutti i nodi tranne Triage)
    # lista -> random.choices(reachable, weights=routing_weights) al momento
    #          del campionamento del path per ogni entita'.
    routing_weights: list[float] | None = None

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
# SimulationED — sottoclasse con routing probabilistico al Triage
# ---------------------------------------------------------------------------

class SimulationED(Simulation):
    """
    Estende Simulation sovrascrivendo _sample_path per supportare il fork
    probabilistico al nodo Triage.

    Logica di _sample_path:
      - Per ogni nodo del percorso, se il nodo ha routing_weights definiti,
        il successore viene scelto con random.choices(reachable, weights).
      - Se routing_weights e' None (tutti i nodi originali e quelli tandem),
        il comportamento e' identico all'originale: si segue reachable[0].
      - Il path completo viene campionato una sola volta per entita', al
        momento dell'arrivo alla Segreteria, esattamente come nel motore base.

    Nessun altro metodo e' modificato.
    """

    def _sample_path(self, entry_node_id: int) -> list[int]:
        """
        Campiona il percorso completo dell'entita' a partire da entry_node_id.

        Per i nodi con routing_weights definiti (es. Triage) sceglie il
        successore probabilisticamente; per tutti gli altri usa reachable[0]
        (routing deterministico tandem, comportamento originale).
        """
        path    = [entry_node_id]
        current = self._node_by_id(entry_node_id)

        while current.reachable:
            if current.routing_weights is not None:
                # fork probabilistico: campiona UN successore secondo i pesi
                next_id = random.choices(
                    current.reachable,
                    weights=current.routing_weights,
                    k=1,
                )[0]
            else:
                # routing deterministico originale: primo elemento di reachable
                next_id = current.reachable[0]

            path.append(next_id)
            current = self._node_by_id(next_id)

        return path


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
# PRONTO SOCCORSO — topologia ED
# ===========================================================================
#
# Struttura a 7 nodi (IDs fissi):
#
#   0  Segreteria      M/M/2/50     FIFO   <- unico punto di ingresso esterno
#   1  Triage          M/M/c_t/50   FIFO   <- c_t = num_triage_nurses (DoE)
#   2  Corsia_Bianca   M/D/1/inf    FIFO   <- codice bianco (non urgente)
#   3  Corsia_Gialla   M/D/1/5      FIFO   <- codice giallo (urgente)
#   4  Corsia_Verde    M/D/1/5      FIFO   <- codice verde (poco urgente)
#   5  Corsia_Rossa    M/D/3/5      FIFO   <- codice rosso (emergenza)
#   6  Hospitalizzazione M/M/3/cap_h FIFO  <- reparto degenza (DoE)
#
# Percorso di default (routing deterministico tandem, reachable[0]):
#   Segreteria -> Triage -> Corsia_Bianca -> Hospitalizzazione
#
# Ogni paziente riceve la corsia in modo probabilistico al momento del triage.
# In questo modo ogni scenario DoE testa un percorso specifico mantenendo
# la stessa logica di routing tandem del motore originale.
#
# Parametri fissi (da specifica clinica):
#   Segreteria  : mu=2.0  (tempo medio servizio 0.5 min = 30 s)
#   Triage      : mu=1.0  (tempo medio servizio 1 min)
#   C. Bianca   : mu=0.5  (2 min), svc=D, c=1, N=inf
#   C. Gialla   : mu=0.25 (4 min), svc=D, c=1, N=5
#   C. Verde    : mu=0.25 (4 min), svc=D, c=1, N=5
#   C. Rossa    : mu=0.1  (10 min),svc=D, c=3, N=5
#   Hospital.   : mu=0.033(30 min),svc=M, c=3, N=cap_h
# ---------------------------------------------------------------------------

# Nomi e ID dei nodi — usati anche come chiavi nei risultati
NODE_SEGRETERIA   = 0
NODE_TRIAGE       = 1
NODE_BIANCA       = 2
NODE_GIALLA       = 3
NODE_VERDE        = 4
NODE_ROSSA        = 5
NODE_HOSPITAL     = 6

ALL_NODE_NAMES = [
    "Segreteria",
    "Triage",
    "Corsia_Bianca",
    "Corsia_Gialla",
    "Corsia_Verde",
    "Corsia_Rossa",
    "Hospitalizzazione",
]


def _build_ed_nodes(
    lam: float,
    num_triage_nurses: int,
    cap_h: int | None,
    ward_probs: dict[int, float] | None = None,
) -> list:
    """
    Costruisce la lista di QueueNode per la topologia PS con routing
    probabilistico dal Triage alle corsie.

    Parametri
    ---------
    lam               : arrival rate esterno (pazienti/min)
    num_triage_nurses : numero di infermieri al triage (c del nodo Triage)
    cap_h             : capacita' del reparto Hospitalizzazione (None = inf)
    ward_probs        : dizionario {node_id: probabilita'} per le corsie.
                        Default:
                          NODE_BIANCA=0.40, NODE_GIALLA=0.20,
                          NODE_VERDE=0.30,  NODE_ROSSA=0.10

    Topologia risultante:
        Segreteria [0] -> Triage [1] --(prob)--> Corsia_X [2/3/4/5]
                                                      |
                                           Hospitalizzazione [6]
    """
    if ward_probs is None:
        ward_probs = {
            NODE_BIANCA: 0.40,
            NODE_GIALLA: 0.20,
            NODE_VERDE:  0.30,
            NODE_ROSSA:  0.10,
        }

    # Ordine fisso delle corsie nel reachable del Triage
    ward_ids    = [NODE_BIANCA, NODE_GIALLA, NODE_VERDE, NODE_ROSSA]
    ward_weights = [ward_probs.get(wid, 0.0) for wid in ward_ids]

    # Normalizzazione difensiva (somma deve essere 1.0)
    total_w = sum(ward_weights)
    if abs(total_w - 1.0) > 1e-9:
        ward_weights = [w / total_w for w in ward_weights]

    # --- nodo 0 : Segreteria  M/M/2/50 FIFO ---
    n_segreteria = make_node(
        id=NODE_SEGRETERIA,
        name="Segreteria",
        lam=lam,
        mu=2.0,           # 2 pz/min  ->  30 s medi per accettazione
        c=2,              # 2 segretari
        N=50,
        arr=DistributionArrivalTimes.M,
        svc=DistributionServiceTimes.D,
        disc=QueueDiscipline.FIFO,
        reachable=[NODE_TRIAGE],
    )

    # --- nodo 1 : Triage  M/M/c_t/50 FIFO  + routing probabilistico ---
    n_triage = make_node(
        id=NODE_TRIAGE,
        name="Triage",
        lam=0.0,
        mu=1.0,               # 1 pz/min  ->  1 min medio per triage
        c=num_triage_nurses,
        N=50,
        arr=DistributionArrivalTimes.M,
        svc=DistributionServiceTimes.M,
        disc=QueueDiscipline.FIFO,
        reachable=ward_ids,          # tutte e 4 le corsie
    )
    # Assegna i pesi: SimulationED li usa in _sample_path
    n_triage.routing_weights = ward_weights

    # --- nodo 2 : Corsia Bianca  M/D/1/inf FIFO ---
    n_bianca = make_node(
        id=NODE_BIANCA,
        name="Corsia_Bianca",
        lam=0.0,
        mu=0.5,           # 2 min medi
        c=1,
        N=None,
        arr=DistributionArrivalTimes.M,
        svc=DistributionServiceTimes.M,
        disc=QueueDiscipline.FIFO,
        reachable=[NODE_HOSPITAL],
    )

    # --- nodo 3 : Corsia Gialla  M/D/1/5 FIFO ---
    n_gialla = make_node(
        id=NODE_GIALLA,
        name="Corsia_Gialla",
        lam=0.0,
        mu=0.25,          # 4 min medi
        c=1,
        N=5,
        arr=DistributionArrivalTimes.M,
        svc=DistributionServiceTimes.M,
        disc=QueueDiscipline.FIFO,
        reachable=[NODE_HOSPITAL],
    )

    # --- nodo 4 : Corsia Verde  M/D/1/5 FIFO ---
    n_verde = make_node(
        id=NODE_VERDE,
        name="Corsia_Verde",
        lam=0.0,
        mu=0.25,
        c=1,
        N=5,
        arr=DistributionArrivalTimes.M,
        svc=DistributionServiceTimes.M,
        disc=QueueDiscipline.FIFO,
        reachable=[NODE_HOSPITAL],
    )

    # --- nodo 5 : Corsia Rossa  M/D/3/5 FIFO ---
    n_rossa = make_node(
        id=NODE_ROSSA,
        name="Corsia_Rossa",
        lam=0.0,
        mu=0.1,           # 10 min medi  (emergenza)
        c=3,
        N=5,
        arr=DistributionArrivalTimes.M,
        svc=DistributionServiceTimes.M,
        disc=QueueDiscipline.FIFO,
        reachable=[NODE_HOSPITAL],
    )

    # --- nodo 6 : Hospitalizzazione  M/M/3/cap_h FIFO ---
    n_hospital = make_node(
        id=NODE_HOSPITAL,
        name="Hospitalizzazione",
        lam=0.0,
        mu=0.033,         # ~30 min medi di degenza
        c=3,
        N=cap_h,
        arr=DistributionArrivalTimes.M,
        svc=DistributionServiceTimes.D,
        disc=QueueDiscipline.FIFO,
        reachable=[],     # nodo terminale
    )

    return [
        n_segreteria,
        n_triage,
        n_bianca,
        n_gialla,
        n_verde,
        n_rossa,
        n_hospital,
    ]


# ===========================================================================
# SCENARIO ED — singola run del Pronto Soccorso
# ===========================================================================

def run_ed_scenario(
    lam: float             = 4 / 60,
    num_triage_nurses: int = 2,
    cap_h: int | None      = None,
    ward_probs: dict | None = None,
    n_runs: int            = 10,
    sim_time: float        = 50_000,
    seed: int              = 123459,
) -> list[dict]:
    """
    Esegue n_runs repliche del modello PS con routing probabilistico
    dal Triage alle corsie e stampa la tabella per tutti i nodi.

    Parametri
    ---------
    lam               : arrival rate (pazienti/min). Default = 4/ora.
    num_triage_nurses : infermieri al triage.
    cap_h             : capacita' reparto Hospitalizzazione (None = inf).
    ward_probs        : dizionario {node_id: prob} per le corsie.
                        Default: Bianca=0.40, Gialla=0.20, Verde=0.30, Rossa=0.10
    n_runs            : numero di repliche.
    sim_time          : durata simulazione (minuti simulati).
    seed              : seed iniziale.

    Return
    ------
    Lista di dizionari statistici (uno per replica).
    """
    if ward_probs is None:
        ward_probs = {
            NODE_BIANCA: 0.40,
            NODE_GIALLA: 0.20,
            NODE_VERDE:  0.30,
            NODE_ROSSA:  0.10,
        }
    cap_str = str(cap_h) if cap_h is not None else "inf"

    print("\n" + "=" * 72)
    print(" SCENARIO PS — Pronto Soccorso (routing probabilistico)")
    print("=" * 72)
    print(f"  lam={lam:.4f} paz/min ({lam*60:.1f} paz/ora)  |  "
          f"triage_nurses={num_triage_nurses}  |  cap_hospital={cap_str}")
    print(f"  Routing triage -> corsie:")
    for nid, p in ward_probs.items():
        print(f"    {ALL_NODE_NAMES[nid]:<20} p={p:.2f}")
    print(f"  {n_runs} repliche x {sim_time:.0f} min simulati")

    random.seed(seed)
    all_stats = []
    for _ in range(n_runs):
        nodes = _build_ed_nodes(lam, num_triage_nurses, cap_h, ward_probs)
        sim   = SimulationED(nodes=nodes)
        sim.run(until=sim_time)
        all_stats.append(sim.final_statistics())

    print_replications(
        all_stats,
        ALL_NODE_NAMES,
        title=f"PS  lam={lam:.4f}  nurses={num_triage_nurses}"
              f"  cap_h={cap_str}  ({n_runs} run x {sim_time:.0f} min)",
    )
    return all_stats


# ===========================================================================
# SCENARIO ED — DoE multi-scenario (griglia di combinazioni)
# ===========================================================================

def run_ed_doe(
    arrival_rates:      list[float]      = None,
    triage_nurses_list: list[int]        = None,
    cap_h_list:         list[int | None] = None,
    ward_probs: dict | None              = None,
    n_runs: int         = 10,
    sim_time: float     = 50_000,
    seed: int           = 123459,
) -> list[dict]:
    """
    Griglia DoE per il Pronto Soccorso con routing probabilistico.
    Ogni combinazione (lam, num_triage_nurses, cap_h) e' un singolo scenario;
    per ogni scenario vengono eseguite n_runs repliche con SimulationED.

    Parametri DoE (liste di livelli)
    ---------------------------------
    arrival_rates      : es. [4/60, 6/60, 8/60]   (pazienti/min)
    triage_nurses_list : es. [1, 2, 3]
    cap_h_list         : es. [5, 10, None]
    ward_probs         : probabilita' fisse per le corsie (invarianti nel DoE).
                         Default: Bianca=0.40, Gialla=0.20, Verde=0.30, Rossa=0.10

    Risposta per ogni combinazione (media sulle n_runs repliche):
      Wq_overall, W_overall,
      rho e Lost per: Segreteria, Triage, ogni corsia, Hospitalizzazione

    Return
    ------
    Lista di dizionari risultato — uno per combinazione DoE.
    """
    if arrival_rates is None:
        arrival_rates = [4 / 60, 6 / 60, 8 / 60]
    if triage_nurses_list is None:
        triage_nurses_list = [1, 2, 3]
    if cap_h_list is None:
        cap_h_list = [5, 10, None]
    if ward_probs is None:
        ward_probs = {
            NODE_BIANCA: 0.40,
            NODE_GIALLA: 0.20,
            NODE_VERDE:  0.30,
            NODE_ROSSA:  0.10,
        }

    print("\n" + "=" * 72)
    print(" DoE PS — Pronto Soccorso (routing probabilistico)")
    print("=" * 72)
    print(f"  Routing triage -> corsie:")
    for nid, p in ward_probs.items():
        print(f"    {ALL_NODE_NAMES[nid]:<20} p={p:.2f}")
    print(f"  Fattore A — lam (paz/min)      : {[f'{r*60:.1f}/h' for r in arrival_rates]}")
    print(f"  Fattore B — triage_nurses      : {triage_nurses_list}")
    print(f"  Fattore C — cap_hospital       : {cap_h_list}")
    print(f"  Repliche x combinazione        : {n_runs}")
    print(f"  Tempo simulazione              : {sim_time:.0f} min")
    total_combos = len(arrival_rates) * len(triage_nurses_list) * len(cap_h_list)
    print(f"  Totale combinazioni            : {total_combos}")
    print(f"  Totale run simulazione         : {total_combos * n_runs}")

    # Intestazione tabella — una colonna rho e Lost per ogni corsia + hospital
    hdr = (
        f"{'#':>3}  {'lam/h':>6}  {'nurses':>6}  {'cap_h':>6}  "
        f"{'Wq_tot':>8}  {'W_tot':>8}  "
        f"{'rho_Seg':>8}  {'rho_Tri':>8}  "
        f"{'rho_Bia':>8}  {'rho_Gia':>8}  {'rho_Ver':>8}  {'rho_Ros':>8}  "
        f"{'rho_Hos':>8}  "
        f"{'Lost_Bia':>9}  {'Lost_Gia':>9}  {'Lost_Ver':>9}  {'Lost_Ros':>9}  "
        f"{'Lost_Hos':>9}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))

    results_doe = []
    combo_num   = 0

    for lam in arrival_rates:
        for num_triage_nurses in triage_nurses_list:
            for cap_h in cap_h_list:
                combo_num += 1
                random.seed(seed + combo_num)

                run_stats = []
                for _ in range(n_runs):
                    nodes = _build_ed_nodes(lam, num_triage_nurses,
                                            cap_h, ward_probs)
                    sim = SimulationED(nodes=nodes)
                    sim.run(until=sim_time)
                    run_stats.append(sim.final_statistics())

                # Media sulle repliche per una singola metrica
                def avg(node_name: str, metric: str) -> float:
                    vals = [s[node_name][metric] for s in run_stats
                            if node_name in s]
                    return sum(vals) / len(vals) if vals else 0.0

                Wq_tot      = avg("overall",          "Wq")
                W_tot       = avg("overall",          "W")
                rho_seg     = avg("Segreteria",       "rho")
                rho_tri     = avg("Triage",           "rho")
                rho_bia     = avg("Corsia_Bianca",    "rho")
                rho_gia     = avg("Corsia_Gialla",    "rho")
                rho_ver     = avg("Corsia_Verde",     "rho")
                rho_ros     = avg("Corsia_Rossa",     "rho")
                rho_hos     = avg("Hospitalizzazione","rho")
                lost_bia    = avg("Corsia_Bianca",    "total_lost")
                lost_gia    = avg("Corsia_Gialla",    "total_lost")
                lost_ver    = avg("Corsia_Verde",     "total_lost")
                lost_ros    = avg("Corsia_Rossa",     "total_lost")
                lost_hos    = avg("Hospitalizzazione","total_lost")
                cap_str     = str(cap_h) if cap_h is not None else "inf"

                row = (
                    f"{combo_num:>3}  {lam*60:>6.1f}  {num_triage_nurses:>6}  "
                    f"{cap_str:>6}  "
                    f"{Wq_tot:>8.3f}  {W_tot:>8.3f}  "
                    f"{rho_seg:>8.4f}  {rho_tri:>8.4f}  "
                    f"{rho_bia:>8.4f}  {rho_gia:>8.4f}  "
                    f"{rho_ver:>8.4f}  {rho_ros:>8.4f}  "
                    f"{rho_hos:>8.4f}  "
                    f"{lost_bia:>9.1f}  {lost_gia:>9.1f}  "
                    f"{lost_ver:>9.1f}  {lost_ros:>9.1f}  "
                    f"{lost_hos:>9.1f}"
                )
                print(row)

                results_doe.append({
                    "combo":             combo_num,
                    "lam":               lam,
                    "lam_per_ora":       lam * 60,
                    "num_triage_nurses": num_triage_nurses,
                    "cap_h":             cap_h,
                    # medie per nodo
                    "Wq_tot":            Wq_tot,
                    "W_tot":             W_tot,
                    "rho_Segreteria":    rho_seg,
                    "rho_Triage":        rho_tri,
                    "rho_Bianca":        rho_bia,
                    "rho_Gialla":        rho_gia,
                    "rho_Verde":         rho_ver,
                    "rho_Rossa":         rho_ros,
                    "rho_Hospital":      rho_hos,
                    "Lost_Bianca":       lost_bia,
                    "Lost_Gialla":       lost_gia,
                    "Lost_Verde":        lost_ver,
                    "Lost_Rossa":        lost_ros,
                    "Lost_Hospital":     lost_hos,
                })

    print()
    return results_doe


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Scenari di validazione originali (invariati)
    # ------------------------------------------------------------------
    run_scenario1(n_runs=10, sim_time=50_000)
    run_scenario2(n_runs=10, sim_time=50_000)
    run_scenario3(n_runs=10, sim_time=50_000,
                  lam=0.3, mu_A=1.0, mu_B=1.2, mu_C=0.8)

    # ------------------------------------------------------------------
    # Probabilita' di assegnazione corsia (configurabili globalmente)
    # ------------------------------------------------------------------
    #   Bianca = 0.40  (codice bianco, non urgente)
    #   Gialla = 0.20  (codice giallo, urgente)
    #   Verde  = 0.30  (codice verde, poco urgente)
    #   Rossa  = 0.10  (codice rosso, emergenza)
    # ------------------------------------------------------------------
    WARD_PROBS = {
        NODE_BIANCA: 0.40,
        NODE_GIALLA: 0.20,
        NODE_VERDE:  0.30,
        NODE_ROSSA:  0.10,
    }

    # ------------------------------------------------------------------
    # Scenario PS — singola run (parametri base da specifica)
    #   4 pazienti/ora, 2 infermieri triage, hospitalizzazione infinita
    # ------------------------------------------------------------------
    run_ed_scenario(
        lam               = 6 / 60,
        num_triage_nurses = 2,
        cap_h             = None,
        ward_probs        = WARD_PROBS,
        n_runs            = 10,
        sim_time          = 50_000,
        seed              = 123459,
    )
    
    run_ed_doe(
        arrival_rates      = [4/60, 8/60],   # − and +
        triage_nurses_list = [1, 3],         # − and +
        cap_h_list         = [5, 10],        # − and +
        ward_probs         = WARD_PROBS,
        n_runs             = 10,
        sim_time           = 50_000,
        seed               = 123459,
    )