from pydantic import BaseModel, ConfigDict
from enum import Enum
import heapq, random
from dataclasses import dataclass, field


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
    ARRIVAL = 'Arrival'
    DEPARTURE = 'Departure'


# Entity must be defined before Event, since Event references it
class Entity(BaseModel):
    id: int

    # predetermined route: list of node ids the entity will visit, in order.
    # sampled once at creation as a random permutation of a random subset
    # of stations (no cycles, each station at most once).
    path: list[int] = []
    path_index: int = 0   # which hop of the path we are currently at

    # state variables of Entity
    # one entry per node visited, in order:
    arrival_times: list[float] = []     # arrival_times[i] = time entering node i
    service_starts: list[float] = []    # service_starts[i] = time service began at node i
    departure_times: list[float] = []   # departure_times[i] = time leaving node i

# each node is a complete queue -> useful for multiple queues
# each node is a station with servers, exists for the whole simulation
class QueueNode(BaseModel):
    id: int
    name: str
    arrival_rate: float                             # lambda
    service_rate: float                             # mu
    arrival_distribution: DistributionArrivalTimes  # A
    service_distribution: DistributionServiceTimes  # S
    num_servers: int                                # c
    sys_capacity: int | None                        # N, where None means infinite
    queue_discipline: QueueDiscipline               # D
    # In the mesh model every node generates its own external arrivals
    # AND can receive entities routed from any other node. The successor
    # of a node is not fixed: it is the next id in the entity's own path.
    # 'reachable' lists which node ids this node may route to (the
    # topology). If empty, defaults to "all other nodes" at build time.
    reachable: list[int] = []

    # state variables of queue nodes
    waiting_queue: list[Entity] = []
    busy_servers: int = 0

    # area-under-curve method) 
    area_queue_length: float = 0.0   # integral of Lq(t) dt
    area_system_length: float = 0.0  # integral of L(t)  dt (queue + in service)
    last_event_time: float = 0.0
    total_arrivals: int = 0
    total_lost: int = 0
    total_served: int = 0

@dataclass(order=True)  # order by time -> makes heapq work directly on Event
class Event:
    time: float
    event_type: EventType = field(compare=False)
    node: QueueNode = field(compare=False)
    entity: Entity = field(compare=False, default=None)

class Simulation(BaseModel):
    # state variables of the system
    clock: float = 0.0
    event_list: list[Event] = []


    nodes: list[QueueNode] = []
    entity_served: list[Entity] = []
    entity_counter: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _node_by_id(self, node_id: int) -> QueueNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"No node with id {node_id}")

    def _sample_path(self, entry_node_id: int) -> list[int]:
        """Build a random path: a random-length permutation of a random
        subset of the stations, starting at entry_node_id, with no repeats
        (no cycles). The entry node is always first; the remaining stations
        are drawn from entry_node.reachable (or all other nodes)."""
        entry = self._node_by_id(entry_node_id)
        candidates = list(entry.reachable) if entry.reachable else \
            [n.id for n in self.nodes if n.id != entry_node_id]
        # random subset size in [0, len(candidates)], then random order
        k = random.randint(0, len(candidates))
        extra = random.sample(candidates, k)
        return [entry_node_id] + extra

    def schedule(self, event: Event):
        heapq.heappush(self.event_list, event)

    # numerical technique to track time changes of states
    def run(self, until: float):
        # every node is an external source in the mesh model
        for node in self.nodes:
            self.schedule_arrival(node)

        while self.event_list:
            next_event = heapq.heappop(self.event_list)
            if next_event.time > until:
                break

            # update area-under-curve stats for the elapsed interval
            self._update_area_stats(next_event.node, next_event.time)

            self.clock = next_event.time
            self.process_event(next_event)

    def schedule_arrival(self, node: QueueNode):
        """Generate the next EXTERNAL arrival at 'node'. The new entity gets
        its full random path sampled now (predetermined route), starting at
        this node."""
        time = self.clock + self.arrival_time(node)
        path = self._sample_path(node.id)
        ent = Entity(id=self.entity_counter, path=path, path_index=0,
                     arrival_times=[time])
        self.entity_counter += 1
        self.schedule(Event(time=time, event_type=EventType.ARRIVAL, node=node, entity=ent))

    # time changes of the state variables as a function of the events (activities)
    def process_event(self, event: Event):
        if event.event_type == EventType.ARRIVAL:
            self.handle_arrival(event)
        elif event.event_type == EventType.DEPARTURE:
            self.handle_departure(event)

    def _update_area_stats(self, node: QueueNode, now: float):
        """Accumulate Lq(t) and L(t) areas up to 'now', using the state
        the node was in during [last_event_time, now)."""
        dt = now - node.last_event_time
        if dt > 0:
            node.area_queue_length += len(node.waiting_queue) * dt
            node.area_system_length += (len(node.waiting_queue) + node.busy_servers) * dt
        node.last_event_time = now

    def handle_arrival(self, event: Event):
        node = event.node
        entity = event.entity
        node.total_arrivals += 1

        total_in_system = len(node.waiting_queue) + node.busy_servers
        if node.sys_capacity is not None and total_in_system >= node.sys_capacity:
            node.total_lost += 1  # entity lost (blocked)
        elif node.busy_servers < node.num_servers:  # a server is free
            node.busy_servers += 1
            entity.service_starts.append(self.clock)
            service_time = self.service_time(node)

            self.schedule(Event(
                time=self.clock + service_time,
                event_type=EventType.DEPARTURE,
                node=node,
                entity=entity
            ))
        else:
            node.waiting_queue.append(entity)

        # In the mesh model EVERY node generates its own external arrivals.
        # Re-arm the external arrival stream ONLY when this event is an
        # external arrival (i.e. the entity is at the first hop of its path
        # and that hop is this node). Routed arrivals (path_index > 0) must
        # not spawn new external arrivals, otherwise the rate would explode.
        if entity.path_index == 0 and entity.path and entity.path[0] == node.id:
            self.schedule_arrival(node)  # schedule next external arrival

    def handle_departure(self, event: Event):
        node = event.node
        entity = event.entity
        entity.departure_times.append(self.clock)
        node.total_served += 1

        # free the server / pull the next waiting entity into service
        if node.waiting_queue:
            next_entity = self.pick_next_entity(node)
            next_entity.service_starts.append(self.clock)
            service_time = self.service_time(node)
            self.schedule(Event(
                time=self.clock + service_time,
                event_type=EventType.DEPARTURE,
                node=node,
                entity=next_entity
            ))
        else:
            node.busy_servers -= 1

        # routing: advance along the entity's predetermined path
        entity.path_index += 1
        if entity.path_index < len(entity.path):
            next_node = self._node_by_id(entity.path[entity.path_index])
            self.route_to_next(entity, next_node)
        else:
            # path complete -> entity exits the system
            self.entity_served.append(entity)

    def final_statistics(self) -> dict:
        """Compute Wq, Lq, W, L, rho and other descriptive stats per node
        and for the overall simulation. In the mesh model each entity has
        its OWN path, so the i-th time-field entry corresponds to the i-th
        node in THAT entity's path. To collect per-node stats we look up,
        for each served entity, the position of the node in its path."""
        # flush remaining area up to the final clock for every node
        for node in self.nodes:
            self._update_area_stats(node, self.clock)

        results = {}
        for node in self.nodes:
            # Lq, L from area-under-curve
            Lq = node.area_queue_length / self.clock if self.clock > 0 else 0.0
            L = node.area_system_length / self.clock if self.clock > 0 else 0.0

            # rho = server utilization
            rho = (node.area_system_length - node.area_queue_length) / (self.clock * node.num_servers) \
                if self.clock > 0 else 0.0

            # per-node Wq, W (stage-level): for each served entity, find
            # where this node sits in the entity's path; if present and the
            # entity has recorded times for that hop, accumulate the wait.
            wq_node, w_node = [], []
            for e in self.entity_served:
                if node.id in e.path:
                    i = e.path.index(node.id)
                    if i < len(e.arrival_times) and i < len(e.departure_times) \
                            and i < len(e.service_starts):
                        wq_node.append(e.service_starts[i] - e.arrival_times[i])
                        w_node.append(e.departure_times[i] - e.arrival_times[i])

            results[node.name] = {
                "Lq": Lq,                           # avg number of entities waiting
                "L": L,                             # avg number of entities in system
                "rho": rho,                         # server utilization
                "sys_capacity": node.sys_capacity,  # N (Kendall) None is infinite
                "num_servers": node.num_servers,    # c (Kendall)
                "total_arrivals": node.total_arrivals,
                "total_served": node.total_served,
                "total_lost": node.total_lost,
                "Wq": sum(wq_node) / len(wq_node) if wq_node else 0.0,  # stage-level wait
                "W": sum(w_node) / len(w_node) if w_node else 0.0,      # stage-level total time
            }

        # overall = full path through the system, first node entered ->
        # last node departed. entity_served only contains entities that
        # completed their whole path (see handle_departure).
        wq_overall, w_overall = [], []
        for e in self.entity_served:
            if e.arrival_times and e.departure_times and e.service_starts:
                hops = min(len(e.arrival_times), len(e.service_starts))
                wq_overall.append(sum(
                    e.service_starts[i] - e.arrival_times[i] for i in range(hops)
                ))
                w_overall.append(e.departure_times[-1] - e.arrival_times[0])

        results["overall"] = {
            "n_entities_served": len(self.entity_served),
            "Wq": sum(wq_overall) / len(wq_overall) if wq_overall else 0.0,
            "W": sum(w_overall) / len(w_overall) if w_overall else 0.0,
            "simulation_time": self.clock,
        }
        return results

    def pick_next_entity(self, node: QueueNode):
        if node.queue_discipline == QueueDiscipline.FIFO:
            return node.waiting_queue.pop(0)
        elif node.queue_discipline == QueueDiscipline.LIFO:
            return node.waiting_queue.pop(-1)
        elif node.queue_discipline == QueueDiscipline.SIRO:
            entity = random.choice(node.waiting_queue)
            node.waiting_queue.remove(entity)
            return entity

    def arrival_time(self, node: QueueNode) -> float:
        if node.arrival_distribution == DistributionArrivalTimes.M:
            return random.expovariate(node.arrival_rate)
        elif node.arrival_distribution == DistributionArrivalTimes.D:
            return 1.0 / node.arrival_rate
        return random.uniform(0.5, 1.5)

    def service_time(self, node: QueueNode) -> float:
        if node.service_distribution == DistributionServiceTimes.M:
            return random.expovariate(node.service_rate)
        elif node.service_distribution == DistributionServiceTimes.D:
            return 1.0 / node.service_rate
        return random.uniform(0.5, 1.5)

    def route_to_next(self, entity: Entity, next_node: QueueNode, delay: float = 0.0):
        arrival_time_next = self.clock + delay
        entity.arrival_times.append(arrival_time_next)
        self.schedule(Event(
            time=arrival_time_next,
            event_type=EventType.ARRIVAL,
            node=next_node,
            entity=entity
        ))


def print_comparison(stats: dict, node_name: str, theoretical: dict | None = None,
                     overall: bool = True, title: str = ""):
    """Print simulation results for a node side-by-side with theoretical
    values (if provided). 'theoretical' is a dict with keys among:
    rho, Lq, L, Wq, W. Pass None when no theoretical reference exists
    (e.g. for stages of a tandem queue without a closed-form solution)."""
    s = stats[node_name]
    o = stats["overall"] if overall else None

    if title:
        print(f"\n{title}")
        print("=" * len(title))

    col1 = "Metric"
    col2 = "Simulated"
    col3 = "Theoretical"
    col4 = "Abs. error"
    print(f"{col1:<22}{col2:>14}{col3:>16}{col4:>14}")
    print("-" * 66)

    def row(label: str, sim_val, theo_val=None, fmt: str = "{:.4f}"):
        sim_str = fmt.format(sim_val) if isinstance(sim_val, (int, float)) else str(sim_val)
        if theo_val is None:
            theo_str = "-"
            err_str = "-"
        else:
            theo_str = fmt.format(theo_val)
            err_str = fmt.format(abs(sim_val - theo_val))
        print(f"{label:<22}{sim_str:>14}{theo_str:>16}{err_str:>14}")

    t = theoretical or {}

    print(f"\n[Node-level: {node_name}]")
    row("c (servers)",   s["num_servers"],  fmt="{}")
    row("N (capacity)",  s["sys_capacity"] if s["sys_capacity"] is not None else "inf", fmt="{}")
    row("rho",           s["rho"],          t.get("rho"))
    row("Lq",            s["Lq"],           t.get("Lq"))
    row("L",             s["L"],            t.get("L"))
    row("Wq (stage)",    s["Wq"],           t.get("Wq"))
    row("W  (stage)",    s["W"],            t.get("W"))
    row("Arrivals",      s["total_arrivals"], fmt="{}")
    row("Served",        s["total_served"],   fmt="{}")
    row("Lost (blocked)",s["total_lost"],     fmt="{}")

    if overall and o is not None:
        print(f"\n[System-wide]")
        row("Simulation time", o["simulation_time"], fmt="{:.2f}")
        row("entities served", o["n_entities_served"], fmt="{}")
        row("Wq (overall)", o["Wq"], t.get("Wq_overall"))
        row("W  (overall)", o["W"],  t.get("W_overall"))


if __name__ == "__main__":
    # --- Example 1: single M/M/1 queue, lambda=0.8, mu=1.0 (rho = 0.8) ---
    node = QueueNode(
        id=1,
        name="M/M/1",
        arrival_rate=0.8,
        service_rate=1.0,
        arrival_distribution=DistributionArrivalTimes.M,
        service_distribution=DistributionServiceTimes.M,
        num_servers=1,
        sys_capacity=None,
        queue_discipline=QueueDiscipline.FIFO,
    )

    sim = Simulation(nodes=[node])
    sim.run(until=100000)
    stats = sim.final_statistics()

    # M/M/1 closed-form formulas
    lam, mu = node.arrival_rate, node.service_rate
    rho_theo = lam / mu
    Lq_theo = rho_theo**2 / (1 - rho_theo)
    L_theo = rho_theo / (1 - rho_theo)
    Wq_theo = Lq_theo / lam
    W_theo = L_theo / lam

    print_comparison(
        stats,
        node_name="M/M/1",
        theoretical={
            "rho": rho_theo, "Lq": Lq_theo, "L": L_theo,
            "Wq": Wq_theo,   "W":  W_theo,
            "Wq_overall": Wq_theo, "W_overall": W_theo,  # same as stage for 1-node system
        },
        title="Example 1: single M/M/1 queue (lambda=0.8, mu=1.0)",
    )

    # --- Example 2: 3-station mesh (any station -> any station) ---
    # Every station generates its own external arrivals (A) and can route
    # to any other station. Each entity's path is a random permutation of a
    # random subset of stations, sampled at creation (no cycles).
    s1 = QueueNode(
        id=1, name="Station_1",
        arrival_rate=0.3, service_rate=1.0,
        arrival_distribution=DistributionArrivalTimes.M,
        service_distribution=DistributionServiceTimes.M,
        num_servers=1, sys_capacity=None,
        queue_discipline=QueueDiscipline.FIFO,
        reachable=[2, 3],
    )
    s2 = QueueNode(
        id=2, name="Station_2",
        arrival_rate=0.3, service_rate=1.2,
        arrival_distribution=DistributionArrivalTimes.M,
        service_distribution=DistributionServiceTimes.D,
        num_servers=2, sys_capacity=10,
        queue_discipline=QueueDiscipline.LIFO,
        reachable=[1, 3],
    )
    s3 = QueueNode(
        id=3, name="Station_3",
        arrival_rate=0.3, service_rate=0.9,
        arrival_distribution=DistributionArrivalTimes.D,
        service_distribution=DistributionServiceTimes.M,
        num_servers=1, sys_capacity=None,
        queue_discipline=QueueDiscipline.SIRO,
        reachable=[1, 2],
    )

    sim2 = Simulation(nodes=[s1, s2, s3])
    sim2.run(until=50000)
    stats2 = sim2.final_statistics()

    # --- Approximate theoretical reference for mesh stations ---
    # Assumption: the total arrival process at each station is approximately
    # Poisson with effective rate lambda_eff = lambda_ext + sum of routed.
    # P(station j appears in path of entity born at i) = 1/2 when
    # len(reachable_i) = 2 and subset size is uniform in [0, 2].
    # lambda_eff_j = lambda_j + sum_{i != j} lambda_i * P(j in path from i)
    import math
    all_nodes = [s1, s2, s3]
    node_map = {n.id: n for n in all_nodes}

    def compute_lambda_eff(target: QueueNode) -> float:
        lam = target.arrival_rate
        for src in all_nodes:
            if src.id == target.id:
                continue
            R = len(src.reachable) if src.reachable else len(all_nodes) - 1
            p_included = 0.5  # = sum_{k=0}^{R} (1/(R+1))*(k/R) = 1/2
            lam += src.arrival_rate * p_included
        return lam

    def mmc_theoretical(lam_eff: float, mu: float, c: int) -> dict:
        """Approximate M/M/c steady-state formulas."""
        rho = lam_eff / (c * mu)
        if rho >= 1:
            return {"rho": rho, "Lq": float('inf'), "L": float('inf'),
                    "Wq": float('inf'), "W": float('inf')}
        a = lam_eff / mu  # = c * rho
        # P0
        s = sum(a**n / math.factorial(n) for n in range(c))
        s += a**c / (math.factorial(c) * (1 - rho))
        P0 = 1.0 / s
        # Lq (Erlang-C based)
        Lq = P0 * a**c * rho / (math.factorial(c) * (1 - rho)**2)
        Wq = Lq / lam_eff
        W = Wq + 1.0 / mu
        L = lam_eff * W
        return {"rho": rho, "Lq": Lq, "L": L, "Wq": Wq, "W": W}

    for nd in all_nodes:
        lam_eff = compute_lambda_eff(nd)
        theo = mmc_theoretical(lam_eff, nd.service_rate, nd.num_servers)
        is_last = (nd.id == all_nodes[-1].id)
        print_comparison(stats2, node_name=nd.name, theoretical=theo,
                         overall=is_last,
                         title=f"\nExample 2: 3-station mesh - {nd.name} (approx lambda_eff={lam_eff:.2f})")