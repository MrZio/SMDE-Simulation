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
    next_node: 'QueueNode | None' = None
    is_source: bool = True                          # False = receives entities only via routing

    waiting_queue: list[Entity] = []
    busy_servers: int = 0

    # --- bookkeeping for statistics (area-under-curve method) ---
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
    clock: float = 0.0
    event_list: list[Event] = []
    nodes: list[QueueNode] = []
    entity_served: list[Entity] = []
    entity_counter: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def schedule(self, event: Event):
        heapq.heappush(self.event_list, event)

    def run(self, until: float):
        for node in self.nodes:
            if node.is_source:
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
        time = self.clock + self.arrival_time(node)
        cust = Entity(id=self.entity_counter, arrival_times=[time])
        self.entity_counter += 1
        self.schedule(Event(time=time, event_type=EventType.ARRIVAL, node=node, entity=cust))

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

        if node.is_source:
            self.schedule_arrival(node)  # schedule next external arrival

    def handle_departure(self, event: Event):
        node = event.node
        entity = event.entity
        entity.departure_times.append(self.clock)
        node.total_served += 1

        # only count as "fully served" once it leaves the LAST node
        if not node.next_node:
            self.entity_served.append(entity)

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

        if node.next_node:
            self.route_to_next(entity, node.next_node)

    def final_statistics(self) -> dict:
        """Compute Wq, Lq, W, L, rho and other descriptive stats per node
        and for the overall simulation. entity time fields are lists,
        indexed by the order in which nodes were visited (0 = first node
        visited, -1 = last node visited)."""
        # flush remaining area up to the final clock for every node
        for node in self.nodes:
            self._update_area_stats(node, self.clock)

        # build a node-name -> position-in-path index, so we know which
        # list index in each entity's time fields corresponds to which
        # node. all entities follows same path
        node_order = []
        n = self.nodes[0] if self.nodes else None
        # find a source node to start the path from
        for candidate in self.nodes:
            if candidate.is_source:
                n = candidate
                break
        while n is not None:
            node_order.append(n)
            n = n.next_node
        name_to_index = {nd.name: i for i, nd in enumerate(node_order)}

        results = {}
        for node in self.nodes:
            idx = name_to_index.get(node.name)

            # Lq, L from area-under-curve
            Lq = node.area_queue_length / self.clock if self.clock > 0 else 0.0
            L = node.area_system_length / self.clock if self.clock > 0 else 0.0

            # rho = server utilization
            rho = (node.area_system_length - node.area_queue_length) / (self.clock * node.num_servers) \
                if self.clock > 0 else 0.0

            # per-node Wq, W (stage-level), using each served entity's
            # i-th visit (idx) - only entities that actually reached this
            # node have an entry at position idx
            wq_node, w_node = [], []
            for c in self.entity_served:
                if idx is not None and idx < len(c.arrival_times) and idx < len(c.departure_times):
                    wq_node.append(c.service_starts[idx] - c.arrival_times[idx])
                    w_node.append(c.departure_times[idx] - c.arrival_times[idx])

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
        # last node departed. entity_served only contains entities
        # that exited the LAST node (see handle_departure).
        wq_overall, w_overall = [], []
        for c in self.entity_served:
            if c.arrival_times and c.departure_times and c.service_starts:
                wq_overall.append(sum(
                    c.service_starts[i] - c.arrival_times[i]
                    for i in range(len(c.arrival_times))
                ))
                w_overall.append(c.departure_times[-1] - c.arrival_times[0])

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
    sim.run(until=10000)
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

    # --- Example 2: tandem queue (multi-stage), node1 -> node2 ---
    node1 = QueueNode(
        id=1, name="Station_1",
        arrival_rate=0.8, service_rate=1.0,
        arrival_distribution=DistributionArrivalTimes.M,
        service_distribution=DistributionServiceTimes.M,
        num_servers=1, sys_capacity=None,
        queue_discipline=QueueDiscipline.FIFO,
        is_source=True,
    )
    node2 = QueueNode(
        id=2, name="Station_2",
        arrival_rate=0.0, service_rate=1.2,
        arrival_distribution=DistributionArrivalTimes.M,
        service_distribution=DistributionServiceTimes.M,
        num_servers=1, sys_capacity=5,
        queue_discipline=QueueDiscipline.FIFO,
        is_source=False,
    )
    node1.next_node = node2

    sim2 = Simulation(nodes=[node1, node2])
    sim2.run(until=10000)
    stats2 = sim2.final_statistics()

    # Theoretical reference for Station_1: pure M/M/1 with lam=0.8, mu=1.0
    # For Station_2: arrivals are the departures of Station_1, which for an
    # OPEN tandem M/M/1 network (Jackson's theorem) are Poisson(lam=0.8).
    # But Station_2 has finite capacity N=5 -> blocking, so the closed-form
    # M/M/1 no longer applies exactly. We leave its theoretical column empty.
    s1_theo = {
        "rho": 0.8, "Lq": 0.8**2 / 0.2, "L": 0.8 / 0.2,
        "Wq": (0.8**2 / 0.2) / 0.8, "W": (0.8 / 0.2) / 0.8,
    }

    print_comparison(
        stats2, node_name="Station_1", theoretical=s1_theo, overall=False,
        title="\nExample 2: tandem queue - Station_1 (M/M/1, infinite capacity)",
    )
    print_comparison(
        stats2, node_name="Station_2", theoretical=None, overall=True,
        title="\nExample 2: tandem queue - Station_2 (M/M/1/5, no closed-form due to blocking)",
    )
