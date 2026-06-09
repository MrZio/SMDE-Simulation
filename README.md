# Queue Model Implementation — Practical Session

Three deliverables, one coherent system. Each addresses a part of the spec and
they cross-validate each other.

| File | Language | Spec requirement |
|------|----------|------------------|
| `queue_sim.py` | Python | Modular, event-driven queue model in Kendall's notation + TRACE + theory validation |
| `prng_validation.R` | R | GNA/GVA: a PRNG validated by a battery of ≥5 statistical tests |
| `mm1_validation.gps` | GPSS | Independent (GPSS-compatible) validation model |

The Python simulator and the R generator use the **same LCG**
(`a=1103515245, c=12345, m=2³¹, seed=42`), so the randomness validated in R is
exactly the randomness driving the simulation.

---

## 1. `queue_sim.py` — the simulator

Maps directly onto the four core requirements:

- **Flexible parametrization** — `demo_single(A, S, c, N, D, lam, mu)` accepts any
  valid Kendall notation. `Distribution` turns each letter (`M`/`D`/`G`) into a
  sampler; `Station` takes `c`, `capacity` (`N`) and `discipline` (`FIFO`/`LIFO`/`SIRO`).
- **Modularity** — `Source` (the `A` stream) is decoupled from `Station` (the
  `S/c/N/D` node). A `Station` carries one `router` callback; returning another
  `Station` connects them. `demo_tandem()` builds `M/M/1 → M/M/1` with a single
  `router=lambda ent: s2`. Multi-stage networks need **no changes** to the core.
- **Event-driven** — `Simulator` holds a Future Event List (a binary heap). The
  clock jumps straight to the next event (`self.clock = ev.time`); there is no
  fixed-step loop. Events are `ARRIVE` and `DEPART`; queue-state changes are
  side effects of those two.
- **Validation** — `theory()` returns closed-form `L, Lq, W, Wq, ρ` for the
  Markovian cases (M/M/1, M/M/1/N, M/M/c) and the run prints simulated-vs-theory
  side by side. The **TRACE** (timestamped event log) is kept in `sim.trace` and
  partly printed; it is what you diff against GPSS.

Run it:
```bash
python3 queue_sim.py
```

Validation actually obtained (20 000 time units, ρ = 0.8 reference case):

| metric | simulated | theory | rel. err |
|--------|-----------|--------|----------|
| ρ  | 0.7916 | 0.80 | 1.1 % |
| L  | 3.911  | 4.00 | 2.2 % |
| Lq | 3.120  | 3.20 | 2.5 % |
| W  | 3.951  | 4.00 | 1.2 % |
| Wq | 3.151  | 3.20 | 1.5 % |

M/M/1/5 (loss), M/M/2, LIFO and the tandem network all validate within ~2.5 %.

---

## 2. `prng_validation.R` — generator + test battery

LCG implementation, then **8 tests** (spec asks for ≥5):
Frequency, Serial, Gap, Poker, Order (via `randtoolbox`), plus Chi-square,
Kolmogorov–Smirnov, and a Runs test, plus an ACF plot for independence.
A robust `interpret()` reduces multi-dimensional p-values to the most
conservative one and prints PASS/FAIL at α = 0.05, then a summary table.

Run it:
```r
Rscript prng_validation.R     # or open as an R Notebook
```

Coherence check: the LCG first values are
`0.582308, 0.519819, 0.465976, 0.777037, 0.422865` (mean over 10 000 ≈ 0.498,
variance ≈ 0.0837) — identical in R and in Python's `LCG`.

> Note: this classic LCG has weak low-order bits and may FAIL the higher-order
> serial/order tests. That is the lesson — the battery is the instrument that
> tells you whether a generator is fit for purpose.

---

## 3. `mm1_validation.gps` — GPSS cross-check

A transaction-flow M/M/1 with the same `λ=1.0, μ=1.25`. Two `QUEUE` probes
(`SYSTEM`, `LINE`) recover `L/W` and `Lq/Wq`; the `SERVER` facility gives ρ.
Because GPSS uses a different engine and paradigm, matching the same theoretical
numbers is independent evidence the model is right. The file also shows how to
shorten the run and enable `TRACE` to diff event-by-event against the Python
TRACE.

Run it: open in **GPSS World** (or `txtmestat`/JGPSS) and `START`.

---

## Suggested talking points for the report
1. Why event scheduling beats fixed-step (no wasted ticks, exact event times).
2. How the `Source`/`Station`/`router` split makes networks fall out for free.
3. The three-way agreement: theory ≈ Python ≈ GPSS, all fed by an R-validated RNG.
