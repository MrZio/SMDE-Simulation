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

