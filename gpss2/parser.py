#!/usr/bin/env python3
"""
parse_gpss_report.py
====================
Read a GPSS World report / journal (plain text) and extract the queueing
metrics. RULE: only values that appear unambiguously in the report are
reported; anything not present is printed as "N/A" with the reason. Nothing
is invented or derived.

Usage:
    python3 parse_gpss_report.py  <report_or_journal.txt>

What it reads (from the standard GPSS report tables):
  QUEUE table    -> AVE.CONT. (= Lq or L) and AVE.TIME (= Wq or W), ENTRY count
  STORAGE table  -> UTIL. (= rho), ENTRIES (= served)
  FACILITY table -> UTIL. (= rho), ENTRIES (= served)
  BLOCK table    -> entry count of a "LOST*" block (= blocked), TERMINATE counts
  header line    -> END TIME (= simulation time)

Naming conventions used to map entities to nodes (override in CONFIG below):
  * a queue whose name contains 'WAIT'  -> the WAITING queue  -> Lq, Wq
  * a queue whose name contains 'SYS'   -> the SYSTEM  queue  -> L,  W
  * a STORAGE or FACILITY                -> the server(s)      -> rho, served
  * a block whose label contains 'LOST' -> blocked count
  * node id = trailing digits of the name (SERV1/WAIT1/SYS1 -> node "1";
    SERVER/WAIT_Q/SYSTEM with no digits -> node "single")
"""
import sys
import re

# ---- CONFIG: naming patterns (edit if your model uses other names) ---------
WAIT_PAT = "WAIT"     # queue measuring time waiting for a server  -> Lq, Wq
SYS_PAT  = "SYS"      # queue measuring whole time in the node      -> L,  W
LOST_PAT = "LOST"     # block label for blocked/lost entities
SECTION_KEYWORDS = {"QUEUE", "STORAGE", "FACILITY", "LABEL", "FEC", "CEC",
                    "NAME", "GPSS", "RETRY", "TABLE", "SAVEVALUE", "MATRIX",
                    "LOGICSWITCH", "USERCHAIN"}


def to_num(tok):
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return None


def strip_rtf(text):
    """If the file was saved as RTF, recover the plain text."""
    if not text.lstrip().startswith("{\\rtf"):
        return text
    text = re.sub(r"\\par[d]?", "\n", text)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def find_header(lines, first_token, must_contain):
    """Return the index of a table header line: first token == first_token and
    the line contains all of 'must_contain'."""
    for i, ln in enumerate(lines):
        toks = ln.split()
        if toks and toks[0] == first_token and all(c in ln for c in must_contain):
            return i
    return None


def parse_named_table(lines, keyword, must_contain):
    """Parse a GPSS table whose header starts with 'keyword'. Returns
    {entity_name: {column_name: value}}."""
    idx = find_header(lines, keyword, must_contain)
    if idx is None:
        return {}
    cols = lines[idx].split()[1:]          # column names after the keyword
    out = {}
    for ln in lines[idx + 1:]:
        if not ln.strip():
            break
        toks = ln.split()
        if toks[0] in SECTION_KEYWORDS:
            break
        name = toks[0]
        vals = toks[1:]
        # a real data row has mostly numeric values; otherwise we've left the table
        nums = [to_num(v) for v in vals]
        if not vals or sum(n is not None for n in nums) < max(1, len(vals) // 2):
            break
        row = {}
        for c, v in zip(cols, vals):
            row[c] = to_num(v)
        out[name] = row
    return out


def col(row, *candidates):
    """Fetch a column by exact name first, then by substring. None if absent."""
    if row is None:
        return None
    for c in candidates:                    # exact match wins
        if c in row:
            return row[c]
    for c in candidates:                    # then substring
        for k in row:
            if c in k:
                return row[k]
    return None


def parse_blocks(lines):
    """Parse the BLOCK table -> list of dicts {label, loc, type, entry, current}."""
    idx = find_header(lines, "LABEL", ["LOC", "BLOCK", "ENTRY"])
    if idx is None:
        return []
    blocks = []
    for ln in lines[idx + 1:]:
        if not ln.strip():
            break
        toks = ln.split()
        if toks[0] in SECTION_KEYWORDS and toks[0] != "LABEL":
            break
        # trailing run of integers = ... ENTRY_COUNT CURRENT_COUNT [RETRY]
        trail = []
        for t in reversed(toks):
            n = to_num(t)
            if isinstance(n, int):
                trail.append(n)
            else:
                break
        trail.reverse()
        if len(trail) < 2:
            continue
        # first integer token from the left = LOC
        loc_i = next((i for i, t in enumerate(toks) if isinstance(to_num(t), int)), None)
        if loc_i is None:
            continue
        label = " ".join(toks[:loc_i]) if loc_i > 0 else ""
        loc = to_num(toks[loc_i])
        if len(trail) >= 3:
            entry, current = trail[-3], trail[-2]
        else:
            entry, current = trail[-2], trail[-1]
        btype = " ".join(toks[loc_i + 1: len(toks) - len(trail)])
        blocks.append({"label": label, "loc": loc, "type": btype,
                       "entry": entry, "current": current})
    return blocks


def sim_time(lines):
    for i, ln in enumerate(lines):
        if "START TIME" in ln and "END TIME" in ln:
            for ln2 in lines[i + 1:]:
                if ln2.strip():
                    nums = [to_num(t) for t in ln2.split()]
                    nums = [n for n in nums if n is not None]
                    if len(nums) >= 2:
                        return nums[1]       # END TIME is the 2nd number
                    break
    return None


def node_key(name):
    m = re.search(r"(\d+)$", name)
    return m.group(1) if m else "single"


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    if len(sys.argv) < 2:
        print("usage: python3 parse_gpss_report.py <report_or_journal.txt>")
        sys.exit(1)
    with open(sys.argv[1], "r", errors="replace") as f:
        text = strip_rtf(f.read())
    lines = text.splitlines()

    queues   = parse_named_table(lines, "QUEUE",   ["AVE.CONT.", "AVE.TIME"])
    storages = parse_named_table(lines, "STORAGE", ["UTIL."])
    faclts   = parse_named_table(lines, "FACILITY",["UTIL."])
    blocks   = parse_blocks(lines)
    st = sim_time(lines)

    # ---- group entities by node ------------------------------------------
    servers = {}
    for nm, r in {**storages, **faclts}.items():
        servers[nm] = r
    nodes = {}
    def node(k):
        return nodes.setdefault(k, {"wait": None, "sys": None, "server": None,
                                    "server_name": None, "lost": None})
    for nm, r in queues.items():
        if WAIT_PAT in nm.upper():
            node(node_key(nm))["wait"] = r
        elif SYS_PAT in nm.upper():
            node(node_key(nm))["sys"] = r
    for nm, r in servers.items():
        n = node(node_key(nm)); n["server"] = r; n["server_name"] = nm
    for b in blocks:
        if LOST_PAT in (b["label"] or "").upper():
            node(node_key(b["label"]))["lost"] = b["entry"]

    print("=" * 60)
    print(f"GPSS report: {sys.argv[1]}")
    print("=" * 60)

    n_nodes = len(nodes)
    for k in sorted(nodes):
        nd = nodes[k]
        wait, sysq, srv = nd["wait"], nd["sys"], nd["server"]
        rho = col(srv, "UTIL.")
        Lq  = col(wait, "AVE.CONT.")
        Wq  = col(wait, "AVE.TIME")
        L   = col(sysq, "AVE.CONT.")
        W   = col(sysq, "AVE.TIME")
        served = col(srv, "ENTRIES")
        lost = nd["lost"]
        # arrivals = admitted (SYS or WAIT entries) + lost
        admitted = col(sysq, "ENTRY")
        if admitted is None:
            admitted = col(wait, "ENTRY")
        arrivals = (admitted + lost) if (admitted is not None and lost is not None) \
            else (admitted if (admitted is not None and lost is None) else None)
        lost_disp = lost if lost is not None else (0 if srv or wait else None)

        print(f"\nNode '{k}'"
              + (f"  (server: {nd['server_name']})" if nd['server_name'] else ""))
        print(f"  rho            = {fmt(rho)}"
              + ("" if rho is not None else "   <- no STORAGE/FACILITY UTIL. found"))
        print(f"  Lq             = {fmt(Lq)}"
              + ("" if Lq is not None else f"   <- no '{WAIT_PAT}' queue found"))
        print(f"  L              = {fmt(L)}"
              + ("" if L is not None else f"   <- no '{SYS_PAT}' queue (L not in report)"))
        print(f"  Wq             = {fmt(Wq)}"
              + ("" if Wq is not None else f"   <- no '{WAIT_PAT}' queue found"))
        print(f"  W              = {fmt(W)}"
              + ("" if W is not None else f"   <- no '{SYS_PAT}' queue (W not in report)"))
        print(f"  Arrivals       = {fmt(arrivals)}"
              + ("" if arrivals is not None else "   <- no SYS/WAIT ENTRY count"))
        print(f"  Served         = {fmt(served)}"
              + ("" if served is not None else "   <- no server ENTRIES count"))
        if lost is not None:
            print(f"  Lost (blocked) = {lost}")
        elif srv or wait:
            print(f"  Lost (blocked) = 0   (no '{LOST_PAT}' block in model -> no blocking)")
        else:
            print(f"  Lost (blocked) = N/A")

    # ---- overall ---------------------------------------------------------
    terms = [b for b in blocks if b["type"].strip().upper().startswith("TERMINATE")]
    n_served = max((b["entry"] for b in terms), default=None)
    print("\n" + "-" * 60)
    print("OVERALL (system-wide)")
    print("-" * 60)
    print(f"  Simulation time      = {fmt(st)}"
          + ("" if st is not None else "   <- END TIME not found"))
    if n_served is not None:
        print(f"  Customers served     = {n_served}   "
              f"(largest TERMINATE count; verify it is the customer exit, not the timer)")
        if len(terms) > 1:
            allt = ", ".join(f"{(b['label'] or 'loc'+str(b['loc']))}={b['entry']}" for b in terms)
            print(f"      all TERMINATE blocks: {allt}")
    else:
        print("  Customers served     = N/A   <- no TERMINATE block found")

    # Wq overall: rigorous only for a single node
    if n_nodes == 1:
        only = nodes[next(iter(nodes))]
        wq = col(only["wait"], "AVE.TIME")
        w  = col(only["sys"], "AVE.TIME")
        print(f"  Wq (overall)         = {fmt(wq)}   (single node: = node Wq)")
        print(f"  W  (overall)         = {fmt(w)}"
              + ("   (single node: = node W)" if w is not None
                 else "   <- no SYS queue; add a QUEUE spanning arrival->departure"))
    else:
        print("  Wq (overall)         = N/A   <- multi-node: per-customer sum of "
              "stage waits is not in the GPSS report (would need a custom probe)")
        print("  W  (overall)         = N/A   <- multi-node: needs ONE QUEUE spanning "
              "the first arrival to the final departure; per-node SYS queues do not give it")


if __name__ == "__main__":
    main()