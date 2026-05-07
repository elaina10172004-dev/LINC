import math
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Customer:
    idx: int
    x: float
    y: float
    demand: int
    ready_time: float
    due_time: float
    service_time: float


@dataclass
class Instance:
    name: str
    capacity: int
    depot: Customer
    customers: List[Customer]


def _parse_solomon_lines(lines: list[str]) -> Instance:
    name = lines[0].strip()

    cap = None
    for i in range(min(15, len(lines))):
        if "CAPACITY" in lines[i].upper():
            source = lines[i + 1] if i + 1 < len(lines) else lines[i]
            tokens = re.findall(r"[-+]?\d+", source)
            if tokens:
                cap = int(tokens[-1])
            break

    tstart = None
    for i, line in enumerate(lines):
        upper = line.upper()
        if "CUST" in upper and "XCOORD" in upper and "YCOORD" in upper:
            tstart = i + 1
            break
    if tstart is None:
        raise ValueError("Could not locate Solomon table header")

    rows = []
    for line in lines[tstart:]:
        if not line.strip():
            continue
        toks = line.split()
        if len(toks) < 7:
            continue
        try:
            rid, x, y, q, r, d, s = toks[:7]
            rows.append(
                (
                    int(rid),
                    float(x),
                    float(y),
                    int(float(q)),
                    float(r),
                    float(d),
                    float(s),
                )
            )
        except Exception:
            continue

    if not rows:
        raise ValueError("No rows parsed from Solomon file")

    rid, x, y, q, r, d, s = rows[0]
    depot = Customer(rid, x, y, q, r, d, s)
    customers = [Customer(rid, x, y, q, r, d, s) for rid, x, y, q, r, d, s in rows[1:]]
    if cap is None:
        cap = 9_999_999
    return Instance(name=name, capacity=cap, depot=depot, customers=customers)


def read_solomon(path: str) -> Instance:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return _parse_solomon_lines(lines)


def euclid(a: Customer, b: Customer) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def route_cost_and_feasible(inst: Instance, route: list[int]) -> Tuple[float, bool, list[float]]:
    cap_used = 0
    time = inst.depot.ready_time
    dist = 0.0
    arr = []
    prev = inst.depot
    feasible = True

    for cid in route:
        customer = inst.customers[cid]
        cap_used += customer.demand
        if cap_used > inst.capacity:
            feasible = False

        travel = euclid(prev, customer)
        dist += travel
        time += travel
        if time < customer.ready_time:
            time = customer.ready_time
        if time > customer.due_time + 1e-9:
            feasible = False

        arr.append(time)
        time += customer.service_time
        prev = customer

    dist += euclid(prev, inst.depot)
    time += euclid(prev, inst.depot)
    if inst.depot.due_time > 0 and time > inst.depot.due_time + 1e-9:
        feasible = False
    return dist, feasible, arr
