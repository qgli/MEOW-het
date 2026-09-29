"""Floor-plan generation for second-generation scenes (bpy-free).

Re-derives the 2D skeleton of the first-generation generator (room_builder.py:
6 parametric shapes + BSP split) and adds:

  * Connectivity guarantee: room-adjacency graph -> maximum-shared-span
    spanning tree (Kruskal over spans, widest first); every tree edge gets a
    door (width adapted to the shared span, minimum 0.7 m); loop edges get a
    door with some probability. BFS reachability is 100% by construction,
    asserted in tests over many seeds.
  * Functional room typing (living guaranteed; bedrooms ~ area; hallway in
    the corridor BSP variant) driving furniture sets and light schedules.
  * Windows on exterior spans with 1D anti-overlap (as in the first-generation
    wall_decorator), one closed entry door per dwelling.
  * Per-room light schedule with fixture-type / CCT / lumens randomness;
    the photometric exposure is derived separately.

All outputs are plain dicts/dataclass fields on GenesisSpec; solids.py
realizes them; nothing here imports bpy.
"""
from __future__ import annotations

import math
import random

# shapes
# generators re-derived from the first-generation room_builder: all return CCW 2D rings.


def _rect(rng):
    w, d = rng.uniform(6.0, 10.0), rng.uniform(5.0, 8.0)
    return [(0, 0), (w, 0), (w, d), (0, d)]


def _l_shape(rng):
    w, h = rng.uniform(7.0, 10.0), rng.uniform(7.0, 10.0)
    cx, cy = w * rng.uniform(0.35, 0.6), h * rng.uniform(0.35, 0.6)
    return [(0, 0), (w, 0), (w, h - cy), (w - cx, h - cy), (w - cx, h), (0, h)]


def _u_shape(rng):
    w, h = rng.uniform(8.0, 11.0), rng.uniform(6.0, 9.0)
    nw = w * rng.uniform(0.25, 0.4)
    nh = h * rng.uniform(0.35, 0.55)
    wing = (w - nw) / 2
    return [(0, 0), (w, 0), (w, h), (w - wing, h), (w - wing, h - nh),
            (wing, h - nh), (wing, h), (0, h)]


def _t_shape(rng):
    top_w = rng.uniform(8.0, 11.0)
    top_h = rng.uniform(2.5, 3.5)
    stem_w = top_w * rng.uniform(0.35, 0.5)
    stem_h = rng.uniform(4.0, 6.0)
    sx = (top_w - stem_w) / 2
    return [(sx, 0), (sx + stem_w, 0), (sx + stem_w, stem_h), (top_w, stem_h),
            (top_w, stem_h + top_h), (0, stem_h + top_h), (0, stem_h),
            (sx, stem_h)]


def _circle(rng, n=16):   # 16-gon: edges 1.5-1.8m fit an entry door (24-gon edges ~1.1m are too short)
    r = rng.uniform(3.8, 4.6)
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def _pentagon(rng):
    base = rng.uniform(4.0, 5.2)
    pts = []
    for i in range(5):
        a = 2 * math.pi * i / 5 + rng.uniform(-0.12, 0.12)
        r = base * rng.uniform(0.8, 1.2)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


SHAPES = ["rect", "l_shape", "u_shape", "t_shape", "circle", "pentagon"]


# BSP (re-derived from the first-generation generator)

def _bsp(rng, x0, y0, x1, y1, depth, max_depth, axis, min_area=16.0):
    w, h = x1 - x0, y1 - y0
    if depth >= max_depth or w * h < min_area * 1.6:
        return [(x0, y0, x1, y1)]

    def try_split(ax):
        if ax == 0 and w >= 4.5:
            c = x0 + w * rng.uniform(0.38, 0.62)
            if (c - x0) * h >= min_area and (x1 - c) * h >= min_area:
                return (_bsp(rng, x0, y0, c, y1, depth + 1, max_depth, 1, min_area)
                        + _bsp(rng, c, y0, x1, y1, depth + 1, max_depth, 1, min_area))
        if ax == 1 and h >= 4.5:
            c = y0 + h * rng.uniform(0.38, 0.62)
            if w * (c - y0) >= min_area and w * (y1 - c) >= min_area:
                return (_bsp(rng, x0, y0, x1, c, depth + 1, max_depth, 0, min_area)
                        + _bsp(rng, x0, c, x1, y1, depth + 1, max_depth, 0, min_area))
        return None
    out = try_split(axis) or try_split(1 - axis)
    return out if out is not None else [(x0, y0, x1, y1)]


# adjacency + spanning tree

def _rect_adjacency(rects, tol=1e-6, min_span=1.1):
    """Pairwise shared wall spans between axis-aligned rooms.
    Returns edges: (i, j, axis, c, s0, s1) — rooms i,j share the line
    {axis=c} over interval [s0, s1] (span >= min_span)."""
    edges = []
    for i in range(len(rects)):
        ax0, ay0, ax1, ay1 = rects[i]
        for j in range(i + 1, len(rects)):
            bx0, by0, bx1, by1 = rects[j]
            if abs(ax1 - bx0) < tol or abs(bx1 - ax0) < tol:      # share x-line
                c = ax1 if abs(ax1 - bx0) < tol else bx1
                s0, s1 = max(ay0, by0), min(ay1, by1)
                if s1 - s0 >= min_span:
                    edges.append((i, j, "x", c, s0, s1))
            if abs(ay1 - by0) < tol or abs(by1 - ay0) < tol:      # share y-line
                c = ay1 if abs(ay1 - by0) < tol else by1
                s0, s1 = max(ax0, bx0), min(ax1, bx1)
                if s1 - s0 >= min_span:
                    edges.append((i, j, "y", c, s0, s1))
    return edges


def _max_span_tree(n_rooms, edges):
    """Kruskal, widest shared span first -> spanning tree edge indices."""
    order = sorted(range(len(edges)), key=lambda k: -(edges[k][5] - edges[k][4]))
    parent = list(range(n_rooms))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    tree = []
    for k in order:
        i, j = edges[k][0], edges[k][1]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append(k)
    return tree


# function typing

def assign_functions(rng, areas, corridor_idx=None):
    """living guaranteed on the largest non-corridor room; bedrooms ~ total
    area; the rest dining/study by size."""
    n = len(areas)
    funcs = [None] * n
    # a single big room is an open-plan studio (sleep/living/kitchenette
    # zones coexist via layout._studio_zones), not a mono-function hall.
    # Small single rooms stay "living".
    if n == 1 and corridor_idx is None:
        if areas[0] >= 18.0 and rng.random() < 0.85:
            return ["studio"]
        return ["living"]
    if corridor_idx is not None:
        funcs[corridor_idx] = "hallway"
    order = sorted((k for k in range(n) if funcs[k] is None),
                   key=lambda k: -areas[k])
    funcs[order[0]] = "living"
    rest = order[1:]
    total = sum(areas)
    n_bed = max(1 if rest else 0, min(len(rest), int(round(total / 60.0))))
    for k in rest[:n_bed]:
        funcs[k] = "bedroom"
    pool = list(rest[n_bed:])
    # bathroom / kitchen room types. Real-home logic: the
    # smallest leftover room is the bath (if plausibly bath-sized); a
    # mid-size one (~8-16 m2) becomes the kitchen. Probabilities keep
    # variety (some dwellings legitimately have no separate kitchen room).
    if pool and rng.random() < 0.85:
        k_bath = min(pool, key=lambda k: areas[k])
        if areas[k_bath] <= 14.0:   # default BSP min_area=16 -> shapes/n-rooms plans supply the small rooms
            funcs[k_bath] = "bathroom"
            pool.remove(k_bath)
    if pool and rng.random() < 0.80:
        k_kit = min(pool, key=lambda k: abs(areas[k] - 11.0))
        funcs[k_kit] = "kitchen"
        pool.remove(k_kit)
    for k in pool:
        funcs[k] = rng.choice(["dining", "study", "bedroom"])
    return funcs


# light schedule

_LIGHT_TABLE = {
    #  function: (cct_lo, cct_hi, lm_per_m2_lo, lm_per_m2_hi, fixture_weights)
    # lm/m2 minima >= 70 = installed-capacity floor (residential lighting
    # codes specify exactly this); the evening illuminance floor (spec.py)
    # can only switch on lamps that exist, and a night outer room capped
    # below 62 lm/m2 stays under the brightness threshold.
    "living":  (3000, 4000, 90, 160, {"pendant": 0.45, "flush": 0.45, "bulb": 0.10}),
    "bedroom": (2700, 3300, 72, 110, {"pendant": 0.20, "flush": 0.70, "bulb": 0.10}),
    "dining":  (2700, 3500, 100, 180, {"pendant": 0.70, "flush": 0.25, "bulb": 0.05}),
    "study":   (3800, 5000, 120, 200, {"pendant": 0.25, "flush": 0.65, "bulb": 0.10}),
    "hallway": (3000, 4000, 72, 100, {"pendant": 0.10, "flush": 0.80, "bulb": 0.10}),
    # wet rooms: cool bright baths, warm-neutral kitchens
    "bathroom": (3500, 4500, 110, 180, {"pendant": 0.05, "flush": 0.85, "bulb": 0.10}),
    "kitchen":  (3500, 4500, 130, 220, {"pendant": 0.35, "flush": 0.60, "bulb": 0.05}),
    # open-plan studio: living/kitchen mix
    "studio":   (2800, 4000, 95, 170, {"pendant": 0.40, "flush": 0.50, "bulb": 0.10}),
}


def light_schedule(rng, room_name, function, poly_area, interior_pts):
    """Luminaires per room: 1 per ~12 m^2 (with a single point source, the
    corners of a 21 m^2 room fall under the brightness threshold);
    fixture/CCT randomized per function. Per-lamp jitter is renormalized so
    the room's installed capacity (area x lm/m2 draw) holds exactly
    (unnormalized +-20% jitter would break the capacity floor)."""
    lo, hi, lm0, lm1, wts = _LIGHT_TABLE[function]
    n = max(1, round(poly_area / 12.0))
    total_lm = poly_area * rng.uniform(lm0, lm1)
    kinds = list(wts)
    mults = [rng.uniform(0.8, 1.2) for _ in range(n)]
    msum = sum(mults)
    lights = []
    for k in range(n):
        r = rng.random(); acc = 0.0; fixture = kinds[-1]
        for name in kinds:
            acc += wts[name]
            if r <= acc:
                fixture = name
                break
        x, y = interior_pts[k % len(interior_pts)]
        if k >= len(interior_pts):
            # wrap guard: never place two luminaires on the same coordinate
            # (coincident shade solids render exact-zero black)
            x += rng.uniform(-0.7, 0.7)
            y += rng.uniform(-0.7, 0.7)
        lights.append({"room": room_name, "fixture": fixture, "x": x, "y": y,
                       "lumens": total_lm * mults[k] / msum,
                       "cct_k": rng.uniform(lo, hi)})
    return lights


# exact-count room split
def bsp_n_rooms(rng, W, D, n, min_side=2.6, min_area=9.0):
    """Exact-count room split: keep splitting the largest rect until there are
    n rooms (split-largest-first; the plain depth-limited BSP cannot target a
    count). Guards: every room keeps side >= min_side and area >= min_area, so
    doors/windows/furnishing stay legal. Returns < n rects only if the
    footprint physically cannot host n (caller sizes W*D ~ n * 14-20 m^2)."""
    rects = [(0.0, 0.0, W, D)]
    while len(rects) < n:
        # largest splittable rect first
        order = sorted(range(len(rects)),
                       key=lambda i: -(rects[i][2] - rects[i][0])
                       * (rects[i][3] - rects[i][1]))
        done = False
        for i in order:
            x0, y0, x1, y1 = rects[i]
            w, h = x1 - x0, y1 - y0
            axes = []
            if w >= 2 * min_side and (w * h) >= 2 * min_area:
                axes.append(0)
            if h >= 2 * min_side and (w * h) >= 2 * min_area:
                axes.append(1)
            rng.shuffle(axes)
            for ax in axes:
                span = w if ax == 0 else h
                lo, hi = min_side, span - min_side
                if hi <= lo:
                    continue
                for _ in range(8):
                    c = rng.uniform(lo, hi)
                    a_ok = (c * h if ax == 0 else w * c) >= min_area
                    b_ok = ((span - c) * h if ax == 0
                            else w * (span - c)) >= min_area
                    if a_ok and b_ok:
                        if ax == 0:
                            rects[i] = (x0, y0, x0 + c, y1)
                            rects.append((x0 + c, y0, x1, y1))
                        else:
                            rects[i] = (x0, y0, x1, y0 + c)
                            rects.append((x0, y0 + c, x1, y1))
                        done = True
                        break
                if done:
                    break
            if done:
                break
        if not done:
            break                                   # nothing splittable left
    return rects
