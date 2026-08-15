"""Affinity slotting: cluster SKUs that travel together, then place the clusters.

Velocity slotting minimises the depot-anchored part of the tour. It says
nothing about the part that dominates a multi-line pick: the walking *between*
picks. Two SKUs that appear in the same order 400 times a period should be
close to each other even if neither is an A mover, and no amount of ABC
analysis will ever discover that, because ABC looks at one SKU at a time.

The pipeline:

1. **Co-occurrence.** Count SKU pairs appearing in the same order. Optionally
   normalise to lift so that slow pairs that *always* travel together are not
   drowned by fast pairs that happen to be everywhere.
2. **Cluster.** Two methods, deliberately different in character:

   * ``greedy`` - seeded region growing on the affinity graph, capacity capped.
     Fast, no eigenvalues, and completely explainable to an operations manager,
     which is why it is the default. Related to the "order-oriented slotting"
     family (Frazelle 2016 ch. 8).
   * ``spectral`` - k smallest eigenvectors of the symmetric normalised
     Laplacian, row-normalised, then k-means (Shi & Malik 2000; Ng, Jordan &
     Weiss 2002; von Luxburg 2007). Finds structure the greedy misses when the
     affinity graph has communities rather than hubs.

3. **Place.** Clusters are ranked by total pick rate and given compact,
   depot-near *regions* rather than contiguous distance bands. That distinction
   matters: locations equidistant from the depot can be in different aisles, and
   two locations in different aisles are far apart even when both are 20 m from
   the dock. Region growing on the bay-column adjacency keeps a cluster inside
   one or two aisles, which is where the intra-order travel saving comes from.

The honest caveat is in the benchmark, not here: affinity slotting only pays
when orders are multi-line. :meth:`OrderStream.single_line_share` is reported
alongside the result for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assignment import Assignment, ConstraintModel, greedy_place
from .layout import Warehouse
from .objective import build_affinity_pairs
from .orders import OrderStream

__all__ = [
    "AffinityGraph",
    "greedy_clusters",
    "spectral_clusters",
    "affinity_slotting",
    "cluster_quality",
]


class AffinityGraph:
    """Sparse symmetric SKU-pair affinity, with the adjacency views clustering needs."""

    def __init__(self, n_skus: int, pairs: list[tuple[int, int, float]]) -> None:
        self.n_skus = n_skus
        self.pairs = list(pairs)
        self.adjacency: dict[int, dict[int, float]] = {}
        for i, j, w in self.pairs:
            self.adjacency.setdefault(i, {})[j] = w
            self.adjacency.setdefault(j, {})[i] = w
        self.degree = np.zeros(n_skus, dtype=float)
        for i, nbrs in self.adjacency.items():
            self.degree[i] = sum(nbrs.values())

    @classmethod
    def from_orders(
        cls,
        stream: OrderStream,
        n_skus: int,
        top_k: int = 12,
        min_count: int = 2,
        normalise: str = "count",
    ) -> "AffinityGraph":
        return cls(n_skus, build_affinity_pairs(stream, n_skus, top_k, min_count, normalise))

    def dense(self) -> np.ndarray:
        m = np.zeros((self.n_skus, self.n_skus), dtype=float)
        for i, j, w in self.pairs:
            m[i, j] = w
            m[j, i] = w
        return m

    def total_weight(self) -> float:
        return float(sum(w for _, _, w in self.pairs))


# ----------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------
def greedy_clusters(graph: AffinityGraph, max_size: int = 24) -> list[list[int]]:
    """Seeded region growing on the affinity graph, capacity capped.

    Seeds are taken in descending total affinity degree, so the SKUs with the
    most co-ordering evidence get to define clusters before the noise does. A
    cluster grows by repeatedly absorbing the unassigned SKU with the largest
    total affinity to the cluster so far - the natural greedy for maximising
    within-cluster weight - and stops at ``max_size`` or when no unassigned SKU
    has any affinity left to it.

    ``max_size`` is not a modelling nicety. Unbounded, the greedy produces one
    giant cluster containing everything the fast movers ever co-occurred with,
    which is the whole catalogue, and the placement step then has nothing to say.
    Capping at roughly the number of slots in two bay columns keeps clusters at a
    size the geometry can actually honour.
    """
    assigned = np.zeros(graph.n_skus, dtype=bool)
    order = np.argsort(-graph.degree, kind="stable")
    clusters: list[list[int]] = []

    for seed in order:
        seed = int(seed)
        if assigned[seed]:
            continue
        cluster = [seed]
        assigned[seed] = True
        # affinity from each candidate to the current cluster
        frontier: dict[int, float] = {}
        for nbr, w in graph.adjacency.get(seed, {}).items():
            if not assigned[nbr]:
                frontier[nbr] = frontier.get(nbr, 0.0) + w
        while len(cluster) < max_size and frontier:
            best = max(frontier.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            del frontier[best]
            if assigned[best]:
                continue
            assigned[best] = True
            cluster.append(best)
            for nbr, w in graph.adjacency.get(best, {}).items():
                if not assigned[nbr]:
                    frontier[nbr] = frontier.get(nbr, 0.0) + w
        clusters.append(cluster)

    for sku in range(graph.n_skus):
        if not assigned[sku]:
            clusters.append([sku])
            assigned[sku] = True
    return clusters


def spectral_clusters(
    graph: AffinityGraph,
    n_clusters: int = 40,
    seed: int = 5,
    epsilon: float = 1e-3,
) -> list[list[int]]:
    """Normalised-cut spectral clustering on the affinity graph.

    Implemented from the definition rather than pulled from a clustering
    library, because the two details that decide whether it works are both in
    the plumbing: the graph is disconnected (most SKU pairs never co-occur) so a
    small uniform ``epsilon`` is added to keep the Laplacian's null space
    one-dimensional, and the eigenvector rows are normalised to the unit sphere
    before k-means, which is the Ng-Jordan-Weiss variant and is materially more
    stable than the unnormalised one here.
    """
    from sklearn.cluster import KMeans

    n = graph.n_skus
    n_clusters = max(1, min(n_clusters, n))
    w = graph.dense()
    if w.sum() > 0:
        w = w / w.max()
    w += epsilon
    np.fill_diagonal(w, 0.0)

    d = w.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    lap = np.eye(n) - (w * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    lap = (lap + lap.T) / 2.0

    vals, vecs = np.linalg.eigh(lap)
    embedding = vecs[:, :n_clusters]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norms, 1e-12)

    labels = KMeans(n_clusters=n_clusters, n_init=8, random_state=seed).fit_predict(embedding)
    clusters: dict[int, list[int]] = {}
    for sku, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(int(sku))
    return [clusters[k] for k in sorted(clusters)]


def cluster_quality(graph: AffinityGraph, clusters: list[list[int]]) -> dict[str, float]:
    """Share of affinity weight captured inside clusters, and size statistics.

    The captured share is the number to look at. If clustering captures 15% of
    the co-occurrence weight, the placement step can only ever act on 15% of the
    inter-pick travel, and the affinity result should be read against that
    ceiling rather than against zero.
    """
    label = np.full(graph.n_skus, -1, dtype=np.int64)
    for c, members in enumerate(clusters):
        for m in members:
            label[m] = c
    inside = total = 0.0
    for i, j, w in graph.pairs:
        total += w
        if label[i] == label[j] and label[i] >= 0:
            inside += w
    sizes = np.asarray([len(c) for c in clusters], dtype=float)
    return {
        "n_clusters": float(len(clusters)),
        "captured_weight_share": inside / total if total else 0.0,
        "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
        "max_size": float(sizes.max()) if len(sizes) else 0.0,
        "singletons": float((sizes == 1).sum()),
    }


# ----------------------------------------------------------------------
# placement
# ----------------------------------------------------------------------
def _column_regions(warehouse: Warehouse) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Group locations into (aisle, bay) columns and return their depot distances."""
    c = warehouse.config
    n_cols = c.n_aisles * c.n_bays
    members: list[list[int]] = [[] for _ in range(n_cols)]
    for idx, loc in enumerate(warehouse.locations):
        members[loc.aisle * c.n_bays + loc.bay].append(idx)
    depot_d = np.asarray(
        [
            warehouse.travel_distance(
                warehouse.depot, warehouse.location_point(warehouse.locations[members[k][0]])
            )
            for k in range(n_cols)
        ]
    )
    return np.arange(n_cols), members, depot_d


def _grow_region(
    free_cols: set[int],
    n_needed: int,
    col_dist: np.ndarray,
    depot_d: np.ndarray,
    depot_pull: float,
) -> list[int]:
    """Take ``n_needed`` free columns forming a compact, depot-near region."""
    if not free_cols:
        return []
    seed = min(free_cols, key=lambda k: (depot_d[k], k))
    region = [seed]
    free_cols.discard(seed)
    while len(region) < n_needed and free_cols:
        nxt = min(
            free_cols,
            key=lambda k: (col_dist[seed, k] + depot_pull * depot_d[k], k),
        )
        region.append(nxt)
        free_cols.discard(nxt)
    return region


def affinity_slotting(
    constraints: ConstraintModel,
    stream: OrderStream,
    method: str = "greedy",
    max_cluster_size: int = 24,
    n_clusters: int = 40,
    top_k: int = 12,
    min_count: int = 2,
    normalise: str = "count",
    pick_rate: np.ndarray | None = None,
    depot_pull: float = 0.35,
    graph: AffinityGraph | None = None,
) -> tuple[Assignment, AffinityGraph, list[list[int]]]:
    """Cluster by order co-occurrence, then place clusters into compact regions.

    Returns the assignment, the affinity graph, and the clusters, because in
    practice the clusters are half the deliverable - a slotting recommendation
    that cannot be explained as "these twenty items now live together, here is
    the evidence" does not get implemented.
    """
    warehouse = constraints.warehouse
    catalog = constraints.catalog
    rate = np.asarray(pick_rate, dtype=float) if pick_rate is not None else catalog.picks

    g = graph or AffinityGraph.from_orders(
        stream, catalog.n_skus, top_k=top_k, min_count=min_count, normalise=normalise
    )
    if method == "greedy":
        clusters = greedy_clusters(g, max_size=max_cluster_size)
    elif method == "spectral":
        clusters = spectral_clusters(g, n_clusters=n_clusters)
    else:
        raise ValueError(f"unknown clustering method {method!r}")

    # Hot clusters first; within a cluster, hot SKUs first.
    cluster_rate = [float(rate[c].sum()) for c in clusters]
    cl_order = sorted(range(len(clusters)), key=lambda k: -cluster_rate[k])

    _, col_members, col_depot = _column_regions(warehouse)
    access = warehouse.access_distance_matrix()
    slots_per_col = len(col_members[0]) if col_members else 1
    free_cols = set(range(len(col_members)))

    location_order: list[int] = []
    sku_order: list[int] = []
    golden = warehouse.locations_by_depot_distance()
    golden_rank = {loc: r for r, loc in enumerate(golden)}

    for k in cl_order:
        members = sorted(clusters[k], key=lambda s: -rate[s])
        sku_order.extend(int(m) for m in members)
        # Size the region exactly. Over-allocating leaves free slots behind the
        # placement cursor, and the next cluster then fills them - which quietly
        # interleaves clusters that were supposed to be separated.
        n_needed = max(1, int(np.ceil(len(members) / max(slots_per_col, 1))))
        region = _grow_region(free_cols, n_needed, access, col_depot, depot_pull)
        slots = [loc for col in region for loc in col_members[col]]
        slots.sort(key=lambda loc: golden_rank[loc])
        location_order.extend(slots)

    # Anything unallocated (rounding slack) goes on the end in depot order.
    used = set(location_order)
    location_order.extend(loc for loc in golden if loc not in used)

    assignment = greedy_place(constraints, sku_order, location_order)
    return assignment, g, clusters
