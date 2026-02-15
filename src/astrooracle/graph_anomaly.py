from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphAnomalyResult:
    n_nodes: int
    n_edges: int
    communities: Dict[str, int]
    node_metrics: pd.DataFrame
    edge_bridges: pd.DataFrame
    meta: Dict[str, float]


def _radec_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=1)


def _great_circle_distance_rad(xyz_a: np.ndarray, xyz_b: np.ndarray) -> np.ndarray:
    # angle between unit vectors.
    dot = np.clip((xyz_a * xyz_b).sum(axis=1), -1.0, 1.0)
    return np.arccos(dot)


def build_knn_graph(
    df: pd.DataFrame,
    *,
    k: int = 10,
    metric: str = "greatcircle",
    max_nodes: int = 5000,
    seed: int = 7,
) -> Tuple["nx.Graph", pd.DataFrame]:
    """Build a kNN graph over candidates on the celestial sphere.

    Returns (G, nodes_df). nodes_df includes xyz coords.
    """

    import networkx as nx

    if df.empty:
        return nx.Graph(), df

    d = df.copy().reset_index(drop=True)
    if len(d) > max_nodes:
        d = d.sample(n=max_nodes, random_state=seed).reset_index(drop=True)

    ra = pd.to_numeric(d["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(d["dec"], errors="coerce").to_numpy(float)
    xyz = _radec_to_xyz(ra, dec)

    # Use sklearn NearestNeighbors in 3D Euclidean space (equivalent to angular for unit vectors).
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(int(k) + 1, len(d)), metric="euclidean")
        nn.fit(xyz)
        dist, idx = nn.kneighbors(xyz)
    except Exception:
        # Fallback O(n^2).
        n = xyz.shape[0]
        dist = np.full((n, min(int(k) + 1, n)), np.nan, dtype=float)
        idx = np.full((n, min(int(k) + 1, n)), -1, dtype=int)
        for i in range(n):
            dot = np.clip(xyz @ xyz[i], -1.0, 1.0)
            ang = np.arccos(dot)
            order = np.argsort(ang)
            sel = order[: dist.shape[1]]
            dist[i, :] = ang[sel]
            idx[i, :] = sel

    G = nx.Graph()
    for i, row in d.iterrows():
        G.add_node(str(row["id"]), ra=float(row["ra"]), dec=float(row["dec"]), anomaly_score=float(row["anomaly_score"]))

    for i in range(idx.shape[0]):
        src = str(d.loc[i, "id"])
        for jpos in range(1, idx.shape[1]):
            j = int(idx[i, jpos])
            if j < 0 or j >= idx.shape[0]:
                continue
            dst = str(d.loc[j, "id"])
            if src == dst:
                continue
            w = float(dist[i, jpos])
            # Store angular distance in radians.
            if not G.has_edge(src, dst):
                G.add_edge(src, dst, weight=w)

    d = d.copy()
    d["x"] = xyz[:, 0]
    d["y"] = xyz[:, 1]
    d["z"] = xyz[:, 2]
    return G, d


def detect_communities(G: "nx.Graph") -> Dict[str, int]:
    """Community assignment (Louvain if available, else greedy modularity)."""

    import networkx as nx

    if G.number_of_nodes() == 0:
        return {}

    # Louvain (python-louvain)
    try:
        import community as community_louvain  # type: ignore

        part = community_louvain.best_partition(G, weight="weight")
        return {str(k): int(v) for k, v in part.items()}
    except Exception:
        pass

    # Greedy modularity (networkx)
    comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    out: Dict[str, int] = {}
    for i, cset in enumerate(comms):
        for node in cset:
            out[str(node)] = int(i)
    return out


def graph_anomaly(
    df: pd.DataFrame,
    *,
    k: int = 10,
    max_nodes: int = 5000,
    bridges_top: int = 50,
) -> GraphAnomalyResult:
    """Compute graph-based anomaly context and return node and edge diagnostics."""

    import networkx as nx

    G, nodes_df = build_knn_graph(df, k=k, max_nodes=max_nodes)
    comm = detect_communities(G)

    if G.number_of_nodes() == 0:
        empty = pd.DataFrame()
        return GraphAnomalyResult(
            n_nodes=0,
            n_edges=0,
            communities={},
            node_metrics=empty,
            edge_bridges=empty,
            meta={"k": float(k), "max_nodes": float(max_nodes)},
        )

    # Node-level metrics.
    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G, weight="weight")
    clo = nx.closeness_centrality(G, distance="weight")

    # Edge betweenness as bridge score.
    ebet = nx.edge_betweenness_centrality(G, weight="weight")

    node_rows = []
    for n in G.nodes:
        node_rows.append(
            {
                "id": str(n),
                "degree": float(deg.get(n, 0)),
                "betweenness": float(bet.get(n, 0.0)),
                "closeness": float(clo.get(n, 0.0)),
                "community": int(comm.get(str(n), -1)),
            }
        )

    node_metrics = pd.DataFrame(node_rows)

    edge_rows = []
    for (u, v), score in ebet.items():
        w = float(G.edges[u, v].get("weight", math.nan))
        edge_rows.append({"u": str(u), "v": str(v), "edge_betweenness": float(score), "distance_rad": w})

    edge_bridges = pd.DataFrame(edge_rows)
    if not edge_bridges.empty:
        edge_bridges = edge_bridges.sort_values("edge_betweenness", ascending=False).head(int(bridges_top)).reset_index(drop=True)

    return GraphAnomalyResult(
        n_nodes=int(G.number_of_nodes()),
        n_edges=int(G.number_of_edges()),
        communities=comm,
        node_metrics=node_metrics,
        edge_bridges=edge_bridges,
        meta={"k": float(k), "max_nodes": float(max_nodes), "bridges_top": float(bridges_top)},
    )


def plot_graph_context(
    df: pd.DataFrame,
    *,
    k: int = 10,
    max_nodes: int = 2000,
    title: str = "Graph anomaly context (kNN on sphere)",
):
    """Return a Plotly figure (2D RA/Dec) colored by community."""

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotly not installed") from e

    res = graph_anomaly(df, k=k, max_nodes=max_nodes)
    if res.n_nodes == 0:
        return px.scatter(title="No candidates")

    d = df.copy()
    d["id"] = d["id"].astype(str)
    d = d.merge(res.node_metrics[["id", "community", "betweenness"]], on="id", how="left")

    d['_node_size'] = np.clip(d['betweenness'].fillna(0.0).to_numpy(float) * 50.0 + 3.0, 3.0, 25.0)

    fig = px.scatter(
        d,
        x='ra',
        y='dec',
        color='community',
        size='_node_size',
        hover_data=['id', 'anomaly_score', 'community', 'betweenness'],
        title=title,
    )

    return fig
