import time
from typing import Sequence

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Tuple, Dict, Optional
import heapq

from tools.utils import global_log_debug_df, get_stop_string
from tools.utils_opengate import get_global_log
from tools.pixelHits import ENERGY_keV, TOA
from tools.pixelClusters import PIX_X_ID, PIX_Y_ID, SIZE, DELTA_TOA

global_log = get_global_log()


# ----------------------------
# Quadtree for fast XY queries
# ----------------------------
class QuadNode:
    __slots__ = ("xmin", "ymin", "xmax", "ymax", "points", "div", "nw", "ne", "sw",
                 "se", "cap")

    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int, cap: int = 8):
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        self.points: List[Tuple[int, int]] = []
        self.div = False
        self.nw = self.ne = self.sw = self.se = None
        self.cap = cap

    def _inside(self, x: int, y: int) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def _intersects(self, rxmin: int, rymin: int, rxmax: int, rymax: int) -> bool:
        return not (
                rxmax < self.xmin or rxmin > self.xmax or rymax < self.ymin or rymin > self.ymax)

    def _subdivide(self):
        mx = (self.xmin + self.xmax) // 2
        my = (self.ymin + self.ymax) // 2
        self.nw = QuadNode(self.xmin, my + 1, mx, self.ymax, self.cap)
        self.ne = QuadNode(mx + 1, my + 1, self.xmax, self.ymax, self.cap)
        self.sw = QuadNode(self.xmin, self.ymin, mx, my, self.cap)
        self.se = QuadNode(mx + 1, self.ymin, self.xmax, my, self.cap)
        self.div = True
        # Reinsert existing points
        old = self.points
        self.points = []
        for px, py in old:
            self._insert_into_children(px, py)

    def _insert_into_children(self, x: int, y: int):
        if self.nw._inside(x, y):
            self.nw.insert(x, y)
        elif self.ne._inside(x, y):
            self.ne.insert(x, y)
        elif self.sw._inside(x, y):
            self.sw.insert(x, y)
        elif self.se._inside(x, y):
            self.se.insert(x, y)
        else:
            # Fallback: if bounds too tight, keep at this node
            self.points.append((x, y))

    def insert(self, x: int, y: int):
        if not self._inside(x, y):
            return False
        if not self.div and len(self.points) < self.cap:
            self.points.append((x, y))
            return True
        if not self.div:
            self._subdivide()
        self._insert_into_children(x, y)
        return True

    def query_rect(self, rxmin: int, rymin: int, rxmax: int, rymax: int) -> bool:
        # Returns True if any point intersects rectangle (early-exit)
        if not self._intersects(rxmin, rymin, rxmax, rymax):
            return False
        for (x, y) in self.points:
            if rxmin <= x <= rxmax and rymin <= y <= rymax:
                return True
        if self.div:
            return (
                    self.nw.query_rect(rxmin, rymin, rxmax, rymax) or
                    self.ne.query_rect(rxmin, rymin, rxmax, rymax) or
                    self.sw.query_rect(rxmin, rymin, rxmax, rymax) or
                    self.se.query_rect(rxmin, rymin, rxmax, rymax)
            )
        return False


@dataclass
class Cluster:
    cid: int
    qt: "QuadNode"
    xs: List[int] = field(default_factory=list)
    ys: List[int] = field(default_factory=list)
    toas: List[float] = field(default_factory=list)
    es: List[Optional[float]] = field(default_factory=list)  # optional energy per pixel
    first_toa: float = float("inf")
    last_toa: float = float("-inf")

    @property
    def size(self) -> int:
        return len(self.xs)

    def can_be_added(self, x: int, y: int) -> bool:
        return self.qt.query_rect(x - 1, y - 1, x + 1, y + 1)

    def add_pixel(self, x: int, y: int, toa: float, e: Optional[float] = None):
        self.xs.append(x)
        self.ys.append(y)
        self.toas.append(toa)
        self.es.append(e)
        if toa < self.first_toa:
            self.first_toa = toa
        if toa > self.last_toa:
            self.last_toa = toa
        self.qt.insert(x, y)

    def merge_from(self, other: "Cluster"):
        for x, y, t, e in zip(other.xs, other.ys, other.toas, other.es):
            self.xs.append(x)
            self.ys.append(y)
            self.toas.append(t)
            self.es.append(e)
            self.qt.insert(x, y)
        if other.first_toa < self.first_toa:
            self.first_toa = other.first_toa
        if other.last_toa > self.last_toa:
            self.last_toa = other.last_toa


class PixelProcessor:
    def __init__(self, window: float = 100,
                 world_bounds: Tuple[int, int, int, int] = (-(1 << 20), -(1 << 20),
                                                            (1 << 20), (1 << 20))):
        self.window = window
        self.world_bounds = world_bounds
        self.open_clusters: Dict[int, Cluster] = {}
        self._next_cid = 1
        self._heap: List[Tuple[float, int, int]] = []  # (last_toa, cid, ver)
        self._version: Dict[int, int] = {}

    def _new_cluster(self) -> Cluster:
        cid = self._next_cid;
        self._next_cid += 1
        qt = QuadNode(*self.world_bounds)
        c = Cluster(cid=cid, qt=qt)
        self.open_clusters[cid] = c
        self._version[cid] = 0
        return c

    def _touch_heap(self, cid: int):
        v = self._version.get(cid, 0) + 1
        self._version[cid] = v
        c = self.open_clusters[cid]
        heapq.heappush(self._heap, (c.last_toa, cid, v))

    def _close_ready(self, current_toa: float) -> Iterator[Dict]:
        threshold = current_toa - self.window
        while self._heap and self._heap[0][0] < threshold:
            last_toa, cid, ver = heapq.heappop(self._heap)
            c = self.open_clusters.get(cid)
            if c is None:
                continue
            if self._version.get(cid, -1) != ver or c.last_toa != last_toa:
                continue
            del self.open_clusters[cid]
            self._version.pop(cid, None)
            yield {
                "id": cid,
                "x": c.xs, "y": c.ys, "toa": c.toas, "e": c.es,
                "first_toa": c.first_toa, "last_toa": c.last_toa,
            }

    def process_stream(self, pixels: Iterable[Sequence]) -> Iterator[Dict]:
        """
        pixels: iterable of (x, y, ToA[, E]). Input can be partially unsorted.
        """
        sorted_pixels = sorted(pixels, key=lambda p: p[2])
        for row in sorted_pixels:
            x, y, toa = int(row[0]), int(row[1]), float(row[2])
            e = float(row[3]) if len(row) > 3 and row[3] is not None else None

            # 1) Close too-old clusters BEFORE matching the current pixel
            for closed in self._close_ready(toa):
                yield closed

            # 2) Find spatially adjacent candidates that are also time-valid
            candidates: List[Cluster] = []
            for c in self.open_clusters.values():
                # Temporal guard (redundant once we close first, but safe)
                if c.last_toa < toa - self.window:
                    continue
                if c.can_be_added(x, y):
                    candidates.append(c)

            if candidates:
                candidates.sort(key=lambda c: c.size, reverse=True)
                last_cluster = candidates[0]
                for other in candidates[1:]:
                    if other.cid == last_cluster.cid:
                        continue
                    last_cluster.merge_from(other)
                    self.open_clusters.pop(other.cid, None)
                    self._version.pop(other.cid, None)
                last_cluster.add_pixel(x, y, toa, e)
                self._touch_heap(last_cluster.cid)
            else:
                last_cluster = self._new_cluster()
                last_cluster.add_pixel(x, y, toa, e)
                self._touch_heap(last_cluster.cid)

        # Flush remaining clusters
        while self._heap:
            last_toa, cid, ver = heapq.heappop(self._heap)
            c = self.open_clusters.get(cid)
            if c is None or self._version.get(cid, -1) != ver or c.last_toa != last_toa:
                continue
            del self.open_clusters[cid]
            self._version.pop(cid, None)
            yield {
                "id": cid,
                "x": c.xs, "y": c.ys, "toa": c.toas, "e": c.es,
                "first_toa": c.first_toa, "last_toa": c.last_toa,
            }


# ----------------------------
# Convenience API
# ----------------------------
def process_pixels(
        pixels: Iterable[Tuple[int, int, float]],
        window: float = 100,
        world_bounds: Tuple[int, int, int, int] = (-(1 << 20), -(1 << 20), (1 << 20),
                                                   (1 << 20)),
) -> List[Dict]:
    """
    Collect all clusters for a finite input stream of (x, y, ToA).
    """
    pp = PixelProcessor(window=window, world_bounds=world_bounds)
    return list(pp.process_stream(pixels))


def to_pixelCluster_df(clusters):
    """
    Build a DataFrame with columns: X, Y, Energy (keV), ToA (ns), size, delta_toa
    - X, Y: energy-weighted centroid over pixels with valid energy
    - Energy (keV): sum of pixel energies
    - ToA (ns): cluster first_toa
    - size: number of pixels in cluster
    - delta_toa: last_toa - first_toa
    Fallback: if no valid energies, use unweighted centroid and energy 0.0.
    """
    recs = []
    for c in clusters:
        xs = c.get('x', []) or []
        ys = c.get('y', []) or []
        es = c.get('e', []) or []

        sum_e = 0.0
        wx = 0.0
        wy = 0.0
        for x, y, e in zip(xs, ys, es):
            if e is None:
                continue
            try:
                fe = float(e)
            except Exception:
                continue
            if fe != fe or fe == float('inf') or fe == float('-inf'):
                continue
            sum_e += fe
            wx += fe * float(x)
            wy += fe * float(y)

        if sum_e > 0.0:
            cx = wx / sum_e
            cy = wy / sum_e
            energy = sum_e
        else:
            n = len(xs)
            cx = (sum(float(x) for x in xs) / n) if n else float('nan')
            cy = (sum(float(y) for y in ys) / n) if n else float('nan')
            energy = 0.0

        first_toa = float(c.get('first_toa', float('nan')))
        last_toa = float(c.get('last_toa', float('nan')))
        size = len(xs)
        delta_toa = last_toa - first_toa if size > 1 else float('nan')

        recs.append({
            PIX_X_ID: cx,
            PIX_Y_ID: cy,
            ENERGY_keV: energy,
            TOA: first_toa,
            SIZE: size,
            DELTA_TOA: delta_toa,
        })

    df = pd.DataFrame(recs)
    if not df.empty:
        df.sort_values(['ToA (ns)'], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

def df_to_xyte_array(
        df,
        n_pixels: int,
        pid_col: str = 'PixelID (int16)',
        toa_col: str = 'ToA (ns)',
        energy_col: str = 'Energy (keV)',
) -> np.ndarray:
    # PixelID -> X,Y (vectorized)
    pid = df[pid_col].to_numpy(dtype=np.int32, copy=False)
    x = pid // np.int32(n_pixels)
    y = pid % np.int32(n_pixels)

    # ToA and Energy
    t = df[toa_col].to_numpy(copy=False)
    e = df[energy_col].to_numpy(copy=False)

    # Assemble into a compact 2D array (will be float64; cast if needed)
    out = np.empty((pid.shape[0], 4), dtype=np.float64)
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = t
    out[:, 3] = e

    return out


def pixelHits2pixelClusters(pixelHits_df, window_ns, npix):
    stime = time.time()
    global_log.info(f"Offline [pixelClusters]: START")

    if pixelHits_df.empty:
        global_log.error(
            "Offline [pixelClusters]: Empty pixel hits dataframe, probably no hit produced.")
        global_log.info(f"Offline [pixelClusters]: {get_stop_string(stime)}")
        return []

    global_log.debug(f"Input pixel hits dataframe with {len(pixelHits_df)} entries")

    hits_meas_array = df_to_xyte_array(pixelHits_df, npix)
    clusters = process_pixels(hits_meas_array, window=window_ns)
    clusters_centroid_df = to_pixelCluster_df(clusters)

    global_log.debug(f"{len(clusters_centroid_df)} clusters")
    global_log_debug_df(clusters_centroid_df)
    global_log.info(f"Offline [pixelClusters]: {get_stop_string(stime)}")
    return clusters_centroid_df
