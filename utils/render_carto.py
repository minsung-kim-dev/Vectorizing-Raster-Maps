#!/usr/bin/env python3
"""Render OpenMapTiles-schema GeoJSON in an approximation of the OSM Carto style.

Output is in Web Mercator, pixel-aligned to the standard OSM 256px tile grid at
the chosen zoom, and cropped to the GeoJSON bbox, so it can be compared pixel
for pixel with a mosaic of tile.openstreetmap.org tiles (see compare_osm.py).

Usage:
    python render_carto.py map_0001.geojson.gz [--zoom Z] [--scale 1|2] [-o out.png]
Zoom defaults to metadata.source_zoom (the zoom the vector data was cut at).
"""
import argparse, gzip, json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patheffects as pe
from matplotlib import font_manager

# ----------------------------------------------------------------------------
# Projection helpers (global pixel coords at zoom z, 256px tiles)
# ----------------------------------------------------------------------------

def lonlat_to_px(lon, lat, z):
    n = 256 * 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat = max(min(lat, 85.05112878), -85.05112878)
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def pixel_window(bbox, z):
    """Integer pixel window [x0,y0,x1,y1) covering the bbox at zoom z."""
    x0, y1 = lonlat_to_px(bbox[0], bbox[1], z)
    x1, y0 = lonlat_to_px(bbox[2], bbox[3], z)
    return math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)

# ----------------------------------------------------------------------------
# OSM Carto palette (values from openstreetmap-carto *.mss)
# ----------------------------------------------------------------------------
LAND = "#f2efe9"
WATER = "#aad3df"

AREA_COLORS = {
    # landcover (class, subclass)
    "forest": "#add19e", "wood": "#add19e",
    "farmland": "#eef0d5", "orchard": "#aedfa3", "vineyard": "#aedfa3",
    "grass": "#cdebb0", "grassland": "#cdebb0", "meadow": "#cdebb0",
    "park": "#c8facc", "village_green": "#cdebb0", "recreation_ground": "#dffce2",
    "scrub": "#c8d7ab", "heath": "#d6d99f",
    "bare_rock": "#eee5dc", "scree": "#ede4dc", "sand": "#f5e9c6", "beach": "#fff1ba",
    "wetland": LAND,  # drawn as pattern on top
    "ice": "#ddecec", "glacier": "#ddecec",
    # landuse (class)
    "residential": "#e0dfdf", "suburb": None, "neighbourhood": None,
    "commercial": "#f2dad9", "retail": "#ffd6d1", "industrial": "#ebdbe8",
    "railway": "#ebdbe8", "garages": "#dfddce", "quarry": "#c5c3c3",
    "cemetery": "#aacbaf", "military": None,
    "school": "#ffffe5", "kindergarten": "#ffffe5", "university": "#ffffe5",
    "college": "#ffffe5", "hospital": "#ffffe5", "library": "#ffffe5",
    "pitch": "#aae0cb", "stadium": "#dffce2", "track": "#aae0cb",
    "playground": "#dffce2", "zoo": None, "theme_park": None,
    "bus_station": "#e9e7e2", "construction": "#c7c7b4",
    "farmyard": "#f5dcba", "allotments": "#c9e1bf", "brownfield": "#c7c7b4",
    "dam": "#adadad",
}
AREA_OUTLINES = {  # thin outlines some Carto areas get
    "pitch": "#88c9b2", "stadium": "#88c9b2", "track": "#88c9b2",
    "cemetery": None, "school": "#e2c9a7", "kindergarten": "#e2c9a7",
    "hospital": "#e2c9a7", "theme_park": "#734a08",
}
WATER_CLASSES = {"lake", "pond", "river", "ocean", "swimming_pool", "reservoir", "basin", "dock"}

# Road styling: (fill, casing, minzoom, {zoom: (total_width, casing_width)})
ROADS = {
    "motorway":  ("#e892a2", "#dc2a67", 5,  {12: (3.5, 0.5), 13: (8.0, 0.6), 14: (10.0, 0.6)}),
    "trunk":     ("#f9b29c", "#c84e2f", 5,  {12: (3.5, 0.5), 13: (8.0, 0.6), 14: (10.0, 0.6)}),
    "primary":   ("#fcd6a4", "#a06b00", 8,  {12: (3.5, 0.5), 13: (8.0, 0.6), 14: (10.0, 0.6)}),
    "secondary": ("#f7fabf", "#707d05", 9,  {12: (3.5, 0.5), 13: (8.0, 0.6), 14: (10.0, 0.6)}),
    "tertiary":  ("#ffffff", "#8f8f8f", 10, {12: (2.5, 0.4), 13: (6.0, 0.5), 14: (9.0, 0.6)}),
    "minor":     ("#ffffff", "#bbbbbb", 12, {12: (1.0, 0.0), 13: (2.5, 0.5), 14: (3.5, 0.5)}),
    "service":   ("#ffffff", "#bbbbbb", 14, {14: (2.5, 0.5)}),
    "raceway":   ("#ffc0cb", None,      12, {12: (1.2, 0.0), 13: (2.0, 0.0), 14: (3.0, 0.0)}),
}
MINOR_Z12_COLOR = "#bbbbbb"  # at z12 Carto draws residential/unclassified as a thin grey line
ROAD_ORDER = ["raceway", "service", "minor", "tertiary", "secondary", "primary", "trunk", "motorway"]

# Feature min-zooms that mirror Carto (so the render drops what OSM would drop)
MINZOOM = {
    ("building", None): 14,
    ("waterway", "river"): 8, ("waterway", "canal"): 12, ("waterway", "stream"): 13,
    ("waterway", "ditch"): 14, ("waterway", "drain"): 14,
    ("transportation", "track"): 13, ("transportation", "path"): 13,
    ("place", "city"): 6, ("place", "town"): 9, ("place", "village"): 12,
    ("place", "hamlet"): 14, ("place", "suburb"): 12, ("place", "isolated_dwelling"): 15,
    ("place", "neighbourhood"): 15,
}

PATH_STYLE = {  # subclass -> (color, width, dash)
    "footway": ("#fa8072", 1.0, (1, 3)), "pedestrian": ("#fa8072", 1.0, (1, 3)),
    "steps": ("#fa8072", 2.0, (1, 1.5)), "cycleway": ("#0000ff", 1.0, (1, 3)),
    "bridleway": ("#008000", 1.0, (1, 3)), "path": ("#5a5a5a", 0.8, (2, 2.5)),
}

BOUNDARY_COLOR = "#8d618b"
PROTECTED_COLOR = "#008000"
LABEL_FONT = None


def pick_font():
    # Ordered fallback list: matplotlib (>=3.6) takes each glyph from the first
    # family that has it, so Latin, CJK, Korean etc. labels all render.
    # (Hiragino Sans GB is left out: it has no regular weight, so matplotlib
    # logs a weight-substitution warning; Arial Unicode MS covers its glyphs.)
    global LABEL_FONT
    fams = []
    for fam in ("Noto Sans", "DejaVu Sans", "Noto Sans CJK JP", "Hiragino Sans",
                "Apple SD Gothic Neo", "Arial Unicode MS"):
        try:
            font_manager.findfont(fam, fallback_to_default=False)
            fams.append(fam)
        except Exception:
            continue
    LABEL_FONT = fams or "sans-serif"


def zw(table, z):
    """Width tuple for zoom z from a sparse table (nearest lower key, else lowest)."""
    keys = sorted(table)
    best = keys[0]
    for k in keys:
        if k <= z:
            best = k
    return table[best]

# ----------------------------------------------------------------------------
# Geometry -> matplotlib paths
# ----------------------------------------------------------------------------

class Proj:
    def __init__(self, z, x0, y0, scale):
        self.z, self.x0, self.y0, self.s = z, x0, y0, scale

    def pts(self, coords):
        out = []
        for lon, lat, *_ in coords:
            x, y = lonlat_to_px(lon, lat, self.z)
            out.append(((x - self.x0) * self.s, (y - self.y0) * self.s))
        return out


def signed_area(p):
    a = 0.0
    for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1]):
        a += x1 * y2 - x2 * y1
    return a / 2


def polygon_path(geom, proj):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    verts, codes = [], []
    for poly in polys:
        for i, ring in enumerate(poly):
            p = proj.pts(ring)
            if len(p) < 3:
                continue
            # outer rings one orientation, holes the other (nonzero fill)
            if (signed_area(p) > 0) != (i == 0):
                p = p[::-1]
            verts += p + [p[0]]
            codes += [Path.MOVETO] + [Path.LINETO] * (len(p) - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes) if verts else None


def line_parts(geom, proj):
    if geom["type"] == "LineString":
        return [proj.pts(geom["coordinates"])]
    if geom["type"] == "MultiLineString":
        return [proj.pts(c) for c in geom["coordinates"]]
    if geom["type"] in ("Polygon", "MultiPolygon"):
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        return [proj.pts(r) for poly in polys for r in poly]
    return []


def lines_path(parts):
    verts, codes = [], []
    for p in parts:
        if len(p) < 2:
            continue
        verts += p
        codes += [Path.MOVETO] + [Path.LINETO] * (len(p) - 1)
    return Path(verts, codes) if verts else None


def path_area(path):
    if path is None:
        return 0
    ext = path.get_extents()
    return ext.width * ext.height

# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def visible(layer, cls, z):
    mz = MINZOOM.get((layer, cls), MINZOOM.get((layer, None)))
    return mz is None or z >= mz


def render(path, out, zoom=None, scale=1):
    pick_font()
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as fh:
        data = json.load(fh)
    meta = data.get("metadata", {})
    z = zoom if zoom is not None else int(meta.get("source_zoom", round(meta.get("camera_zoom", 13))))
    bbox = data["bbox"]
    x0, y0, x1, y1 = pixel_window(bbox, z)
    W, H = (x1 - x0) * scale, (y1 - y0) * scale
    proj = Proj(z, x0, y0, scale)
    S = scale  # width multiplier

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor(LAND)
    ax.set_facecolor(LAND)
    pt = 72 / 100  # px -> points at dpi=100

    feats = data["features"]
    L = lambda f: f["properties"].get("omt_layer")
    C = lambda f: f["properties"].get("class")
    SC = lambda f: f["properties"].get("subclass")

    # ---- 1. landcover + landuse areas, largest first (Carto orders by way_area) ----
    areas = []
    for f in feats:
        if L(f) not in ("landcover", "landuse"):
            continue
        key = SC(f) if L(f) == "landcover" and SC(f) in AREA_COLORS else C(f)
        color = AREA_COLORS.get(key)
        p = polygon_path(f["geometry"], proj) if f["geometry"]["type"] in ("Polygon", "MultiPolygon") else None
        if p is None:
            continue
        areas.append((path_area(p), p, key, color))
    areas.sort(key=lambda a: -a[0])
    for _, p, key, color in areas:
        if color:
            ax.add_patch(PathPatch(p, facecolor=color, edgecolor="none", lw=0, zorder=1))
        oc = AREA_OUTLINES.get(key)
        if oc and z >= 13:
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor=oc, lw=0.6 * S * pt, zorder=1.1))
        if key == "wetland":
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor="#4aa5fa", lw=0, hatch="-", alpha=0.45, zorder=1.2))

    # ---- 2. water polygons ----
    for f in feats:
        if L(f) == "water" and f["geometry"]["type"] in ("Polygon", "MultiPolygon"):
            p = polygon_path(f["geometry"], proj)
            if p is not None:
                ax.add_patch(PathPatch(p, facecolor=WATER, edgecolor="none", lw=0, zorder=2))

    # ---- 3. waterway lines ----
    ww_w = {"river": {8: 0.7, 12: 2.0, 13: 2.6, 14: 4.0}, "canal": {12: 2.0, 13: 2.6, 14: 4.0},
            "stream": {13: 1.0, 14: 1.6}, "ditch": {14: 0.8}, "drain": {14: 0.8}}
    for f in feats:
        if L(f) != "waterway" or not visible("waterway", C(f), z):
            continue
        w = zw(ww_w.get(C(f), {13: 1.0}), z)
        p = lines_path(line_parts(f["geometry"], proj))
        if p is not None:
            ls = (0, (4, 2)) if f["properties"].get("intermittent") == 1 else "solid"
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor=WATER, lw=w * S * pt, linestyle=ls,
                                   capstyle="round", joinstyle="round", zorder=3))

    # ---- 4. buildings (z14+ in Carto) ----
    if visible("building", None, z):
        for f in feats:
            if L(f) == "building":
                p = polygon_path(f["geometry"], proj)
                if p is not None:
                    ax.add_patch(PathPatch(p, facecolor="#d9d0c9", edgecolor="#c4b6ab", lw=0.5 * S * pt, zorder=4))

    # ---- 5. protected areas (park layer) ----
    for f in feats:
        if L(f) == "park" and f["geometry"]["type"] in ("Polygon", "MultiPolygon"):
            p = polygon_path(f["geometry"], proj)
            if p is None:
                continue
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor=PROTECTED_COLOR, lw=6 * S * pt,
                                   alpha=0.12, joinstyle="round", zorder=5))
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor=PROTECTED_COLOR, lw=1.2 * S * pt,
                                   alpha=0.35, joinstyle="round", zorder=5.1))

    # ---- 6. admin boundaries ----
    for f in feats:
        if L(f) != "boundary":
            continue
        lvl = f["properties"].get("admin_level", 10)
        if lvl >= 9 and z < 14:
            continue
        if lvl >= 7 and z < 11:
            continue
        w = {2: 2.5, 3: 2.0, 4: 1.6, 5: 1.2, 6: 1.2}.get(lvl, 1.0)
        dash = (0, (6, 2)) if lvl <= 4 else (0, (4, 2, 1, 2))
        p = lines_path(line_parts(f["geometry"], proj))
        if p is not None:
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor=BOUNDARY_COLOR, lw=w * S * pt,
                                   linestyle=dash, alpha=0.6 if lvl <= 6 else 0.45, zorder=6))

    # ---- 7. roads ----
    trans = [f for f in feats if L(f) == "transportation"]
    # tracks & paths underneath real roads
    for f in trans:
        c = C(f)
        if c == "track" and visible("transportation", "track", z):
            p = lines_path(line_parts(f["geometry"], proj))
            if p is not None:
                ax.add_patch(PathPatch(p, facecolor="none", edgecolor="#ffffff", lw=2.2 * S * pt, alpha=0.4, zorder=7))
                ax.add_patch(PathPatch(p, facecolor="none", edgecolor="#996600", lw=1.2 * S * pt,
                                       linestyle=(0, (3 * S, 4 * S)), zorder=7.1))
        elif c == "path" and visible("transportation", "path", z) and f["geometry"]["type"] in ("LineString", "MultiLineString"):
            color, w, dash = PATH_STYLE.get(SC(f), PATH_STYLE["path"])
            p = lines_path(line_parts(f["geometry"], proj))
            if p is not None:
                ax.add_patch(PathPatch(p, facecolor="none", edgecolor="#ffffff", lw=(w + 1.0) * S * pt, alpha=0.4, zorder=7))
                ax.add_patch(PathPatch(p, facecolor="none", edgecolor=color, lw=w * S * pt,
                                       linestyle=(0, (dash[0] * S, dash[1] * S)), capstyle="round", zorder=7.1))
        elif c == "path" and SC(f) == "pedestrian" and f["geometry"]["type"] in ("Polygon", "MultiPolygon") and z >= 13:
            p = polygon_path(f["geometry"], proj)
            if p is not None:
                ax.add_patch(PathPatch(p, facecolor="#dddde8", edgecolor="#999999", lw=0.5 * S * pt, zorder=7))

    # casings for all classes, then fills, in rank order (so junctions merge like Carto)
    road_paths = {}
    for f in trans:
        c = C(f)
        if c not in ROADS or z < ROADS[c][2]:
            continue
        if f["geometry"]["type"] not in ("LineString", "MultiLineString"):
            continue
        tunnel = f["properties"].get("brunnel") == "tunnel"
        bridge = f["properties"].get("brunnel") == "bridge"
        road_paths.setdefault(c, []).append((lines_path(line_parts(f["geometry"], proj)), tunnel, bridge))

    zbase = 8
    for i, c in enumerate(ROAD_ORDER):
        if c not in road_paths:
            continue
        fill, casing, _, table = ROADS[c]
        total, cw = zw(table, z)
        lowzoom_minor = (c == "minor" and z < 13)
        for p, tunnel, bridge in road_paths[c]:
            if p is None:
                continue
            if lowzoom_minor:
                ax.add_patch(PathPatch(p, facecolor="none", edgecolor=MINOR_Z12_COLOR, lw=total * S * pt,
                                       capstyle="round", joinstyle="round", zorder=zbase + i))
                continue
            if casing and cw > 0:
                cw_eff = cw + (0.5 if bridge else 0)
                ax.add_patch(PathPatch(p, facecolor="none", edgecolor="#000000" if bridge else casing,
                                       lw=(total + (1.0 if bridge else 0)) * S * pt,
                                       linestyle=(0, (4 * S, 2 * S)) if tunnel else "solid",
                                       capstyle="butt" if bridge else "round", joinstyle="round",
                                       zorder=zbase + i))
        for p, tunnel, bridge in road_paths[c]:
            if p is None or lowzoom_minor:
                continue
            fw = total - 2 * cw if casing else total
            fcol = fill
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor=fcol, lw=max(fw, 0.5) * S * pt,
                                   alpha=0.6 if tunnel else 1.0,
                                   capstyle="round", joinstyle="round", zorder=zbase + i + 0.5))

    # railways
    for f in trans:
        if C(f) != "rail" or f["geometry"]["type"] not in ("LineString", "MultiLineString"):
            continue
        p = lines_path(line_parts(f["geometry"], proj))
        if p is None:
            continue
        if z >= 13:
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor="#707070", lw=3 * S * pt, zorder=17))
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor="#ffffff", lw=1 * S * pt,
                                   linestyle=(0, (8 * S, 12 * S)), zorder=17.1))
        else:
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor="#787878", lw=1.6 * S * pt, zorder=17))
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor="#ffffff", lw=0.6 * S * pt,
                                   linestyle=(0, (6 * S, 6 * S)), zorder=17.1))

    # ---- 8. point symbols ----
    for f in feats:
        if f["geometry"]["type"] != "Point":
            continue
        (px, py), = proj.pts([f["geometry"]["coordinates"]])
        if L(f) == "mountain_peak" and z >= 11:
            col = "#d40000" if C(f) == "volcano" else "#d08f55"
            ax.plot(px, py, marker="^", ms=6 * S * pt, color=col, mec="none", zorder=20)
        elif L(f) == "poi" and C(f) == "railway" and z >= 12:
            ax.plot(px, py, marker="s", ms=(6 if z >= 13 else 4) * S * pt, color="#7981b0", mec="none", zorder=20)

    # ---- 9. labels ----
    halo = lambda w=1.5: [pe.withStroke(linewidth=w * 2 * S * pt, foreground=(1, 1, 1, 0.85))]

    def place_text(x, y, text, size, color="#000000", style="normal", weight="normal", rot=0, h=halo()):
        if not text or not (0 <= x <= W and 0 <= y <= H):
            return
        t = ax.text(x, y, text, fontsize=size * S * pt, color=color, ha="center", va="center", rotation=rot,
                    fontfamily=LABEL_FONT, fontstyle=style, fontweight=weight, path_effects=h, zorder=30, clip_on=True)
        # simple collision avoidance, like Mapnik's: first placed wins
        bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
        if any(bb.overlaps(o) for o in placed):
            t.remove()
            return
        placed.append(bb)

    renderer = fig.canvas.get_renderer()
    placed = []
    prio = {"city": 0, "town": 1, "suburb": 2, "village": 3, "hamlet": 4}
    feats = sorted(feats, key=lambda f: (0 if L(f) == "place" else 1 if L(f) == "poi" else 2,
                                         prio.get(C(f), 9), f["properties"].get("rank", 99)))
    for f in feats:
        if f["geometry"]["type"] != "Point":
            continue
        name = f["properties"].get("name") or f["properties"].get("name_int")
        (px, py), = proj.pts([f["geometry"]["coordinates"]])
        if L(f) == "place" and visible("place", C(f), z):
            size = {"city": 14, "town": 12 if z >= 12 else 10, "village": 10 if z >= 13 else 9,
                    "suburb": 11, "hamlet": 9}.get(C(f), 9)
            place_text(px, py, name, size, color="#000000" if C(f) != "suburb" else "#6f6f6f")
        elif L(f) == "mountain_peak" and z >= 13:
            ele = f["properties"].get("ele")
            txt = f"{name}\n{ele} m" if name and ele else name
            place_text(px, py + 9 * S, txt, 9, color="#d08f55")
        elif L(f) == "poi" and C(f) == "railway" and z >= 13:
            place_text(px, py + 9 * S, name, 9, color="#7981b0", weight="bold")

    # road names on straight-ish midpoints (Carto: secondary z13+, tertiary z14+, minor z15+)
    name_min = {"motorway": 13, "trunk": 13, "primary": 13, "secondary": 13, "tertiary": 14, "minor": 15, "service": 16}
    for f in feats:
        if L(f) != "transportation_name":
            continue
        if z < name_min.get(C(f), 99):
            continue
        name = f["properties"].get("name")
        best = max(line_parts(f["geometry"], proj), key=lambda p: len(p), default=None)
        if not name or not best or len(best) < 2:
            continue
        seglen = lambda a, b: math.hypot(b[0] - a[0], b[1] - a[1])
        total = sum(seglen(a, b) for a, b in zip(best, best[1:]))
        if total < len(name) * 5.5 * S:
            continue
        acc, mid = 0, None
        for a, b in zip(best, best[1:]):
            l = seglen(a, b)
            if acc + l >= total / 2:
                t = (total / 2 - acc) / l if l else 0
                mid = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a, b)
                break
            acc += l
        if mid:
            x, y, a, b = mid
            ang = -math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            if ang > 90: ang -= 180
            if ang < -90: ang += 180
            place_text(x, y, name, 9, color="#222222", rot=ang)

    # water names: rivers on lines z13+, lakes by area
    for f in feats:
        name = f["properties"].get("name")
        if not name:
            continue
        if L(f) == "waterway" and C(f) == "river" and z >= 13:
            parts = line_parts(f["geometry"], proj)
            best = max(parts, key=len)
            if len(best) >= 2:
                a, b = best[len(best) // 2 - 1], best[len(best) // 2]
                ang = -math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
                if ang > 90: ang -= 180
                if ang < -90: ang += 180
                place_text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, name, 9, color="#4d80b3", style="italic", rot=ang)
        elif L(f) == "water" and f["geometry"]["type"] in ("Polygon", "MultiPolygon"):
            p = polygon_path(f["geometry"], proj)
            if p is not None and path_area(p) > (2500 * S * S):
                e = p.get_extents()
                place_text((e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2, name, 9, color="#4d80b3", style="italic")

    fig.savefig(out, dpi=100, facecolor=LAND)
    plt.close(fig)
    return {"zoom": z, "pixel_window": [x0, y0, x1, y1], "size": [W, H], "scale": scale}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("geojson")
    ap.add_argument("--zoom", type=int)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    base = os.path.basename(a.geojson).split(".")[0]
    out = a.out or f"{base}_carto_z{a.zoom or 'src'}{'@%dx' % a.scale if a.scale > 1 else ''}.png"
    info = render(a.geojson, out, a.zoom, a.scale)
    print(out, json.dumps(info))
