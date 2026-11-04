#!/usr/bin/env python3
"""Render OpenMapTiles-schema GeoJSON with the OSM OpenMapTiles style.

Same interface and pixel grid as render_carto.py (Web Mercator, aligned to the 256px OSM tile grid,
cropped to the GeoJSON bbox), but drawn from the official OpenMapTiles style
(https://openmaptiles.org/styles/osm-openmaptiles/) as published in github.com/openmaptiles/openmaptiles
at commit 6c11838d38030148832c039c2ea367274db86a87 and stored in utils/omt_style/:
    layers/*/style.json -> utils/omt_style/*.json        (the style rules)
    style/style-header.json -> utils/omt_style/style-header.json
    style/icons/*.svg -> utils/omt_style/icons/           (fill patterns and icons)
Style design CC-BY 4.0, (c) OpenMapTiles.

Everything drawn comes from the style: colours, widths, dashes, opacities, zoom ranges, filters, fill
patterns and icons (the style's SVGs, rasterised with cairosvg), one-way arrows (the style's oneway
icons), and label fonts (the style's text-font stack: the first font installed is used, as MapLibre
does). Missing pieces raise an error instead of being approximated:
    - cairosvg + the cairo library are required:  conda install -c conda-forge cairosvg cairo
    - a font from each label's text-font stack must be installed (Open Sans, else Noto Sans)
Simplified: labels are placed at a point / the middle of a line with a first-come overlap check
(no MapLibre placement priorities); road shields are drawn once per road at their natural size
(no icon-text-fit); line-offset / line-gap-width are not drawn.

Usage:
    python render_omt.py crop.geojson.gz [--zoom Z] [--scale 1|2] [-o out.png]
Zoom defaults to metadata.camera_zoom. Style zoom = zoom - 1 (the style is written for 512px tiles).
"""
import argparse, colorsys, gzip, io, json, math, os, re, sys
from functools import lru_cache
from pathlib import Path as FilePath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_carto as rc  # projection + geometry helpers; also switches matplotlib to Agg

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from PIL import Image

STYLE_DIR = FilePath(__file__).resolve().parent / "omt_style"
ICON_DIR = STYLE_DIR / "icons"
FONT_WEIGHTS = {"Light": "light", "Regular": "normal", "Semibold": "semibold", "Bold": "bold"}


def load_style(style_dir=STYLE_DIR):
    """All style rules (layers/*/style.json), sorted by their 'order', plus the background colour."""
    header = json.load(open(style_dir / "style-header.json"))
    rules = [l for f in sorted(style_dir.glob("*.json")) if f.name != "style-header.json"
             for l in json.load(open(f))["layers"]]
    rules.sort(key=lambda l: l["order"])
    background = next(l["paint"]["background-color"] for l in header["layers"] if l["type"] == "background")
    return background, rules


def style_colours(style_dir=STYLE_DIR):
    """Every constant colour in the style, as sorted 0-255 RGB tuples (used for palette PNGs)."""
    background, rules = load_style(style_dir)
    cols = set()

    def walk(v):
        if is_color(v):
            cols.add(tuple(round(c * 255) for c in parse_color(v)[:3]))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(background)
    for r in rules:
        walk(r.get("paint", {}))
    return sorted(cols)

# ----------------------------------------------------------------------------
# Icons, patterns and fonts from the style
# ----------------------------------------------------------------------------

def require_cairo():
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>')
    except Exception as e:
        raise RuntimeError("render_omt.py draws the style's SVG patterns and icons with cairosvg, which needs "
                           "the cairo library: conda install -c conda-forge cairosvg cairo") from e


@lru_cache(maxsize=None)
def icon_image(name, scale):
    """The style's icon utils/omt_style/icons/<name>.svg as an RGBA float array."""
    import cairosvg
    png = cairosvg.svg2png(url=str(ICON_DIR / f"{name}.svg"), scale=scale)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")) / 255.0


@lru_cache(maxsize=None)
def pattern_canvas(name, scale, W, H):
    """A pattern icon tiled over the whole W x H canvas (clipped to each polygon when drawn)."""
    tile = icon_image(name, scale)
    reps = (math.ceil(H / tile.shape[0]), math.ceil(W / tile.shape[1]), 1)
    return np.tile(tile, reps)[:H, :W]


def has_icon(name):
    return isinstance(name, str) and bool(name) and (ICON_DIR / f"{name}.svg").exists()


@lru_cache(maxsize=None)
def font_props(stack):
    """matplotlib font for a text-font stack: the first installed font, as MapLibre picks glyphs.
    'Open Sans Semibold Italic' -> family 'Open Sans', weight semibold, style italic."""
    for name in stack:
        words = name.split()
        style = "italic" if words[-1] == "Italic" else "normal"
        words = words[:-1] if style == "italic" else words
        weight = FONT_WEIGHTS.get(words[-1], "normal")
        family = " ".join(words[:-1] if words[-1] in FONT_WEIGHTS else words)
        prop = font_manager.FontProperties(family=family, weight=weight, style=style)
        try:
            font_manager.findfont(prop, fallback_to_default=False)
            return prop
        except ValueError:
            continue
    raise RuntimeError(f"none of the style's fonts {list(stack)} is installed "
                       "(e.g. brew install --cask font-open-sans font-noto-sans)")

# ----------------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------------

def parse_color(c):
    """CSS colour string (hsl/hsla/rgb/rgba/#hex/name) -> RGBA tuple in 0..1."""
    c = c.strip()
    m = re.match(r"(hsla?|rgba?)\(([^)]*)\)", c)
    if m:
        parts = [p.strip() for p in m.group(2).split(",")]
        a = float(parts[3]) if len(parts) > 3 else 1.0
        if m.group(1).startswith("hsl"):
            h, s, l = float(parts[0]) / 360, float(parts[1].rstrip("%")) / 100, float(parts[2].rstrip("%")) / 100
            r, g, b = colorsys.hls_to_rgb(h, l, s)
        else:
            r, g, b = (float(p) / 255 for p in parts[:3])
        return (r, g, b, a)
    return mc.to_rgba(c)


def is_color(v):
    return isinstance(v, str) and (v.startswith(("hsl", "rgb", "#")) or v in ("white", "black", "transparent"))

# ----------------------------------------------------------------------------
# Style expressions (MapLibre style spec)
# ----------------------------------------------------------------------------

def interp(a, b, t):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    if is_color(a) and is_color(b):
        ca, cb = parse_color(a), parse_color(b)
        return tuple(x + (y - x) * t for x, y in zip(ca, cb))
    return a if t < 1 else b


def interp_stops(stops, x, base=1.0):
    """Exponential interpolation between [input, value] stops (base 1 = linear)."""
    if x <= stops[0][0]:
        return stops[0][1]
    for (x0, v0), (x1, v1) in zip(stops, stops[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return v1
            t = (x - x0) / (x1 - x0) if base == 1 else (base ** (x - x0) - 1) / (base ** (x1 - x0) - 1)
            return interp(v0, v1, t)
    return stops[-1][1]


def evaluate(v, z, props):
    """Evaluate a paint/layout value: constant, legacy {stops} function, or expression."""
    if isinstance(v, dict) and "stops" in v:
        x = props.get(v["property"]) if "property" in v else z
        if x is None:
            return v.get("default")
        if v.get("type") == "categorical" or not isinstance(x, (int, float)):
            return next((val for key, val in v["stops"] if key == x), v.get("default"))
        first = v["stops"][0][1]
        if v.get("type") == "interval" or (isinstance(first, str) and not is_color(first)) or isinstance(first, list):
            out = first   # step function (strings that aren't colours, arrays)
            for key, val in v["stops"]:
                if x >= key:
                    out = val
            return out
        return interp_stops(v["stops"], x, v.get("base", 1))
    if isinstance(v, list) and v and v[0] in EXPRESSION_OPS:
        return expr(v, z, props)
    return v   # constants, incl. plain arrays such as text-font stacks and dash arrays


EXPRESSION_OPS = {"get", "has", "zoom", "literal", "coalesce", "image", "to-string", "string", "to-number", "concat", "match",
                  "case", "step", "interpolate", "==", "!=", "<", "<=", ">", ">=", "all", "any", "!", "in"}


def expr(e, z, props):
    op, args = e[0], e[1:]
    ev = lambda a: evaluate(a, z, props)
    if op == "get":
        return props.get(args[0])
    if op == "has":
        return args[0] in props
    if op == "zoom":
        return z
    if op == "literal":
        return args[0]
    if op == "coalesce":
        return next((x for x in map(ev, args) if x not in (None, "")), None)
    if op == "image":   # an image is available only if the style has that icon
        name = ev(args[0])
        return name if has_icon(name) else None
    if op in ("to-string", "string"):
        x = ev(args[0])
        return "" if x is None else str(x)
    if op == "to-number":   # first argument convertible to a number, as in the style spec
        for a in args:
            try:
                return float(ev(a))
            except (TypeError, ValueError):
                continue
        return 0
    if op == "concat":
        return "".join("" if x is None else str(x) for x in map(ev, args))
    if op == "match":
        x = ev(args[0])
        for labels, out in zip(args[1:-1:2], args[2:-1:2]):
            if x == labels or (isinstance(labels, list) and x in labels):
                return ev(out)
        return ev(args[-1])
    if op == "case":
        for cond, out in zip(args[0:-1:2], args[1:-1:2]):
            if ev(cond):
                return ev(out)
        return ev(args[-1])
    if op == "step":
        x, out = ev(args[0]), ev(args[1])
        for thr, val in zip(args[2::2], args[3::2]):
            if x is not None and x >= thr:
                out = ev(val)
        return out
    if op == "interpolate":
        kind, x = args[0], ev(args[1])
        base = kind[1] if kind[0] == "exponential" else 1.0
        stops = [(a, ev(b)) for a, b in zip(args[2::2], args[3::2])]
        return interp_stops(stops, x, base) if x is not None else stops[0][1]
    if op in ("==", "!=", "<", "<=", ">", ">="):
        return compare(op, ev(args[0]), ev(args[1]))
    if op == "all":
        return all(ev(a) for a in args)
    if op == "any":
        return any(ev(a) for a in args)
    if op == "!":
        return not ev(args[0])
    if op == "in":
        needle, hay = ev(args[0]), ev(args[1])
        return hay is not None and needle in hay
    raise NotImplementedError(f"style expression {op!r} is not supported: {e}")


def compare(op, a, b):
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    try:  # ordering comparisons with a missing (None) value are false, as in the style spec
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        return a >= b
    except TypeError:
        return False


def matches(f, z, props):
    """Style filter: legacy syntax (["==", "class", "x"]) or expression syntax (["==", ["get", "class"], "x"])."""
    if f is None:
        return True
    op, args = f[0], f[1:]
    if op == "all":
        return all(matches(a, z, props) for a in args)
    if op == "any":
        return any(matches(a, z, props) for a in args)
    if op == "none":
        return not any(matches(a, z, props) for a in args)
    if op == "!":
        return not matches(args[0], z, props)
    if args and isinstance(args[0], str) and op in ("==", "!=", "<", "<=", ">", ">=", "in", "!in", "has", "!has"):
        key = args[0]                                   # legacy filter
        val = props.get(key)
        if op == "has":
            return key in props
        if op == "!has":
            return key not in props
        if op == "in":
            return val in args[1:]
        if op == "!in":
            return val not in args[1:]
        return compare(op, val, args[1])
    return bool(expr(f, z, props))

# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

GEOM_TYPE = {"Point": "Point", "MultiPoint": "Point", "LineString": "LineString",
             "MultiLineString": "LineString", "Polygon": "Polygon", "MultiPolygon": "Polygon"}
CAPS = {"butt": "butt", "round": "round", "square": "projecting"}
JOINS = {"miter": "miter", "round": "round", "bevel": "bevel"}
# property defaults from the MapLibre style spec, used when a rule doesn't set the property
SPEC_DEFAULTS = {"fill-color": "#000000", "line-color": "#000000",
                 "fill-opacity": 1, "line-opacity": 1, "line-width": 1, "icon-opacity": 1, "text-opacity": 1,
                 "text-color": "#000000", "text-halo-width": 0, "icon-size": 1, "icon-rotate": 0,
                 "symbol-spacing": 250, "text-size": 16, "line-cap": "butt", "line-join": "miter",
                 "text-font": ["Open Sans Regular", "Arial Unicode MS Regular"]}


def rgba(value, opacity=1.0):
    r, g, b, a = parse_color(value) if isinstance(value, str) else value
    return (r, g, b, a * opacity)


def tokens(v, props):
    """Replace {key} placeholders (text-field, icon-image) with feature properties."""
    return re.sub(r"\{([^}]+)\}", lambda m: str(props.get(m.group(1), "")), v).strip() if isinstance(v, str) else v


def render(path, out, zoom=None, scale=1, style_dir=STYLE_DIR):
    require_cairo()
    background, rules = load_style(FilePath(style_dir))
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as fh:
        data = json.load(fh)
    z = zoom if zoom is not None else int(data["metadata"]["camera_zoom"])
    zs = z - 1  # style zoom (512px tiles)
    x0, y0, x1, y1 = rc.pixel_window(data["bbox"], z)
    W, H = (x1 - x0) * scale, (y1 - y0) * scale
    proj = rc.Proj(z, x0, y0, scale)
    S, pt = scale, 72 / 100  # style px -> canvas px, canvas px -> points at dpi=100

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    bg = rgba(evaluate(background, zs, {}))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    by_layer = {}
    for f in data["features"]:
        props = dict(f["properties"], **{"$type": GEOM_TYPE.get(f["geometry"]["type"])})
        by_layer.setdefault(props.get("omt_layer"), []).append((f, props))

    renderer = fig.canvas.get_renderer()
    placed = []

    def place_text(x, y, text, size, font, color, halo_color, halo_w, rot=0):
        if not text or not (0 <= x <= W and 0 <= y <= H):
            return
        effects = [pe.withStroke(linewidth=2 * halo_w * S * pt, foreground=halo_color)] if halo_w else None
        t = ax.text(x, y, text, fontsize=size * S * pt, fontproperties=font, color=color, ha="center",
                    va="center", rotation=rot, path_effects=effects, zorder=1000, clip_on=True)
        bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
        if any(bb.overlaps(o) for o in placed):
            t.remove()
        else:
            placed.append(bb)

    def draw_icon(img, x, y, opacity=1.0, deg=0.0, zorder=900):
        h, w = img.shape[:2]
        im = ax.imshow(img, extent=(x - w / 2, x + w / 2, y + h / 2, y - h / 2), alpha=opacity,
                       zorder=zorder, interpolation="bilinear")
        if deg:
            im.set_transform(Affine2D().rotate_deg_around(x, y, deg) + ax.transData)

    def along_line(parts, spacing):
        """(x, y, direction in radians) every `spacing` px along each line part."""
        for p in parts:
            dist = spacing / 2
            for (ax_, ay), (bx, by) in zip(p, p[1:]):
                seg = math.hypot(bx - ax_, by - ay)
                while dist <= seg and seg > 0:
                    t = dist / seg
                    yield ax_ + (bx - ax_) * t, ay + (by - ay) * t, math.atan2(by - ay, bx - ax_)
                    dist += spacing
                dist -= seg

    for zorder, rule in enumerate(rules):
        paint, layout = rule.get("paint", {}), rule.get("layout", {})
        if layout.get("visibility") == "none":
            continue
        if not (rule.get("minzoom", 0) <= zs < rule.get("maxzoom", 24)):
            continue
        for f, props in by_layer.get(rule["source-layer"], []):
            if not matches(rule.get("filter"), zs, props):
                continue
            geom, gtype = f["geometry"], props["$type"]
            P = lambda key: evaluate(paint.get(key, SPEC_DEFAULTS.get(key)), zs, props)
            L = lambda key: evaluate(layout.get(key, SPEC_DEFAULTS.get(key)), zs, props)

            if rule["type"] == "fill" and gtype == "Polygon":
                p = rc.polygon_path(geom, proj)
                if p is None:
                    continue
                pattern = P("fill-pattern")
                if pattern:   # the style's SVG pattern, tiled and clipped to the polygon
                    im = ax.imshow(pattern_canvas(pattern, S, int(W), int(H)), extent=(0, W, H, 0),
                                   alpha=P("fill-opacity"), zorder=zorder, interpolation="nearest")
                    im.set_clip_path(PathPatch(p, transform=ax.transData))
                    continue
                if "fill-pattern" in paint:   # pattern rule with no pattern for this feature
                    continue
                outline = P("fill-outline-color")
                ax.add_patch(PathPatch(p, facecolor=rgba(P("fill-color"), P("fill-opacity")),
                                       edgecolor=rgba(outline, P("fill-opacity")) if outline else "none",
                                       lw=0.5 * S * pt if outline else 0, zorder=zorder))

            elif rule["type"] == "line" and gtype in ("LineString", "Polygon"):
                p = rc.lines_path(rc.line_parts(geom, proj))
                width = P("line-width")
                if p is None or not width:
                    continue
                dash = P("line-dasharray")
                ls = (0, tuple(max(d, 0.01) for d in dash)) if dash else "solid"
                ax.add_patch(PathPatch(p, facecolor="none", edgecolor=rgba(P("line-color"), P("line-opacity")),
                                       lw=width * S * pt, linestyle=ls, capstyle=CAPS[L("line-cap")],
                                       joinstyle=JOINS[L("line-join")], zorder=zorder))

            elif rule["type"] == "symbol":
                icon = tokens(L("icon-image"), props)
                icon_drawn = False
                shield = None   # line icon with a label (road shields): drawn once, under the label
                if has_icon(icon):
                    img = icon_image(icon, S * L("icon-size"))
                    if L("symbol-placement") == "line" and gtype == "LineString":
                        if "text-field" in layout:
                            shield = img
                        else:   # icon repeated along the line, e.g. one-way arrows
                            for x, y, ang in along_line(rc.line_parts(geom, proj), L("symbol-spacing") * S):
                                draw_icon(img, x, y, P("icon-opacity"), math.degrees(ang) + L("icon-rotate"))
                    elif gtype == "Point":
                        (ix, iy), = proj.pts([geom["coordinates"]] if geom["type"] == "Point" else geom["coordinates"][:1])
                        if 0 <= ix <= W and 0 <= iy <= H:
                            draw_icon(img, ix, iy, P("icon-opacity"), L("icon-rotate"), zorder=950)
                            icon_drawn = True
                if "text-field" not in layout:
                    continue
                text = tokens(L("text-field"), props) or ""
                if not text:
                    continue
                if L("text-transform") == "uppercase":
                    text = text.upper()
                size, font = L("text-size"), font_props(tuple(L("text-font")))
                color = rgba(P("text-color"), P("text-opacity"))
                halo = P("text-halo-color")
                halo_w, halo_rgba = (P("text-halo-width"), rgba(halo)) if halo else (0, None)
                if gtype == "Point":
                    (x, y), = proj.pts([geom["coordinates"]] if geom["type"] == "Point" else geom["coordinates"][:1])
                    offset = L("text-offset")
                    if icon_drawn and offset:   # text-offset is in ems
                        x, y = x + offset[0] * size * S, y + offset[1] * size * S
                    place_text(x, y, text, size, font, color, halo_rgba, halo_w)
                else:  # line placement: middle of the longest part, along the line
                    parts = rc.line_parts(geom, proj)
                    best = max(parts, key=len) if parts else []
                    if len(best) >= 2:
                        a, b = best[len(best) // 2 - 1], best[len(best) // 2]
                        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                        if shield is not None:   # shields stay upright, label centred on them
                            if 0 <= mx <= W and 0 <= my <= H:
                                draw_icon(shield, mx, my, P("icon-opacity"), zorder=990)
                            place_text(mx, my, text, size, font, color, halo_rgba, halo_w)
                            continue
                        ang = -math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
                        ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
                        place_text(mx, my, text, size, font, color, halo_rgba, halo_w, ang)

    ax.set_xlim(0, W)   # imshow (patterns/icons) can reset the view; restore the pixel window
    ax.set_ylim(H, 0)
    ax.set_aspect("auto")
    fig.savefig(out, dpi=100, facecolor=bg)
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
    out = a.out or f"{base}_omt_z{a.zoom or 'src'}{'@%dx' % a.scale if a.scale > 1 else ''}.png"
    info = render(a.geojson, out, a.zoom, a.scale)
    print(out, json.dumps(info))
