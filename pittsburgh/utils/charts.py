"""
charts.py
Generates analysis charts for the Pittsburgh Transit Analyzer.

Each map function returns raw PNG bytes.
Chart 2 returns (table_png, map_png, csv_df) — table and map are separate images.
Charts 3, 4b return (png, csv_df).

Dependencies:
    pip install contextily pyproj pillow
"""

import io
import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── contextily / pyproj ───────────────────────────────────────────────────────
try:
    import contextily as ctx
    from pyproj import Transformer
    CTX_AVAILABLE = True
except ImportError:
    CTX_AVAILABLE = False

# ── Icon path ─────────────────────────────────────────────────────────────────
_STATIC_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "icons")
_BUS_ICON    = os.path.join(_STATIC_DIR, "bus.png")

# ── Colors ────────────────────────────────────────────────────────────────────
RED    = "#E74C3C"
ORANGE = "#E67E22"
GRAY   = "#7F8C8D"
GREEN  = "#27AE60"
BLUE   = "#4A90D9"

# POGOH official palette
POGOH_DARK  = "#003865"   # dark navy  → departure marker fill
POGOH_TEAL  = "#00B2A9"   # teal/cyan  → arrival marker fill

# Route line colors (cycling palette for multi-route maps)
ROUTE_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#0097A7",
]

# Basemap tile source — CartoDB Positron (no API key required, clean light-grey style)
BASEMAP = ctx.providers.CartoDB.Positron if CTX_AVAILABLE else None

plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.grid":          False,
})


# ── Geo helpers ───────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    """Return distance in metres between two WGS-84 points."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def filter_by_radius(df, lat, lon, radius_m, lat_col="stop_lat", lon_col="stop_lon"):
    """Return rows within radius_m metres; appends dist_m column."""
    d = df.copy()
    d["dist_m"] = d.apply(
        lambda r: haversine_m(lat, lon, float(r[lat_col]), float(r[lon_col])), axis=1
    )
    return d[d["dist_m"] <= radius_m].reset_index(drop=True)


def _deg(radius_m):
    """Metres → approximate degrees (for fallback axis limits)."""
    return radius_m / 111_000


def _fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _empty_chart(msg):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=13, color=GRAY, transform=ax.transAxes)
    return _fig_to_bytes(fig)


# ── Projection helpers ────────────────────────────────────────────────────────

def _to_merc(lons, lats):
    """Project WGS-84 lon/lat to Web Mercator (EPSG:3857). Returns (xs, ys)."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = t.transform(list(lons), list(lats))
    return np.array(xs), np.array(ys)


def _circle_merc(lon, lat, radius_m, n=256):
    """Return (xs, ys) of a geodetic circle projected to Web Mercator."""
    angles  = np.linspace(0, 2 * math.pi, n)
    lat_r   = radius_m / 111_000
    lon_r   = radius_m / (111_000 * math.cos(math.radians(lat)))
    lons_c  = lon + lon_r * np.cos(angles)
    lats_c  = lat + lat_r * np.sin(angles)
    return _to_merc(lons_c, lats_c)


# ── Basemap helper ────────────────────────────────────────────────────────────

def _setup_map_ax(ax, lon, lat, radius_m):
    """
    Set axis limits in Web Mercator and add a Stadia AlidadeSmooth basemap.
    Returns True if basemap was added, False on failure/unavailability.
    The axis is always set to a square aspect ratio.
    """
    if not CTX_AVAILABLE:
        return False
    try:
        pad   = 1.35
        deg   = _deg(radius_m) * pad
        xs, ys = _to_merc([lon - deg, lon + deg], [lat - deg, lat + deg])
        ax.set_xlim(xs[0], xs[1])
        ax.set_ylim(ys[0], ys[1])
        ax.set_aspect("equal")

        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=BASEMAP,
            attribution_size=5,
        )
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        return True
    except Exception as e:
        print(f"[basemap] failed: {e}")
        return False


def _fallback_limits(ax, lon, lat, radius_m):
    """Set plain WGS-84 limits when basemap is unavailable."""
    deg = _deg(radius_m) * 1.35
    ax.set_xlim(lon - deg, lon + deg)
    ax.set_ylim(lat - deg, lat + deg)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


# ── Bus icon helper ───────────────────────────────────────────────────────────

def _load_bus_icon(zoom=0.055):
    """
    Load bus.png and return a matplotlib OffsetImage.
    Uses zoom on the original 512px image instead of resizing,
    which preserves sharpness at display size.
    Returns None if the file is missing or PIL is unavailable.
    """
    try:
        from PIL import Image
        img = Image.open(_BUS_ICON).convert("RGBA")
        arr = np.array(img)
        return OffsetImage(arr, zoom=zoom)
    except Exception as e:
        print(f"[bus icon] {e}")
        return None


def _place_bus_icons(ax, xs, ys, zoom=0.055):
    """
    Place a bus icon at each (x, y) position on ax.
    Falls back to a simple scatter dot if icon is unavailable.
    """
    if len(xs) == 0:
        return
    # Test-load once to check availability
    test = _load_bus_icon(zoom)
    if test is None:
        ax.scatter(xs, ys, c=BLUE, s=60, zorder=5, alpha=0.9,
                   edgecolors="white", linewidths=0.5)
        return
    for x, y in zip(xs, ys):
        ab = AnnotationBbox(_load_bus_icon(zoom), (x, y),
                            frameon=False, zorder=6, pad=0)
        ax.add_artist(ab)


# ── Chart 1: Bus stop map ─────────────────────────────────────────────────────

def chart1_bus_stops(gtfs, lat, lon, radius_m):
    """
    Square map of bus stops within radius_m metres.
    Bus stops are rendered as bus icons (bus.png).
    """
    stops_in = filter_by_radius(
        gtfs["stops"].dropna(subset=["stop_lat", "stop_lon"]),
        lat, lon, radius_m,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    osm_ok  = _setup_map_ax(ax, lon, lat, radius_m)

    if osm_ok:
        cx, cy        = _to_merc([lon], [lat])
        cx, cy        = cx[0], cy[0]
        cir_x, cir_y  = _circle_merc(lon, lat, radius_m)

        # Study-area circle
        ax.plot(cir_x, cir_y, color=BLUE, linewidth=2,
                linestyle="--", zorder=3, alpha=0.85)
        ax.fill(cir_x, cir_y, color=BLUE, alpha=0.07, zorder=2)

        # Center marker
        ax.scatter([cx], [cy], c=RED, s=200, zorder=7, marker="*",
                   label=f"Center ({lat:.4f}, {lon:.4f})")

        if not stops_in.empty:
            sx, sy = _to_merc(stops_in["stop_lon"], stops_in["stop_lat"])
            _place_bus_icons(ax, sx, sy)

            # Labels for the 15 nearest stops
            for i, row in stops_in.nsmallest(15, "dist_m").iterrows():
                px, py = _to_merc([row["stop_lon"]], [row["stop_lat"]])
                ax.annotate(
                    row.get("stop_name", ""), (px[0], py[0]),
                    fontsize=6, color="#222", alpha=0.9,
                    xytext=(14, 3), textcoords="offset points", zorder=8,
                )
    else:
        _fallback_limits(ax, lon, lat, radius_m)
        deg = _deg(radius_m)
        ax.add_patch(plt.Circle((lon, lat), deg, color=BLUE, alpha=0.08, zorder=1))
        ax.add_patch(plt.Circle((lon, lat), deg, color=BLUE, fill=False,
                                linewidth=1.8, linestyle="--", zorder=2))
        ax.scatter([lon], [lat], c=RED, s=180, zorder=6, marker="*",
                   label=f"Center ({lat:.4f}, {lon:.4f})")
        if not stops_in.empty:
            _place_bus_icons(ax,
                             stops_in["stop_lon"].values,
                             stops_in["stop_lat"].values)
            for _, row in stops_in.nsmallest(15, "dist_m").iterrows():
                ax.annotate(row.get("stop_name", ""),
                            (row["stop_lon"], row["stop_lat"]),
                            fontsize=6, color="#333", alpha=0.85,
                            xytext=(10, 3), textcoords="offset points")

    count = len(stops_in)
    if count == 0:
        ax.text(0.5, 0.5, "No bus stops found in range",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color=GRAY)

    # Manual legend entry for bus icon
    bus_patch = mpatches.Patch(color=BLUE, label=f"Bus stops ({count})")
    center_patch = mpatches.Patch(color=RED, label=f"Center ({lat:.4f}, {lon:.4f})")
    ax.legend(handles=[center_patch, bus_patch],
              loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title(f"Bus Stops within {radius_m} m  (total: {count})",
                 fontsize=13, pad=10)
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ── Chart 2: Routes — TABLE (separate image) ──────────────────────────────────

def chart2_table(gtfs, lat, lon, radius_m):
    """
    Returns (table_png_bytes, csv_df).
    Table lists all routes serving the study area. No map.
    """
    stops_in = filter_by_radius(
        gtfs["stops"].dropna(subset=["stop_lat", "stop_lon"]),
        lat, lon, radius_m,
    )
    if stops_in.empty:
        return _empty_chart("No bus stops in range — cannot determine routes"), pd.DataFrame()

    stop_ids  = set(stops_in["stop_id"])
    st        = gtfs["stop_times"][gtfs["stop_times"]["stop_id"].isin(stop_ids)]
    trip_ids  = set(st["trip_id"])
    trips     = gtfs["trips"][gtfs["trips"]["trip_id"].isin(trip_ids)]
    route_ids = set(trips["route_id"])
    routes    = gtfs["routes"][gtfs["routes"]["route_id"].isin(route_ids)].copy()

    if routes.empty:
        return _empty_chart("No routes found serving this area"), pd.DataFrame()

    routes = routes.sort_values("route_short_name").reset_index(drop=True)
    n      = len(routes)

    # Dynamic figure height so each row has room; min 4 inches
    row_h  = 0.38
    fig_h  = max(4.0, n * row_h + 1.8)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.axis("off")

    table_data = [
        [str(r.get("route_short_name", "")), str(r.get("route_long_name", ""))]
        for _, r in routes.iterrows()
    ]
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Route", "Name"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF4FB")
        cell.set_edgecolor("#DDD")

    ax.set_title(f"Routes serving this area  ({n} routes)",
                 fontsize=12, pad=12)
    fig.tight_layout()

    csv_df = routes[["route_short_name", "route_long_name"]].copy()
    csv_df.columns = ["Route", "Name"]

    return _fig_to_bytes(fig), csv_df


# ── Chart 2: Routes — MAP (separate image) ───────────────────────────────────

def chart2_map(gtfs, lat, lon, radius_m):
    """
    Returns map_png_bytes.
    Square map with route shapes and bus-icon stops overlaid on basemap.
    """
    stops_in = filter_by_radius(
        gtfs["stops"].dropna(subset=["stop_lat", "stop_lon"]),
        lat, lon, radius_m,
    )
    if stops_in.empty:
        return _empty_chart("No bus stops in range — cannot show route map")

    stop_ids  = set(stops_in["stop_id"])
    st        = gtfs["stop_times"][gtfs["stop_times"]["stop_id"].isin(stop_ids)]
    trip_ids  = set(st["trip_id"])
    trips     = gtfs["trips"][gtfs["trips"]["trip_id"].isin(trip_ids)]
    route_ids = set(trips["route_id"])
    routes    = gtfs["routes"][gtfs["routes"]["route_id"].isin(route_ids)].copy()
    routes    = routes.sort_values("route_short_name").reset_index(drop=True)

    shape_ids = set(trips["shape_id"].dropna())
    shapes_df = gtfs["shapes"][gtfs["shapes"]["shape_id"].isin(shape_ids)]

    fig, ax = plt.subplots(figsize=(8, 8))
    osm_ok  = _setup_map_ax(ax, lon, lat, radius_m)

    if osm_ok:
        cx, cy       = _to_merc([lon], [lat])
        cx, cy       = cx[0], cy[0]
        cir_x, cir_y = _circle_merc(lon, lat, radius_m)

        ax.plot(cir_x, cir_y, color=BLUE, linewidth=1.8,
                linestyle="--", zorder=3, alpha=0.75)

        # Route shapes
        for i, (sid, grp) in enumerate(shapes_df.groupby("shape_id")):
            grp = grp.sort_values("shape_pt_sequence")
            sx, sy = _to_merc(grp["shape_pt_lon"], grp["shape_pt_lat"])
            ax.plot(sx, sy,
                    color=ROUTE_COLORS[i % len(ROUTE_COLORS)],
                    linewidth=1.8, alpha=0.7, zorder=4)

        # Bus stop icons
        bx, by = _to_merc(stops_in["stop_lon"], stops_in["stop_lat"])
        _place_bus_icons(ax, bx, by)

        # Center
        ax.scatter([cx], [cy], c=RED, s=200, zorder=8, marker="*",
                   label="Center")
    else:
        _fallback_limits(ax, lon, lat, radius_m)
        deg = _deg(radius_m)
        ax.add_patch(plt.Circle((lon, lat), deg, color=BLUE, fill=False,
                                linewidth=1.5, linestyle="--", zorder=2))
        for i, (sid, grp) in enumerate(shapes_df.groupby("shape_id")):
            grp = grp.sort_values("shape_pt_sequence")
            ax.plot(grp["shape_pt_lon"], grp["shape_pt_lat"],
                    color=ROUTE_COLORS[i % len(ROUTE_COLORS)],
                    linewidth=1.2, alpha=0.65, zorder=3)
        _place_bus_icons(ax,
                         stops_in["stop_lon"].values,
                         stops_in["stop_lat"].values)
        ax.scatter([lon], [lat], c=RED, s=160, zorder=6, marker="*",
                   label="Center")

    n_routes = len(routes)
    ax.set_title(f"Route paths near study area  ({n_routes} routes)", fontsize=13, pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ── Chart 3: Service frequency & operating hours ──────────────────────────────

def chart3_schedule(gtfs, lat, lon, radius_m):
    """
    Top: headway bar chart per route.
    Bottom: heatmap of trips per hour.
    Returns (png_bytes, csv_df).
    """
    stops_in = filter_by_radius(
        gtfs["stops"].dropna(subset=["stop_lat", "stop_lon"]),
        lat, lon, radius_m,
    )
    if stops_in.empty:
        return _empty_chart("No bus stops in range"), pd.DataFrame()

    stop_ids = set(stops_in["stop_id"])
    st = gtfs["stop_times"][gtfs["stop_times"]["stop_id"].isin(stop_ids)].copy()
    st = st[["trip_id", "arrival_time"]].dropna()
    st = st.merge(gtfs["trips"][["trip_id", "route_id"]], on="trip_id")
    st = st.merge(gtfs["routes"][["route_id", "route_short_name"]], on="route_id")

    def parse_hour(t):
        try:
            return int(str(t).split(":")[0]) % 24
        except Exception:
            return None

    st["hour"] = st["arrival_time"].apply(parse_hour)
    st = st.dropna(subset=["hour"])
    st["hour"] = st["hour"].astype(int)

    summary = (
        st.groupby(["route_id", "route_short_name"])
        .agg(first_hour=("hour", "min"), last_hour=("hour", "max"),
             total_trips=("trip_id", "nunique"))
        .reset_index()
        .sort_values("route_short_name")
    )
    summary["op_hours"]    = (summary["last_hour"] - summary["first_hour"]).clip(lower=1)
    summary["headway_min"] = (summary["op_hours"] * 60 / summary["total_trips"]).round(0).astype(int)
    summary["first_str"]   = summary["first_hour"].apply(lambda h: f"{h:02d}:00")
    summary["last_str"]    = summary["last_hour"].apply(lambda h: f"{h:02d}:xx")

    n = len(summary)
    fig, (ax_bar, ax_heat) = plt.subplots(
        2, 1,
        figsize=(12, max(6, n * 0.62 + 4.5)),
        gridspec_kw={"height_ratios": [2, 1.8]},
    )

    # Headway bar chart
    bar_colors = [
        GREEN  if h <= 15 else
        BLUE   if h <= 30 else
        ORANGE if h <= 60 else RED
        for h in summary["headway_min"]
    ]
    bars = ax_bar.barh(summary["route_short_name"], summary["headway_min"],
                       color=bar_colors, alpha=0.85, height=0.6)
    ax_bar.set_xlabel("Avg. headway (min)")
    ax_bar.set_title(f"Service frequency & hours  ({n} routes in area)", fontsize=12)
    for bar, row in zip(bars, summary.itertuples()):
        x = bar.get_width()
        ax_bar.text(x + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{row.headway_min} min  |  {row.first_str}–{row.last_str}  |  {row.total_trips} trips",
                    va="center", fontsize=7.5, color="#333")
    ax_bar.legend(handles=[
        mpatches.Patch(color=GREEN,  label="≤15 min (frequent)"),
        mpatches.Patch(color=BLUE,   label="16–30 min"),
        mpatches.Patch(color=ORANGE, label="31–60 min"),
        mpatches.Patch(color=RED,    label=">60 min (infrequent)"),
    ], fontsize=8, loc="lower right")
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, summary["headway_min"].max() * 1.45)
    ax_bar.grid(True, alpha=0.3)

    # Heatmap
    hours = list(range(24))
    pivot = (
        st.groupby(["route_short_name", "hour"])
        .agg(trips=("trip_id", "nunique"))
        .reset_index()
        .pivot(index="route_short_name", columns="hour", values="trips")
        .fillna(0)
        .reindex(columns=hours, fill_value=0)
    )
    pivot = pivot.reindex(summary["route_short_name"]).head(20)
    im = ax_heat.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest")
    ax_heat.set_xticks(range(24))
    ax_heat.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax_heat.set_yticks(range(len(pivot)))
    ax_heat.set_yticklabels(pivot.index, fontsize=8)
    ax_heat.set_xlabel("Hour of day")
    ax_heat.set_title("Trips per hour by route (darker = more service)", fontsize=10)
    fig.colorbar(im, ax=ax_heat, label="# trips", shrink=0.8)

    fig.tight_layout(pad=2.0)

    csv_df = summary[["route_short_name", "first_str", "last_str",
                       "headway_min", "total_trips"]].copy()
    csv_df.columns = ["Route", "First service", "Last service",
                      "Avg headway (min)", "Total trips"]
    return _fig_to_bytes(fig), csv_df


# ── Chart 4a: POGOH station map ───────────────────────────────────────────────

def chart4a_pogoh_map(pogoh_stations, pogoh_trips, lat, lon, radius_m):
    """
    Square map of POGOH stations within the study area.
    Departure stations → POGOH navy (#003865)
    Arrival stations   → POGOH teal  (#00B2A9)
    Bubble size ∝ total trip activity.
    """
    stations_in = filter_by_radius(
        pogoh_stations.dropna(subset=["Latitude", "Longitude"]),
        lat, lon, radius_m,
        lat_col="Latitude", lon_col="Longitude",
    )
    if stations_in.empty:
        return _empty_chart("No POGOH stations found within range")

    ids_in = set(stations_in["Id"].astype(str))
    dep = (pogoh_trips[pogoh_trips["Start Station Id"].isin(ids_in)]
           .groupby("Start Station Id").size().rename("departures"))
    arr = (pogoh_trips[pogoh_trips["End Station Id"].isin(ids_in)]
           .groupby("End Station Id").size().rename("arrivals"))

    usage = (
        stations_in[["Id", "Name", "Latitude", "Longitude", "Total Docks", "dist_m"]]
        .set_index("Id").join(dep, how="left").join(arr, how="left")
        .fillna(0).reset_index()
    )
    usage["total"] = usage["departures"] + usage["arrivals"]
    max_total      = max(usage["total"].max(), 1)
    # Smaller bubbles: base 80, max extra 220 (was 500+50)
    sizes          = (usage["total"] / max_total * 220 + 80).values

    fig, ax = plt.subplots(figsize=(8, 8))
    osm_ok  = _setup_map_ax(ax, lon, lat, radius_m)

    if osm_ok:
        cx, cy        = _to_merc([lon], [lat])
        cx, cy        = cx[0], cy[0]
        cir_x, cir_y  = _circle_merc(lon, lat, radius_m)

        ax.plot(cir_x, cir_y, color=POGOH_DARK, linewidth=2,
                linestyle="--", zorder=3, alpha=0.85)
        ax.fill(cir_x, cir_y, color=POGOH_TEAL, alpha=0.06, zorder=2)

        sx, sy = _to_merc(usage["Longitude"], usage["Latitude"])

        # Single circle per station: navy fill, teal border, sized by activity
        sc = ax.scatter(sx, sy, s=sizes, c=usage["total"],
                        cmap="Blues", vmin=0, vmax=max_total,
                        zorder=5, alpha=0.92,
                        edgecolors=POGOH_TEAL, linewidths=2.5)

        for i in range(len(usage)):
            ax.annotate(usage.iloc[i]["Name"], (sx[i], sy[i]),
                        fontsize=6, color="#111", alpha=0.88,
                        xytext=(8, 3), textcoords="offset points", zorder=7)

        ax.scatter([cx], [cy], c=RED, s=220, zorder=8, marker="*",
                   label=f"Center ({lat:.4f}, {lon:.4f})")
    else:
        _fallback_limits(ax, lon, lat, radius_m)
        deg = _deg(radius_m)
        ax.add_patch(plt.Circle((lon, lat), deg, color=POGOH_DARK, fill=False,
                                linewidth=1.5, linestyle="--", zorder=2))
        sc = ax.scatter(usage["Longitude"], usage["Latitude"],
                        s=sizes, c=usage["total"], cmap="Blues",
                        zorder=4, alpha=0.88,
                        edgecolors=POGOH_TEAL, linewidths=1.5)
        for _, row in usage.iterrows():
            ax.annotate(row["Name"], (row["Longitude"], row["Latitude"]),
                        fontsize=6, color="#222", alpha=0.85,
                        xytext=(6, 3), textcoords="offset points")
        ax.scatter([lon], [lat], c=RED, s=160, zorder=5, marker="*",
                   label=f"Center ({lat:.4f}, {lon:.4f})")

    fig.colorbar(sc, ax=ax, label="Total trips (dep + arr)", shrink=0.75)

    # Legend
    legend_handles = [
        mpatches.Patch(color=RED,        label=f"Center ({lat:.4f}, {lon:.4f})"),
        mpatches.Patch(color=POGOH_DARK, label="POGOH station (navy = high activity)"),
        mpatches.Patch(color=POGOH_TEAL, label="POGOH station ring"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"POGOH Stations within {radius_m} m  ({len(stations_in)} stations)\n"
        "Bubble size & color = trip activity",
        fontsize=12,
    )
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ── Chart 4b: POGOH usage bar chart ──────────────────────────────────────────

def chart4b_pogoh_usage(pogoh_stations, pogoh_trips, lat, lon, radius_m):
    """
    Horizontal bar chart: departures vs arrivals per POGOH station.
    Uses POGOH official colors: navy for departures, teal for arrivals.
    Returns (png_bytes, csv_df).
    """
    stations_in = filter_by_radius(
        pogoh_stations.dropna(subset=["Latitude", "Longitude"]),
        lat, lon, radius_m,
        lat_col="Latitude", lon_col="Longitude",
    )
    if stations_in.empty:
        return _empty_chart("No POGOH stations found within range"), pd.DataFrame()

    ids_in = set(stations_in["Id"].astype(str))
    dep = (pogoh_trips[pogoh_trips["Start Station Id"].isin(ids_in)]
           .groupby("Start Station Id").size().rename("departures"))
    arr = (pogoh_trips[pogoh_trips["End Station Id"].isin(ids_in)]
           .groupby("End Station Id").size().rename("arrivals"))

    usage = (
        stations_in[["Id", "Name", "Total Docks", "dist_m"]]
        .set_index("Id").join(dep, how="left").join(arr, how="left")
        .fillna(0).reset_index()
    )
    usage["total"]       = usage["departures"] + usage["arrivals"]
    usage["Total Docks"] = pd.to_numeric(usage["Total Docks"], errors="coerce").fillna(1)
    usage["usage_rate"]  = (usage["total"] / usage["Total Docks"]).round(1)
    usage                = usage.sort_values("total", ascending=False)
    usage["short_name"]  = usage["Name"].apply(
        lambda n: n if len(n) <= 30 else n[:28] + "…"
    )

    n       = len(usage)
    fig_h   = max(4, n * 0.6 + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    bar_h = 0.35
    y     = range(n)
    ax.barh([i + bar_h / 2 for i in y], usage["departures"],
            height=bar_h, color=POGOH_DARK, alpha=0.88, label="Departures")
    ax.barh([i - bar_h / 2 for i in y], usage["arrivals"],
            height=bar_h, color=POGOH_TEAL, alpha=0.88, label="Arrivals")

    ax.set_yticks(list(y))
    ax.set_yticklabels(usage["short_name"], fontsize=9)
    ax.set_xlabel("Number of trips")
    ax.set_title(
        f"POGOH Bike Share Usage  ({n} stations within {radius_m} m)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.35)

    max_val = max(usage["departures"].max(), usage["arrivals"].max(), 1)
    for i, row in enumerate(usage.itertuples()):
        ax.text(max_val * 1.02, i,
                f"rate: {row.usage_rate}×",
                va="center", fontsize=8, color=GRAY)
    ax.set_xlim(0, max_val * 1.28)

    fig.tight_layout()

    csv_df = usage[["Name", "Total Docks", "departures", "arrivals",
                    "total", "usage_rate"]].copy()
    csv_df.columns = ["Station", "Total Docks", "Departures",
                      "Arrivals", "Total trips", "Usage rate (trips/dock)"]
    return _fig_to_bytes(fig), csv_df
