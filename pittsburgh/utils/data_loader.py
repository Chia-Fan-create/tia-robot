"""
data_loader.py
Loads GTFS and POGOH data from local files, and pre-computes
all POGOH station metrics at startup so chart generation is fast.

Directory layout expected:
    data/gtfs/              GTFS .txt files (unzipped from general_transit_Bing.zip)
    data/pogoh_stations.csv Downloaded from WPRDC station-locations dataset
    data/pogoh_trips.csv    Downloaded from WPRDC pogoh-trip-data dataset
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GTFS_DIR = os.path.join(DATA_DIR, "gtfs")


# ── GTFS ──────────────────────────────────────────────────────────────────────

def _read_gtfs_file(filename):
    """Read a single GTFS .txt file into a DataFrame (all columns as str)."""
    path = os.path.join(GTFS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GTFS file not found: {path}\n"
            "Please unzip general_transit_Bing.zip into data/gtfs/"
        )
    return pd.read_csv(path, dtype=str,
                       quoting=0, on_bad_lines="warn", engine="python")


def load_gtfs():
    """
    Load all required GTFS files and return a dict of DataFrames.
    Applies type coercions and fixes a known typo in trips.txt.
    """
    stops      = _read_gtfs_file("stops.txt")
    stop_times = _read_gtfs_file("stop_times.txt")
    trips      = _read_gtfs_file("trips.txt")
    routes     = _read_gtfs_file("routes.txt")
    shapes     = _read_gtfs_file("shapes.txt")

    # stops.txt: lat/lon values have leading whitespace — strip before converting
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"].str.strip(), errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"].str.strip(), errors="coerce")

    # trips.txt: first column header is "oute_id" (missing leading 'r') — fix it
    cols = list(trips.columns)
    if cols[0] != "route_id":
        cols[0] = "route_id"
        trips.columns = cols

    shapes["shape_pt_lat"]      = pd.to_numeric(shapes["shape_pt_lat"],      errors="coerce")
    shapes["shape_pt_lon"]      = pd.to_numeric(shapes["shape_pt_lon"],      errors="coerce")
    shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce")

    return {
        "stops":      stops,
        "stop_times": stop_times,
        "trips":      trips,
        "routes":     routes,
        "shapes":     shapes,
    }


# ── POGOH raw load ─────────────────────────────────────────────────────────────

def load_pogoh():
    """
    Load POGOH station locations and trip records.
    Returns (stations_df, trips_df) after type coercions.
    Only NORMAL trips are kept — GRACE_PERIOD entries are excluded
    because they represent aborted rentals (bike checked out and
    immediately returned to the same station).
    """
    station_path = os.path.join(DATA_DIR, "pogoh_stations.csv")
    trips_path   = os.path.join(DATA_DIR, "pogoh_trips.csv")

    for p in (station_path, trips_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"POGOH file not found: {p}\n"
                "Please download from WPRDC and rename to:\n"
                "  pogoh_stations.csv\n"
                "  pogoh_trips.csv\n"
                "then place them in the data/ directory."
            )

    stations = pd.read_csv(station_path)
    trips    = pd.read_csv(trips_path, low_memory=False)

    # Type coercions for stations
    stations["Latitude"]    = pd.to_numeric(stations["Latitude"],    errors="coerce")
    stations["Longitude"]   = pd.to_numeric(stations["Longitude"],   errors="coerce")
    stations["Total Docks"] = pd.to_numeric(stations["Total Docks"], errors="coerce")
    stations["Id"]          = stations["Id"].astype(str)

    # Type coercions for trips
    trips["Start Station Id"] = trips["Start Station Id"].astype(str)
    trips["End Station Id"]   = trips["End Station Id"].astype(str)
    trips["Start Date"]       = pd.to_datetime(trips["Start Date"], errors="coerce")
    trips["End Date"]         = pd.to_datetime(trips["End Date"],   errors="coerce")
    trips["Duration"]         = pd.to_numeric(trips["Duration"],    errors="coerce")

    # Drop grace-period rows (bike returned immediately — not real usage)
    trips = trips[trips["Closed Status"] == "NORMAL"].reset_index(drop=True)

    return stations, trips


# ── POGOH pre-computation ──────────────────────────────────────────────────────

def precompute_pogoh(stations, trips):
    """
    Pre-compute all POGOH metrics for every station at startup.
    Called once; results are passed directly to chart functions,
    so chart requests only need to filter — not re-aggregate.

    Returns a dict with keys:
        "stations"      — stations DataFrame enriched with summary metrics
        "hourly"        — (station_id, hour) → departure_count pivot (60 × 24)
        "turnover"      — (station_id, hour) → hourly turnover rate pivot (60 × 24)
        "balance"       — per-station supply/demand balance metrics
        "data_days"     — number of unique days covered by the trip data
    """
    # Number of unique calendar days in the dataset
    # Used to normalise turnover rate to a per-day figure
    data_days = max(trips["Start Date"].dt.date.nunique(), 1)

    # ── Per-station departure and arrival totals ───────────────────────────────
    dep_total = (trips.groupby("Start Station Id").size()
                 .rename("departures").reset_index()
                 .rename(columns={"Start Station Id": "Id"}))
    arr_total = (trips.groupby("End Station Id").size()
                 .rename("arrivals").reset_index()
                 .rename(columns={"End Station Id": "Id"}))

    enriched = (stations
                .merge(dep_total, on="Id", how="left")
                .merge(arr_total, on="Id", how="left"))
    enriched["departures"] = enriched["departures"].fillna(0).astype(int)
    enriched["arrivals"]   = enriched["arrivals"].fillna(0).astype(int)
    enriched["total"]      = enriched["departures"] + enriched["arrivals"]

    # ── Supply/demand balance metrics ─────────────────────────────────────────
    # net_flow > 0 → more bikes leave than arrive (net outflow station, e.g. morning origin)
    # net_flow < 0 → more bikes arrive than leave  (net inflow station,  e.g. morning destination)
    enriched["net_flow"]       = enriched["departures"] - enriched["arrivals"]
    enriched["balance_ratio"]  = (
        enriched["net_flow"] /
        enriched["total"].replace(0, 1)          # avoid division by zero
    ).round(3)

    # ── Hourly departure counts per station ───────────────────────────────────
    trips_h = trips.copy()
    trips_h["hour"] = trips_h["Start Date"].dt.hour   # 0–23

    hourly_counts = (
        trips_h.groupby(["Start Station Id", "hour"])
        .size()
        .reset_index(name="dep_count")
        .rename(columns={"Start Station Id": "Id"})
    )

    # Pivot to (station × hour) matrix; fill missing hours with 0
    hourly_pivot = (
        hourly_counts
        .pivot(index="Id", columns="hour", values="dep_count")
        .reindex(columns=range(24), fill_value=0)
        .fillna(0)
    )

    # ── Hourly turnover rate = departures per hour / Total Docks ──────────────
    # Represents how many times each dock slot is used in a given hour
    docks_series = stations.set_index("Id")["Total Docks"].fillna(1)
    turnover_pivot = hourly_pivot.div(docks_series, axis=0).round(3)

    # ── Balance metrics DataFrame (one row per station) ───────────────────────
    balance_df = enriched[["Id", "Name", "Total Docks",
                            "departures", "arrivals",
                            "net_flow", "balance_ratio"]].copy()

    return {
        "stations":   enriched,
        "hourly":     hourly_pivot,
        "turnover":   turnover_pivot,
        "balance":    balance_df,
        "data_days":  data_days,
    }
