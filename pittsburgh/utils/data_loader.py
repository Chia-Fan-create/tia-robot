"""
data_loader.py
負責從本機 ./data/ 讀取已下載的 GTFS 及 POGOH 資料。
GTFS 資料請放在 ./data/gtfs/ 目錄下（解壓 general_transit_Bing.zip）
POGOH 資料請放在 ./data/ 目錄下
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GTFS_DIR = os.path.join(DATA_DIR, "gtfs")


def _gtfs(filename):
    path = os.path.join(GTFS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}，請先將 GTFS zip 解壓到 data/gtfs/ 目錄"
        )
    return pd.read_csv(path, dtype=str,
                       quoting=0, on_bad_lines="warn", engine="python")


def load_gtfs():
    """讀取 GTFS 各檔，回傳 dict of DataFrames。"""
    stops      = _gtfs("stops.txt")
    stop_times = _gtfs("stop_times.txt")
    trips      = _gtfs("trips.txt")
    routes     = _gtfs("routes.txt")
    shapes     = _gtfs("shapes.txt")

    # stops: 經緯度前有空格，統一清理
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"].str.strip(), errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"].str.strip(), errors="coerce")

    # trips: 原始檔第一欄有 typo "oute_id"，強制修正
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


def load_pogoh():
    """讀取 POGOH 站點與租借紀錄，回傳 (stations_df, trips_df)。"""
    station_path = os.path.join(DATA_DIR, "pogoh_stations.csv")
    trips_path   = os.path.join(DATA_DIR, "pogoh_trips.csv")

    for p in (station_path, trips_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"找不到 {p}，請從 WPRDC 下載後分別命名為\n"
                "  pogoh_stations.csv\n  pogoh_trips.csv\n"
                "放到 data/ 目錄"
            )

    stations = pd.read_csv(station_path)
    trips    = pd.read_csv(trips_path, low_memory=False)

    stations["Latitude"]  = pd.to_numeric(stations["Latitude"],  errors="coerce")
    stations["Longitude"] = pd.to_numeric(stations["Longitude"], errors="coerce")
    stations["Id"]        = stations["Id"].astype(str)

    trips["Start Station Id"] = trips["Start Station Id"].astype(str)
    trips["End Station Id"]   = trips["End Station Id"].astype(str)
    trips["Start Date"]       = pd.to_datetime(trips["Start Date"], errors="coerce")
    trips["End Date"]         = pd.to_datetime(trips["End Date"],   errors="coerce")
    trips["Duration"]         = pd.to_numeric(trips["Duration"],    errors="coerce")

    return stations, trips
