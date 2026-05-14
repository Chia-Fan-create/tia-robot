"""
app.py — Pittsburgh Transit Analyzer
Run:  python app.py
Open: http://127.0.0.1:5000
"""

import base64
import io
import traceback
import pandas as pd
from flask import Flask, render_template, request, jsonify

from utils.data_loader import load_gtfs, load_pogoh, precompute_pogoh
from utils.charts import (
    chart1_bus_stops,
    chart2_table,
    chart2_map,
    chart3_schedule,
    chart4a_pogoh_map,
    chart4b_turnover_heatmap,
    chart4c_balance,
)

app = Flask(__name__)


# ── Startup: load and pre-compute all data once ───────────────────────────────

print("Loading GTFS data ...")
try:
    GTFS = load_gtfs()
    print(f"  stops:      {len(GTFS['stops'])} rows")
    print(f"  stop_times: {len(GTFS['stop_times'])} rows")
    print(f"  trips:      {len(GTFS['trips'])} rows")
    print(f"  routes:     {len(GTFS['routes'])} rows")
    print(f"  shapes:     {len(GTFS['shapes'])} rows")
    GTFS_OK = True
except FileNotFoundError as e:
    print(f"[WARN] {e}")
    GTFS    = None
    GTFS_OK = False

print("Loading POGOH data ...")
try:
    POGOH_STATIONS, POGOH_TRIPS = load_pogoh()
    print(f"  stations: {len(POGOH_STATIONS)} rows")
    print(f"  trips:    {len(POGOH_TRIPS)} rows (NORMAL only)")

    print("Pre-computing POGOH metrics ...")
    POGOH_PRE = precompute_pogoh(POGOH_STATIONS, POGOH_TRIPS)
    print(f"  data period: {POGOH_PRE['data_days']} days")
    print(f"  hourly pivot: {POGOH_PRE['hourly'].shape}")
    POGOH_OK = True
except FileNotFoundError as e:
    print(f"[WARN] {e}")
    POGOH_PRE = None
    POGOH_OK  = False

print("Ready — visit http://127.0.0.1:5000\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _b64(png_bytes):
    """Encode PNG bytes as a base64 data URI for embedding in JSON."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _df_to_csv_b64(df):
    """Encode a DataFrame as a base64 CSV data URI for browser download."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    raw = buf.getvalue().encode("utf-8")
    return "data:text/csv;base64," + base64.b64encode(raw).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", gtfs_ok=GTFS_OK, pogoh_ok=POGOH_OK)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data   = request.get_json()
        lat    = float(data["lat"])
        lon    = float(data["lon"])
        radius = float(data.get("radius", 300))

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({"error": "Invalid coordinates"}), 400
        if not (50 <= radius <= 5000):
            return jsonify({"error": "Radius must be between 50 and 5000 m"}), 400

        result = {}

        # ── GTFS charts ───────────────────────────────────────────────────────
        if GTFS_OK:
            result["chart1"] = _b64(chart1_bus_stops(GTFS, lat, lon, radius))

            tbl_png, df2 = chart2_table(GTFS, lat, lon, radius)
            result["chart2_table"] = _b64(tbl_png)
            result["chart2_map"]   = _b64(chart2_map(GTFS, lat, lon, radius))
            result["chart2_csv"]   = _df_to_csv_b64(df2) if not df2.empty else None

            png3, df3 = chart3_schedule(GTFS, lat, lon, radius)
            result["chart3"]     = _b64(png3)
            result["chart3_csv"] = _df_to_csv_b64(df3) if not df3.empty else None
        else:
            result["gtfs_error"] = "GTFS data not loaded. Place files in data/gtfs/"

        # ── POGOH charts (use pre-computed data) ──────────────────────────────
        if POGOH_OK:
            result["chart4a"] = _b64(
                chart4a_pogoh_map(POGOH_PRE, lat, lon, radius)
            )

            png4b, df4b = chart4b_turnover_heatmap(POGOH_PRE, lat, lon, radius)
            result["chart4b"]     = _b64(png4b)
            result["chart4b_csv"] = _df_to_csv_b64(df4b) if not df4b.empty else None

            png4c, df4c = chart4c_balance(POGOH_PRE, lat, lon, radius)
            result["chart4c"]     = _b64(png4c)
            result["chart4c_csv"] = _df_to_csv_b64(df4c) if not df4c.empty else None
        else:
            result["pogoh_error"] = "POGOH data not loaded. Place CSV files in data/"

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
