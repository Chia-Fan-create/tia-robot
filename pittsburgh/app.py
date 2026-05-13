"""
app.py — Pittsburgh Transit Analyzer
Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import base64
import io
import traceback
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

from utils.data_loader import load_gtfs, load_pogoh
from utils.charts import (
    chart1_bus_stops,
    chart2_table,
    chart2_map,
    chart3_schedule,
    chart4a_pogoh_map,
    chart4b_pogoh_usage,
)

app = Flask(__name__)

# ── Pre-load all data at startup (loaded once, reused for every request) ──────
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
    print(f"[WARN] GTFS not found: {e}")
    GTFS    = None
    GTFS_OK = False

print("Loading POGOH data ...")
try:
    POGOH_STATIONS, POGOH_TRIPS = load_pogoh()
    print(f"  stations: {len(POGOH_STATIONS)} rows")
    print(f"  trips:    {len(POGOH_TRIPS)} rows")
    POGOH_OK = True
except FileNotFoundError as e:
    print(f"[WARN] POGOH not found: {e}")
    POGOH_STATIONS = POGOH_TRIPS = None
    POGOH_OK       = False

print("Ready — visit http://127.0.0.1:5000\n")


# ── Helper ────────────────────────────────────────────────────────────────────

def _b64(png_bytes):
    """Encode PNG bytes as a data URI for embedding in JSON."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _df_to_csv_b64(df):
    """Encode a DataFrame as a base64 CSV data URI."""
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

        if GTFS_OK:
            # Chart 1 — bus stop map
            result["chart1"] = _b64(chart1_bus_stops(GTFS, lat, lon, radius))

            # Chart 2 — route table (separate image) + route map (separate image)
            tbl_png, df2 = chart2_table(GTFS, lat, lon, radius)
            result["chart2_table"]     = _b64(tbl_png)
            result["chart2_map"]       = _b64(chart2_map(GTFS, lat, lon, radius))
            result["chart2_csv"]       = _df_to_csv_b64(df2) if not df2.empty else None

            # Chart 3 — frequency heatmap
            png3, df3 = chart3_schedule(GTFS, lat, lon, radius)
            result["chart3"]     = _b64(png3)
            result["chart3_csv"] = _df_to_csv_b64(df3) if not df3.empty else None
        else:
            result["gtfs_error"] = "GTFS data not loaded. Place files in data/gtfs/"

        if POGOH_OK:
            # Chart 4a — POGOH station map
            result["chart4a"] = _b64(
                chart4a_pogoh_map(POGOH_STATIONS, POGOH_TRIPS, lat, lon, radius)
            )
            # Chart 4b — POGOH usage bar chart
            png4b, df4b = chart4b_pogoh_usage(POGOH_STATIONS, POGOH_TRIPS, lat, lon, radius)
            result["chart4b"]     = _b64(png4b)
            result["chart4b_csv"] = _df_to_csv_b64(df4b) if not df4b.empty else None
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
