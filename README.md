# TIA-ROBOT — Transit Information Analyzer

A web-based transit analysis tool that visualizes bus stops, routes, service frequency, and bike share usage for a user-defined study area.

Built with Flask and Matplotlib. The same system has been applied to two cities using their respective open data sources.



## Cities

### 📍 Taipei, Taiwan
Original prototype developed as an internal tool, applied to Taipei and New Taipei City bus networks and YouBike stations.

→ See [`taipei/TIA_ROBOT_prototype.pdf`](taipei/TIA_ROBOT_prototype.pdf) for the full prototype presentation.

---

###  📍 Pittsburgh, USA
Rebuilt from scratch using Pittsburgh Regional Transit (PRT) GTFS data and POGOH bike share data.

→ See [`pittsburgh/`](pittsburgh/) for the full source code and setup instructions.

**Charts generated:**
- Bus stop locations within a user-defined radius (OpenStreetMap basemap)
- Routes serving the study area — table and map
- Service frequency and operating hours heatmap
- POGOH bike share station map and usage statistics

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | Python, Flask |
| Data processing | Pandas |
| Visualization | Matplotlib, Contextily |
| Frontend | HTML, CSS, Vanilla JS |
| Map tiles | CartoDB Positron (via contextily) |
| Data sources | GTFS (PRT), WPRDC Open Data |

---

## Project Structure

```
tia-robot/
├── README.md
├── pittsburgh/
│   ├── app.py
│   ├── requirements.txt
│   ├── data/
│   │   ├── gtfs/               # PRT GTFS files (not tracked in git)
│   │   ├── pogoh_stations.csv  # not tracked in git
│   │   └── pogoh_trips.csv     # not tracked in git
│   ├── static/
│   │   └── icons/
│   │       └── bus.png
│   ├── templates/
│   │   └── index.html
│   └── utils/
│       ├── __init__.py
│       ├── charts.py
│       └── data_loader.py
└── taipei/
    └── TIA_ROBOT_prototype.pdf
```

## Installation

```bash
cd pgh_transit
pip install -r requirements.txt
```

## Data Resources

### GTFS Bus Data
1. Download `general_transit_Bing.zip` from https://www.rideprt.org/developerresources/
2. Unzip and place all files into `data/gtfs/`

```
data/
  gtfs/
    stops.txt
    stop_times.txt
    trips.txt
    routes.txt
    shapes.txt
    calendar.txt
    ... (other .txt files)
```

### POGOH Bike Share Data
1. Download the CSV from https://data.wprdc.org/dataset/station-locations and save it as `data/pogoh_stations.csv`
2. Download the CSV from https://data.wprdc.org/dataset/pogoh-trip-data and save it as `data/pogoh_trips.csv`

```
data/
  pogoh_stations.csv
  pogoh_trips.csv
```

## Usage

Start the server:

```bash
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

1. Enter the latitude and longitude of your study center point
2. Adjust the search radius (default: 300 m) using the slider or input field
3. Click **Run Analysis** and wait approximately 10–20 seconds
4. Four charts will appear below:
   - **Chart 1** — Bus stop locations within the study area
   - **Chart 2a / 2b** — Routes serving the area (table + map)
   - **Chart 3** — Service frequency and operating hours heatmap
   - **Chart 4a / 4b** — POGOH bike share station map and usage

## Pittsburgh Quick Coordinates

| Location | Latitude | Longitude |
|----------|----------|-----------|
| Downtown | 40.4417 | -79.9581 |
| Oakland / CMU | 40.4441 | -79.9602 |
| East Liberty | 40.4561 | -79.9163 |
| South Side | 40.4285 | -80.0025 |
