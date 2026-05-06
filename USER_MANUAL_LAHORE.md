# User Manual
## Lahore School Simulation Dashboard
### Agent-Based Modeling for Out-of-School Children

---

**Document version:** 1.0  
**System:** EEP Dashboard — Lahore Simulation Page  
**Audience:** Research analysts, policy officers, EEP team members

---

## Table of Contents

1. Overview
2. Theoretical Background
3. System Architecture
4. Getting Started
5. Step-by-Step Workflow
   - Step 1: Generate Simulation
   - Step 2: Fetch Road Distances & Travel Times
   - Step 3: Run Enrollment Inference
6. Reading the Map
7. Reading the Charts
8. Reading the Results Table
9. Downloading Results
10. Data Sources & Limitations
11. Technical Reference
12. Frequently Asked Questions

---

## 1. Overview

The **Lahore School Simulation** is a page within the EEP (Equity equation project) Dashboard. It uses **Agent-Based Modeling (ABM)** to simulate the school enrollment decisions of households in Lahore, Pakistan.

Unlike a statistical model that aggregates populations, ABM treats each household as an independent **agent** that makes its own enrollment decision based on its own characteristics — income, distance to school, route safety, parental literacy, and available transport. This allows the simulation to capture spatial inequality: two households in different neighborhoods of Lahore, even with similar income, may have very different enrollment probabilities simply because the road distance to their nearest school differs.

The page specifically models **Out-of-School Children (OOSC)** risk — which households are least likely to enroll their children — by combining:

- Real school locations from the POIs (Points of Interest) dataset
- Realistic household placement around those schools
- Actual road distances computed from OpenStreetMap road network data
- A trained machine learning model that predicts enrollment probability

---

## 2. Theoretical Background

### 2.1 Agent-Based Modeling (ABM)

Agent-Based Modeling is a computational simulation method where individual entities (agents) follow defined rules and interact with their environment. In this dashboard:

- Each **household** is one agent
- The agent's behavior (enrolling or not enrolling a child) is determined by its attributes
- There is no central decision — 10,000 agents run independently

ABM is particularly useful for education policy analysis because enrollment decisions are not made at the city level; they are made by individual families facing their own specific constraints.

### 2.2 Haversine Distance (Straight-Line)

The **Haversine formula** calculates the great-circle distance between two points on the surface of a sphere (the Earth). It accounts for the curvature of the Earth, making it more accurate than simple Euclidean distance for geographic coordinates.

The formula is:

```
a = sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)
d = 2R × arctan2(√a, √(1−a))
```

Where R = 6,371 km (Earth's radius), lat and lon are in radians.

In this simulation, Haversine is used in two places:
1. **Instant nearest-school assignment** — for all 10,000 households against all ~3,000 schools simultaneously, using vectorized NumPy operations. This finds which school is geographically closest to each household.
2. **Fallback distance** — if the road routing API is unavailable, the Haversine distance is used instead.

Haversine gives the **shortest possible distance** (as the crow flies). Actual road distances are always longer.

### 2.3 Road Distance via OSRM

The **Open Source Routing Machine (OSRM)** is a free, open-source routing engine based on OpenStreetMap data. It computes realistic driving distances and travel times along actual roads — accounting for road network topology, one-way streets, and permitted routes.

The simulation uses the **OSRM Table API**, which accepts a batch of source and destination coordinates in a single HTTP request and returns a complete matrix of distances and travel times.

For 10,000 households, the simulation sends approximately 100 requests (100 households per request), receiving a distance matrix per batch. This reduces the total API calls from 10,000 down to ~100, cutting processing time from hours to under one minute.


### 2.4 Distance Categories

Road distances are grouped into four categories that the enrollment model was trained on:

| Category | Road Distance | Typical Travel Time |
|---|---|---|
| Near | 0–2 km | 5–15 minutes |
| Moderate | 2–6 km | 20–45 minutes |
| Far | 6–15 km | 50–90 minutes |
| Very Far | 15+ km | 100–180 minutes |



### 2.5 Income and PSLM Quintiles

Household income is one of the key predictors of school enrollment. In this simulation, income is assigned to each household by sampling from **PSLM (Pakistan Social and Living Standards Measurement) income quintiles**.

The PSLM is a nationally representative household survey conducted by the Pakistan Bureau of Statistics. It reports household consumption and income data across five quintiles (20% of households each, ranked from poorest to richest).

In this simulation, exactly **20% of households are assigned to each quintile**, and income within each quintile is drawn uniformly from the quintile's [minimum, maximum] range:

| Quintile | Label | Income Range (PKR/month) |
|---|---|---|
| 1st (poorest 20%) | Q1 (0–20%) | 5,000 – 18,000 |
| 2nd | Q2 (20–40%) | 18,000 – 32,000 |
| 3rd (middle) | Q3 (40–60%) | 32,000 – 50,000 |
| 4th | Q4 (60–80%) | 50,000 – 80,000 |
| 5th (richest 20%) | Q5 (80–100%) | 80,000 – 200,000 |

> **Important:** The income boundaries above are estimates based on published PSLM/HIES summaries for Lahore urban. They have not been computed directly from PSLM microdata. When real microdata is available, the `PSLM_QUINTILES` list in `lahore_simulation.py` should be updated with actual percentile cut-points.

The income value assigned to each household is then **z-score normalized** before being passed to the enrollment model, using Lahore-urban reference statistics (mean = PKR 48,412, std = PKR 13,309). This normalization is required because the model was trained on normalized values.

### 2.6 Other Household Features

In addition to income and distance, each agent is assigned the following attributes using probabilistic sampling based on Lahore-urban distributions:

| Feature | Distribution | Interpretation |
|---|---|---|
| Route safety | Binomial (p=0.78) | Whether the route to school is considered safe (1=safe, 0=unsafe) |
| Head can read/write | Binomial (p=0.62) | Whether the household head is literate |
| Head can solve math | Binomial (p=0.54) | Whether the household head has basic numeracy |
| School facilities score | Beta(2,3) × 5 | Score 0–5 reflecting quality of facilities at the nearest school |



### 2.7 The Enrollment Inference Model (Three-Phase Pipeline)

The enrollment probability for each household is computed by a trained three-phase machine learning pipeline:

**Phase 1 — Neural Network Classifiers**

Five separate neural network models each predict one intermediate outcome from the 10 input features. These intermediate predictions capture different dimensions of the enrollment decision (e.g., whether the household is likely to cite distance as a barrier, whether income is a constraint, etc.).

**Phase 2 — Autoencoder (Dimensionality Reduction)**

The five Phase 1 predictions are compressed into three latent variables (z₁, z₂, z₃) by a trained autoencoder. This step reduces noise and captures the underlying structure in the Phase 1 outputs. The three latent variables represent abstract behavioral patterns learned from real survey data.

**Phase 3 — Gradient Boosted Classifier**

The original 10 input features plus the three autoencoder latent variables (13 features total) are fed into a gradient boosted tree model (XGBoost or scikit-learn GBM). This model outputs the final enrollment probability as a value between 0 and 1.

The full pipeline:

```
10 input features
       │
       ▼
  Phase 1 (5 neural nets)
       │
       ▼ 5 intermediate predictions
  Autoencoder → 3 latent variables (z₁, z₂, z₃)
       │
       ▼ 10 original + 3 latent = 13 features
  Phase 2 (gradient boosted classifier)
       │
       ▼
  Enrollment Probability [0.0 – 1.0]
       │
       ▼
  Final Probability
```

A probability above **0.50** means the model predicts the household is more likely than not to enroll their child. This threshold is used in the dashboard's summary metrics.

---

## 3. System Architecture

```
POIs_Schools.csv          (real school coordinates, ~3,000 schools)
        │
        ▼
load_all_schools()        reads CSV, drops invalid rows
        │
        ▼
generate_households()     places 10,000 agents randomly within 2 km of any school
        │
        ▼
fetch_routes()
  ├── _haversine_vectorized()  finds nearest school per household (instant, numpy)
  └── OSRM Table API           road distance + travel time (~100 HTTP requests)
        │
        ▼
prepare_for_inference()   assigns income (PSLM quintiles) + other features
        │
        ▼
run_batch_inference()     Phase 1 → Autoencoder → Phase 2 pipeline
        │
        ▼
build_lahore_map()        Plotly Scattermapbox on OpenStreetMap tiles
```

---

## 4. Getting Started (Running locally)

### Requirements

- Python 3.10 or higher
- All packages in `requirements.txt` installed
- Internet connection (required for OSRM routing API)

### Running the Dashboard

Open a terminal in the project folder and run:

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`. Navigate to **Lahore Simulation** in the left sidebar.

---

## 5. Step-by-Step Workflow

The Lahore simulation follows a strict three-step sequence. Each step must be completed before the next becomes available.

---

### Step 1: Generate Simulation

**Button:** `Generate Simulation for Lahore`

Click this button to begin. The system will:

1. Load all ~3,000 schools from `POIs_Schools.csv`. Each school record contains a name, FID (unique identifier), address, longitude, and latitude.

2. Randomly place **10,000 synthetic households** across Lahore. Each household is positioned within a 2 km radius of a randomly chosen school. The placement uses uniform distribution inside a disk — meaning households are not clustered at the school location but spread naturally around it.

3. Display the initial map showing all schools (blue dots) and all households (purple dots). At city-wide zoom (zoom level 11), dots will appear dense. Zoom in using the scroll wheel or pinch gesture to explore specific neighborhoods.

**What you see after Step 1:**
- Map with ~3,000 blue school markers
- 10,000 purple household markers
- Status bar: "*3,000 schools loaded | 10,000 households generated*"

> The purple color at this stage means no distance data has been fetched yet.

---

### Step 2: Fetch Road Distances & Travel Times

**Button:** `Fetch Road Distances & Travel Times`

This step contacts the OSRM routing server to compute the actual driving distance and travel time from each household to its nearest school.

**What happens internally:**

1. **Vectorized Haversine** runs first (takes less than 1 second). For all 10,000 households simultaneously, it computes the straight-line distance to every one of the ~3,000 schools and identifies the nearest school for each household. This uses NumPy matrix operations and requires no internet connection.

2. **OSRM Table API** is then called in batches of 100 households. For each batch:
   - The system identifies which unique schools are the nearest school for those 100 households
   - One HTTP request is sent to the OSRM server with all household and school coordinates
   - OSRM returns a distance matrix (road kilometres) and time matrix (minutes)
   - The relevant entry per household is extracted

3. The progress bar shows: `OSRM Table API: X/10,000 households …`

**Typical duration:** 30–60 seconds for 10,000 households.

**What you see after Step 2:**
- Map households re-coloured by **travel time** using the Plasma colour scale (dark purple = short travel time, bright yellow = long travel time)
- A colour bar appears on the right side of the map labelled "Travel Time (min)"
- Summary metrics appear:
  - **Households** — total count (10,000)
  - **Routes fetched** — how many had a valid OSRM response
  - **Avg road dist** — average road distance to nearest school (km)
  - **Avg travel time** — average travel time to nearest school (minutes)

> If some households show `—` for road distance, it means OSRM could not route to that school (e.g., the school coordinate is not on the road network). The Haversine straight-line distance is used as a fallback.

---

### Step 3: Run Enrollment Inference

**Button:** `Run Enrollment Inference` (blue, primary button)

This step runs the trained enrollment probability model on all 10,000 households.

**What happens internally:**

1. **`prepare_for_inference()`** enriches each household with the additional features the model needs:
   - Income sampled from PSLM quintile distributions (20% per quintile)
   - Travel mode, route safety, literacy, numeracy, school facilities assigned probabilistically
   - Road distance mapped to distance category (Near / Moderate / Far / Very Far)

2. **`run_batch_inference()`** runs the full three-phase pipeline (Phase 1 neural nets → Autoencoder → Phase 2 classifier) on all 10,000 households simultaneously using vectorized operations.

3. Route safety penalty is applied: households with `route_safe = 0` have their probability.

**Typical duration:** 5–15 seconds.

**What you see after Step 3:**
- Map households re-coloured by **enrollment probability** using the RdYlGn (Red-Yellow-Green) colour scale:
  - **Green** = high probability of enrollment (likely to enroll)
  - **Yellow** = uncertain (around 50%)
  - **Red** = low probability (unlikely to enroll, high OOSC risk)
- A fifth summary metric appears: **Avg enrollment prob** and **% of households ≥ 50%**
- Three charts appear in the Enrollment Probability Summary section
- The results table expands to include income, quintile, and enrollment probability columns

---

## 6. Reading the Map

The map is built on **OpenStreetMap** tiles, meaning it shows real roads, buildings, and landmarks for Lahore.

### Map Controls

| Action | Effect |
|---|---|
| Scroll wheel / pinch | Zoom in or out |
| Click and drag | Pan to a different area |
| Hover over a marker | Show tooltip with details |
| Double-click | Zoom in to that point |

### Map Layers

**Blue markers (small):** Schools from `POIs_Schools.csv`
- Hover to see school name and FID number

**Coloured markers (households):**

| Stage | Color meaning |
|---|---|
| After Step 1 (no data) | Solid purple — no distance data yet |
| After Step 2 (routes fetched) | Plasma scale — darker = shorter travel time, brighter = longer |
| After Step 3 (inference done) | RdYlGn scale — green = likely to enroll, red = unlikely |

### Household Hover Tooltip

When you hover over any household marker, a tooltip appears showing all available information for that agent:

```
Household #4231
School: Government Girls High School, Gulshan-e-Ravi
Income: PKR 27,450  Q2 (20–40%)
Straight dist: 0.847 km
Road dist: 1.293 km
Travel time: 4.2 min
Category: Near (0–2 km)
Enrollment prob: 73%
```

Fields only appear once they have been computed. Before Step 2, only the household ID and school name show. Before Step 3, income and enrollment probability do not appear.

### Exploring the Map

The map opens at **zoom level 11**, which shows all of Lahore city. At this zoom, individual household dots are small but visible. To explore a specific area:

1. Scroll to zoom in on a neighborhood (e.g., Gulberg, Model Town, Johar Town)
2. Dots spread out and become easier to distinguish
3. Colour patterns reveal which neighborhoods have high vs. low enrollment probability
4. School markers (blue) show how many schools exist in that area

Spatial patterns to look for:
- **Red clusters** (after Step 3) indicate neighborhoods at high OOSC risk — areas where households are unlikely to enroll children
- **Schools with no green households nearby** may indicate that even proximate schools are not attracting enrollment due to income or safety constraints
- **Dense purple clusters** (after Step 1) indicate areas with many schools and thus many simulated households

---

## 7. Reading the Charts

Three charts appear after Step 3 is complete.

### Chart 1: Enrollment Probability Distribution (Histogram)

Shows how enrollment probabilities are distributed across all 10,000 households. The x-axis is probability (0% to 100%), the y-axis is number of households.

- A distribution shifted **right** (toward 100%) means the simulated population is generally likely to enroll
- A distribution shifted **left** (toward 0%) indicates high overall OOSC risk
- A **bimodal distribution** (two peaks) suggests there are two distinct groups — one high-risk and one low-risk

### Chart 2: Average Probability by Distance Category (Bar Chart)

Shows the mean enrollment probability for households in each distance category.

- **Green bars** (≥ 50%) indicate that on average, households at that distance are predicted to enroll
- **Red bars** (< 50%) indicate net OOSC risk at that distance
- The pattern should generally show decreasing probability as distance increases — the steepness of this decline reflects how sensitive enrollment is to distance in this simulation

### Chart 3: Average Probability by Income Quintile (Bar Chart)

Shows the mean enrollment probability for households in each PSLM income quintile.

- **Q1 (poorest)** typically shows the lowest enrollment probability
- **Q5 (richest)** typically shows the highest
- A large gap between Q1 and Q5 indicates that income is a major driver of OOSC risk in this simulation
- If Q1 and Q5 probabilities are similar, it suggests that distance and route safety are dominating over income in this particular simulation run

---

## 8. Reading the Results Table

After Step 2, a table appears showing the first 500 of 10,000 rows (for display performance). After Step 3, the table expands with additional columns.

| Column | Description |
|---|---|
| HH ID | Unique household identifier (1 to 10,000) |
| Latitude | Household's geographic latitude (decimal degrees) |
| Longitude | Household's geographic longitude (decimal degrees) |
| Nearest School | Name of the nearest school by road distance |
| Straight (km) | Haversine straight-line distance to nearest school |
| Road (km) | OSRM road distance to nearest school |
| Travel Time (min) | Estimated driving time to nearest school |
| Income (PKR) | Monthly household income in Pakistani Rupees |
| PSLM Quintile | Which quintile the household's income falls in (Q1–Q5) |
| Enrollment Prob | Model-predicted probability of enrolling a child (0%–100%) |

The full 10,000-row dataset can be downloaded as a CSV file.

> **Note:** The table shows only the first 500 rows for browser performance. The downloaded CSV contains all 10,000 rows.

---

## 9. Downloading Results

After Step 2 or Step 3, a **Download Results as CSV** button appears at the bottom of the page.

The downloaded file `lahore_simulation_results.csv` contains all 10,000 households with all computed columns. This file can be:

- Opened in Microsoft Excel or Google Sheets
- Loaded into R or Python for further analysis
- Used to produce custom maps in QGIS or ArcGIS (using the lat/lon columns)
- Merged with real survey data by matching on geographic proximity

---

## 10. Data Sources & Limitations

### Data Sources

| Data | Source | Notes |
|---|---|---|
| School locations | `POIs_Schools.csv` — collected field data | Real GPS coordinates for ~3,000 Lahore schools |
| Road network & routing | OSRM / OpenStreetMap | Open-source, freely available, no API key required |
| Enrollment model | Trained on EEP survey data | Phase 1, autoencoder, Phase 2 weights in `w_p1/`, `w_ae/`, `w_p2/` |
| Income distributions | Approximate PSLM estimates | **Not from microdata** — see Section 2.5 |

### Known Limitations

1. **Households are synthetic.** No real household addresses are used. The 10,000 agents are randomly generated near schools, not sampled from a census or household survey.

2. **Income quintile bounds are approximate.** The PKR ranges for Q1–Q5 are estimates, not computed from actual PSLM microdata. They should be replaced with real cut-points when microdata is available.

3. **All households are treated as urban.** The model uses urban income statistics and urban probability distributions for all agents, regardless of their exact neighborhood.

4. **School facilities score is randomly assigned.** In the real model, this should reflect actual facilities data for each school. The simulation assigns a random score from a Beta distribution.

5. **OSRM uses driving distance.** In reality, many children walk to school. Walking routes may differ from driving routes (footpaths, shortcuts). The OSRM driving distance is used as a proxy.

6. **The model was not specifically trained on Lahore data.** The inference model was trained on a broader Pakistan urban dataset. Its predictions for Lahore specifically may not capture local patterns precisely.

7. **Route safety is randomly assigned.** The 78% probability of a safe route is a distribution-level assumption, not mapped to actual roads in Lahore.

---

## 11. Technical Reference

### Key Files

| File | Purpose |
|---|---|
| `pages/Lahore_Simulation.py` | Streamlit page — UI, buttons, charts |
| `lahore_simulation.py` | All simulation logic — school loading, household generation, routing, map building, inference preparation |
| `population.py` | `run_batch_inference()` — the three-phase inference pipeline |
| `inference_utils.py` | Model loaders, constants, feature definitions |
| `POIs_Schools.csv` | School coordinates dataset |

### Configurable Parameters (in `lahore_simulation.py`)

| Parameter | Current Value | Description |
|---|---|---|
| `PSLM_QUINTILES` | 5 quintile bands | Income ranges per quintile — update with real PSLM data |
| `_HH_CHUNK` | 100 | Households per OSRM Table API request |
| `max_km` (in `generate_households`) | 2.0 km | Maximum radius from a school for household placement |
| `n` (in `generate_households`) | 10,000 | Number of simulated households |
| Map default zoom | 11 | Initial zoom level (11 = city-wide view of Lahore) |
| Map center | lat=31.5204, lon=74.3587 | Lahore city centre |

### Feature Columns Used by the Model

The enrollment inference model uses exactly these 10 input features:

```
monthly_income    — z-score normalized monthly income
travel_mode       — 0=foot, 1=bicycle, 2=motorcycle, 3=van/rickshaw, 4=public transport
route_safe        — 1=safe, 0=unsafe
read_write        — 1=head can read/write, 0=cannot
solve_math        — 1=head can solve math, 0=cannot
school_facilities — score 0–5 reflecting quality of school
min_distance      — lower bound of distance category (km)
max_distance      — upper bound of distance category (km)
min_time          — lower bound of travel time category (minutes)
max_time          — upper bound of travel time category (minutes)
```

---

## 12. Frequently Asked Questions

**Q: Why does it take 30–60 seconds to fetch routes?**

The OSRM server is a free public service (router.project-osrm.org). For 10,000 households, the system sends approximately 100 HTTP requests, each returning a distance matrix. Network latency to the server (located in Europe) adds roughly 200–500 ms per request. Total: 100 requests × ~400 ms ≈ 40 seconds.

**Q: Why are some road distances blank (—)?**

OSRM cannot route to some school coordinates if the school's GPS point is not on or near the road network. In these cases, the straight-line Haversine distance is retained as a fallback, but road_km and travel_min will show as blank.

**Q: Every time I click Generate, the households move. Is that correct?**

Yes. Each click generates a new random set of 10,000 households. Because placement is random (seeded differently each time), the positions will differ. To reproduce the same layout, a fixed random seed would need to be set in `generate_households()`.

**Q: The map shows households outside Lahore. Why?**

Some schools in `POIs_Schools.csv` may have coordinates near the edges of Lahore district, and households are placed within 2 km of those schools — which may fall just outside the district boundary. This is expected behavior.

**Q: Can I change the number of households?**

Currently 10,000 is hardcoded. 

**Q: What does a probability of exactly 50% mean?**

It means the model is maximally uncertain — it cannot determine whether the household is likely or unlikely to enroll. Households near this threshold are worth investigating further with additional data collection.

**Q: Can I use this for other cities?**

The school data (`POIs_Schools.csv`) currently contains Lahore schools only. For another city, a new POI file would need to be provided, and the PSLM quintile boundaries and income reference statistics would need to be updated to reflect that city's profile.

---

*End of document*
