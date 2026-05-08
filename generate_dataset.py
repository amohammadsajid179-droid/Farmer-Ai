# -*- coding: utf-8 -*-
"""
Farmer AI - Dataset Generator v5.0 (India-Wide Coverage)
Generates a large, realistic, well-distributed crop dataset covering:
  - All major Indian crop types: Cereals, Pulses, Oilseeds, Cash Crops, Horticulture, Spices
  - All agro-climatic zones: Arid, Semi-Arid, Humid, Sub-Humid, Hill, Coastal, NE
  - Data sourced from ICAR, NRAA, and Indian Crop Calendar standards
"""

import numpy as np
import pandas as pd
import os

np.random.seed(2024)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_PATH  = os.path.join(BASE_DIR, "data", "crop_data.csv")

# ── Crop parameter distributions (REALISTIC Indian agronomic ranges) ───────────
# Each entry uses:
#   N=(mean, std), P=(mean, std), K=(mean, std)   -> kg/ha
#   temp=(mean, std)                               -> Celsius
#   hum=(mean, std)                                -> Relative Humidity %
#   ph=(mean, std)                                 -> Soil pH
#   rain=(mean, std)                               -> mm/year (clipped 0-500)
#   season, soils, n (number of samples)
# ─────────────────────────────────────────────────────────────────────────────

CROP_PARAMS = {
    # ── CEREALS ──────────────────────────────────────────────────────────────
    "rice": dict(
        N=(85, 18), P=(45, 12), K=(42, 10),
        temp=(25, 3), hum=(82, 5), ph=(6.4, 0.5), rain=(240, 55),
        season="Kharif", soils=["Clayey", "Loamy"], n=200
    ),
    "wheat": dict(
        N=(82, 15), P=(50, 12), K=(44, 8),
        temp=(21, 3), hum=(78, 7), ph=(6.9, 0.5), rain=(175, 40),
        season="Rabi", soils=["Loamy", "Sandy"], n=200
    ),
    "maize": dict(
        N=(90, 15), P=(55, 10), K=(40, 8),
        temp=(27, 3), hum=(65, 10), ph=(6.2, 0.5), rain=(80, 25),
        season="Kharif", soils=["Sandy", "Loamy"], n=150
    ),
    "barley": dict(
        N=(60, 12), P=(30, 8), K=(30, 6),
        temp=(15, 4), hum=(55, 8), ph=(6.5, 0.5), rain=(80, 25),
        season="Rabi", soils=["Sandy", "Loamy"], n=100
    ),
    "sorghum": dict(   # Jowar — major in Maharashtra, Karnataka, MP
        N=(70, 12), P=(35, 8), K=(40, 8),
        temp=(30, 3), hum=(55, 10), ph=(6.5, 0.5), rain=(70, 20),
        season="Kharif", soils=["Sandy", "Loamy", "Clayey"], n=120
    ),
    "pearl_millet": dict(   # Bajra — Rajasthan, Gujarat, Haryana
        N=(55, 10), P=(25, 7), K=(30, 6),
        temp=(33, 3), hum=(45, 10), ph=(7.0, 0.5), rain=(40, 15),
        season="Kharif", soils=["Sandy"], n=120
    ),
    "finger_millet": dict(  # Ragi — Karnataka, TN, AP
        N=(50, 10), P=(30, 7), K=(30, 6),
        temp=(26, 3), hum=(70, 8), ph=(6.0, 0.5), rain=(100, 25),
        season="Kharif", soils=["Sandy", "Loamy"], n=100
    ),

    # ── PULSES ───────────────────────────────────────────────────────────────
    "chickpea": dict(
        N=(20, 5), P=(55, 8), K=(40, 6),
        temp=(23, 4), hum=(50, 10), ph=(7.2, 0.5), rain=(55, 15),
        season="Rabi", soils=["Sandy", "Loamy"], n=150
    ),
    "lentil": dict(
        N=(12, 4), P=(18, 5), K=(18, 4),
        temp=(22, 3), hum=(58, 6), ph=(6.2, 0.6), rain=(38, 12),
        season="Rabi", soils=["Sandy", "Loamy"], n=120
    ),
    "pigeonpeas": dict(
        N=(14, 4), P=(50, 8), K=(50, 8),
        temp=(28, 2), hum=(64, 6), ph=(5.9, 0.5), rain=(150, 20),
        season="Kharif", soils=["Sandy", "Loamy"], n=120
    ),
    "mungbean": dict(
        N=(22, 4), P=(50, 7), K=(22, 4),
        temp=(30, 2), hum=(55, 7), ph=(6.5, 0.4), rain=(60, 15),
        season="Kharif", soils=["Sandy", "Loamy"], n=120
    ),
    "blackgram": dict(
        N=(22, 4), P=(45, 7), K=(22, 4),
        temp=(29, 2), hum=(62, 6), ph=(6.8, 0.4), rain=(50, 12),
        season="Kharif", soils=["Loamy"], n=100
    ),
    "kidneybeans": dict(
        N=(22, 4), P=(80, 10), K=(60, 8),
        temp=(20, 3), hum=(70, 6), ph=(6.2, 0.5), rain=(110, 15),
        season="Kharif", soils=["Loamy"], n=100
    ),
    "mothbeans": dict(
        N=(22, 4), P=(60, 8), K=(22, 4),
        temp=(30, 3), hum=(50, 8), ph=(7.0, 0.5), rain=(50, 12),
        season="Kharif", soils=["Sandy"], n=100
    ),
    "cowpea": dict(     # Lobia — Tamil Nadu, Andhra, UP
        N=(25, 5), P=(55, 8), K=(45, 7),
        temp=(30, 3), hum=(68, 7), ph=(6.5, 0.5), rain=(80, 20),
        season="Kharif", soils=["Sandy", "Loamy"], n=100
    ),
    "peas": dict(       # Matar — UP, Punjab, HP
        N=(25, 5), P=(50, 8), K=(50, 7),
        temp=(16, 3), hum=(72, 6), ph=(6.5, 0.4), rain=(60, 15),
        season="Rabi", soils=["Loamy", "Sandy"], n=100
    ),

    # ── OILSEEDS ─────────────────────────────────────────────────────────────
    "mustard": dict(    # Main Rabi oilseed — UP, Rajasthan, MP, Haryana
        N=(80, 12), P=(40, 8), K=(40, 7),
        temp=(18, 4), hum=(60, 8), ph=(7.0, 0.5), rain=(55, 15),
        season="Rabi", soils=["Sandy", "Loamy"], n=150
    ),
    "groundnut": dict(  # Peanut — Gujarat, AP, Rajasthan, TN
        N=(25, 5), P=(55, 8), K=(80, 10),
        temp=(28, 3), hum=(60, 8), ph=(6.5, 0.5), rain=(80, 20),
        season="Kharif", soils=["Sandy", "Loamy"], n=150
    ),
    "soybean": dict(    # MP, Maharashtra, Rajasthan
        N=(25, 5), P=(65, 10), K=(50, 8),
        temp=(28, 3), hum=(65, 7), ph=(6.5, 0.4), rain=(100, 20),
        season="Kharif", soils=["Loamy", "Clayey"], n=150
    ),
    "sunflower": dict(  # Karnataka, AP, Maharashtra
        N=(90, 12), P=(55, 8), K=(60, 8),
        temp=(28, 3), hum=(60, 8), ph=(6.5, 0.4), rain=(60, 15),
        season="Kharif", soils=["Loamy", "Sandy"], n=100
    ),
    "sesame": dict(     # Til — Gujarat, Rajasthan, WB, MP
        N=(40, 8), P=(25, 6), K=(25, 5),
        temp=(30, 3), hum=(50, 8), ph=(6.5, 0.4), rain=(50, 15),
        season="Kharif", soils=["Sandy", "Loamy"], n=100
    ),
    "linseed": dict(    # Alsi — MP, UP, Bihar, Rajasthan
        N=(40, 8), P=(30, 6), K=(30, 6),
        temp=(18, 4), hum=(60, 8), ph=(6.5, 0.4), rain=(40, 12),
        season="Rabi", soils=["Loamy", "Clayey"], n=80
    ),
    "castor": dict(     # Erandi — Gujarat, Rajasthan, AP
        N=(60, 10), P=(40, 7), K=(40, 7),
        temp=(30, 3), hum=(55, 8), ph=(6.5, 0.4), rain=(60, 18),
        season="Kharif", soils=["Sandy", "Loamy"], n=80
    ),

    # ── CASH CROPS ───────────────────────────────────────────────────────────
    "sugarcane": dict(
        N=(110, 15), P=(55, 8), K=(120, 15),
        temp=(28, 2), hum=(80, 5), ph=(6.5, 0.4), rain=(200, 35),
        season="Kharif", soils=["Loamy", "Clayey"], n=150
    ),
    "cotton": dict(
        N=(100, 15), P=(45, 8), K=(45, 8),
        temp=(30, 3), hum=(65, 8), ph=(7.0, 0.4), rain=(80, 20),
        season="Kharif", soils=["Loamy", "Sandy", "Clayey"], n=150
    ),
    "jute": dict(
        N=(80, 10), P=(40, 6), K=(40, 6),
        temp=(28, 2), hum=(82, 5), ph=(7.0, 0.4), rain=(120, 20),
        season="Kharif", soils=["Loamy", "Clayey"], n=100
    ),
    "tobacco": dict(    # AP, Karnataka, Gujarat, UP
        N=(80, 12), P=(50, 8), K=(100, 12),
        temp=(28, 3), hum=(65, 7), ph=(6.0, 0.5), rain=(75, 18),
        season="Rabi", soils=["Sandy", "Loamy"], n=80
    ),

    # ── FRUITS ───────────────────────────────────────────────────────────────
    "banana": dict(
        N=(180, 20), P=(60, 8), K=(200, 20),
        temp=(27, 3), hum=(80, 5), ph=(6.5, 0.4), rain=(200, 40),
        season="Kharif", soils=["Loamy"], n=150
    ),
    "mango": dict(
        N=(100, 15), P=(60, 8), K=(80, 10),
        temp=(30, 3), hum=(60, 8), ph=(6.5, 0.5), rain=(100, 30),
        season="Zaid", soils=["Sandy", "Loamy"], n=120
    ),
    "grapes": dict(
        N=(50, 8), P=(30, 6), K=(60, 8),
        temp=(27, 3), hum=(55, 8), ph=(6.5, 0.4), rain=(55, 15),
        season="Rabi", soils=["Sandy"], n=100
    ),
    "watermelon": dict(
        N=(80, 12), P=(40, 7), K=(60, 8),
        temp=(33, 2), hum=(65, 7), ph=(6.5, 0.4), rain=(55, 15),
        season="Zaid", soils=["Sandy"], n=100
    ),
    "apple": dict(
        N=(60, 10), P=(40, 7), K=(60, 8),
        temp=(15, 3), hum=(65, 7), ph=(6.2, 0.4), rain=(120, 25),
        season="Rabi", soils=["Loamy", "Sandy"], n=100
    ),
    "orange": dict(
        N=(80, 12), P=(40, 7), K=(80, 10),
        temp=(25, 3), hum=(65, 7), ph=(6.5, 0.4), rain=(120, 25),
        season="Zaid", soils=["Loamy", "Sandy"], n=100
    ),
    "papaya": dict(
        N=(200, 25), P=(200, 25), K=(200, 25),
        temp=(28, 3), hum=(65, 7), ph=(6.5, 0.4), rain=(120, 25),
        season="Zaid", soils=["Sandy", "Loamy"], n=100
    ),
    "coconut": dict(
        N=(100, 15), P=(40, 7), K=(200, 20),
        temp=(27, 2), hum=(80, 5), ph=(6.0, 0.4), rain=(200, 35),
        season="Kharif", soils=["Sandy", "Loamy"], n=100
    ),
    "pomegranate": dict(  # Maharashtra, Karnataka, AP, Rajasthan
        N=(100, 15), P=(50, 8), K=(80, 10),
        temp=(30, 4), hum=(50, 10), ph=(6.5, 0.5), rain=(50, 15),
        season="Zaid", soils=["Sandy", "Loamy"], n=80
    ),
    "guava": dict(      # UP, Bihar, WB, MP
        N=(100, 15), P=(60, 8), K=(80, 10),
        temp=(28, 4), hum=(65, 8), ph=(6.5, 0.5), rain=(100, 25),
        season="Zaid", soils=["Sandy", "Loamy"], n=80
    ),

    # ── VEGETABLES ───────────────────────────────────────────────────────────
    "potato": dict(     # UP, WB, Punjab, Bihar, Gujarat
        N=(120, 15), P=(80, 10), K=(120, 12),
        temp=(18, 3), hum=(75, 7), ph=(6.0, 0.5), rain=(75, 18),
        season="Rabi", soils=["Loamy", "Sandy"], n=150
    ),
    "onion": dict(      # Maharashtra, Karnataka, MP, Gujarat, Rajasthan
        N=(80, 12), P=(50, 8), K=(80, 10),
        temp=(25, 4), hum=(65, 8), ph=(6.5, 0.4), rain=(70, 18),
        season="Rabi", soils=["Loamy", "Sandy"], n=120
    ),
    "tomato": dict(     # AP, Karnataka, WB, MP
        N=(100, 15), P=(80, 10), K=(80, 10),
        temp=(24, 3), hum=(70, 8), ph=(6.5, 0.4), rain=(60, 15),
        season="Kharif", soils=["Loamy", "Sandy"], n=120
    ),
    "garlic": dict(     # MP, Rajasthan, Gujarat, UP
        N=(60, 10), P=(40, 7), K=(60, 8),
        temp=(20, 4), hum=(60, 8), ph=(6.0, 0.5), rain=(50, 15),
        season="Rabi", soils=["Sandy", "Loamy"], n=100
    ),

    # ── SPICES ───────────────────────────────────────────────────────────────
    "ginger": dict(     # Kerala, Meghalaya, Karnataka, AP, NE India
        N=(100, 15), P=(50, 8), K=(150, 15),
        temp=(26, 2), hum=(80, 5), ph=(5.5, 0.4), rain=(200, 40),
        season="Kharif", soils=["Loamy"], n=100
    ),
    "turmeric": dict(   # AP, TS, Karnataka, Tamil Nadu, Odisha
        N=(120, 15), P=(60, 8), K=(120, 12),
        temp=(28, 2), hum=(75, 6), ph=(6.0, 0.4), rain=(150, 30),
        season="Kharif", soils=["Loamy", "Sandy"], n=100
    ),
    "pepper": dict(     # Kerala, Karnataka — Black Pepper
        N=(100, 15), P=(60, 8), K=(100, 12),
        temp=(28, 2), hum=(85, 4), ph=(5.5, 0.4), rain=(250, 50),
        season="Kharif", soils=["Loamy"], n=80
    ),
    "cardamom": dict(   # Kerala, Karnataka, Tamil Nadu
        N=(60, 10), P=(30, 6), K=(80, 10),
        temp=(22, 2), hum=(85, 4), ph=(5.5, 0.4), rain=(350, 60),
        season="Kharif", soils=["Loamy"], n=80
    ),

    # ── PLANTATION / PERENNIAL ────────────────────────────────────────────────
    "coffee": dict(     # Karnataka, Kerala, TN
        N=(100, 15), P=(50, 8), K=(100, 12),
        temp=(23, 2), hum=(80, 5), ph=(5.5, 0.4), rain=(200, 40),
        season="Kharif", soils=["Loamy"], n=100
    ),
    "tea": dict(        # Assam, WB Darjeeling, Kerala, Tamil Nadu
        N=(120, 15), P=(40, 7), K=(40, 7),
        temp=(20, 3), hum=(85, 4), ph=(5.0, 0.4), rain=(300, 60),
        season="Kharif", soils=["Loamy"], n=80
    ),
    "rubber": dict(     # Kerala, NE India
        N=(100, 15), P=(40, 7), K=(60, 8),
        temp=(27, 2), hum=(85, 4), ph=(5.0, 0.4), rain=(300, 60),
        season="Kharif", soils=["Loamy"], n=80
    ),

    # ── REMAINING ORIGINAL CROPS ──────────────────────────────────────────────
    "lentil_orig": dict(   # aliased to lentil for backward compat
        N=(12, 4), P=(16, 5), K=(16, 4),
        temp=(25, 3), hum=(59, 5), ph=(6.0, 0.7), rain=(38, 10),
        season="Rabi", soils=["Sandy", "Loamy"], n=100
    ),
}

# Normalize duplicate — use a cleaner approach
CROP_PARAMS.pop("lentil_orig", None)

CROP_INFO = {
    "rice":           {"base_yield": 45,  "market_price": 2000},
    "wheat":          {"base_yield": 35,  "market_price": 2150},
    "maize":          {"base_yield": 55,  "market_price": 1850},
    "barley":         {"base_yield": 25,  "market_price": 1800},
    "sorghum":        {"base_yield": 18,  "market_price": 1700},
    "pearl_millet":   {"base_yield": 18,  "market_price": 2150},
    "finger_millet":  {"base_yield": 20,  "market_price": 3200},
    "chickpea":       {"base_yield": 18,  "market_price": 5200},
    "lentil":         {"base_yield": 11,  "market_price": 5400},
    "pigeonpeas":     {"base_yield": 14,  "market_price": 6300},
    "mungbean":       {"base_yield": 12,  "market_price": 6000},
    "blackgram":      {"base_yield": 13,  "market_price": 5700},
    "kidneybeans":    {"base_yield": 15,  "market_price": 5800},
    "mothbeans":      {"base_yield": 10,  "market_price": 5500},
    "cowpea":         {"base_yield": 14,  "market_price": 5000},
    "peas":           {"base_yield": 12,  "market_price": 4500},
    "mustard":        {"base_yield": 18,  "market_price": 5650},
    "groundnut":      {"base_yield": 28,  "market_price": 5850},
    "soybean":        {"base_yield": 22,  "market_price": 3900},
    "sunflower":      {"base_yield": 18,  "market_price": 5800},
    "sesame":         {"base_yield": 8,   "market_price": 6800},
    "linseed":        {"base_yield": 8,   "market_price": 6000},
    "castor":         {"base_yield": 20,  "market_price": 5900},
    "sugarcane":      {"base_yield": 700, "market_price": 315},
    "cotton":         {"base_yield": 20,  "market_price": 6600},
    "jute":           {"base_yield": 25,  "market_price": 4200},
    "tobacco":        {"base_yield": 18,  "market_price": 12000},
    "banana":         {"base_yield": 400, "market_price": 2500},
    "mango":          {"base_yield": 80,  "market_price": 4500},
    "grapes":         {"base_yield": 200, "market_price": 5500},
    "watermelon":     {"base_yield": 300, "market_price": 1800},
    "apple":          {"base_yield": 120, "market_price": 8500},
    "orange":         {"base_yield": 150, "market_price": 4000},
    "papaya":         {"base_yield": 350, "market_price": 2800},
    "coconut":        {"base_yield": 100, "market_price": 3200},
    "pomegranate":    {"base_yield": 100, "market_price": 8000},
    "guava":          {"base_yield": 200, "market_price": 3000},
    "potato":         {"base_yield": 250, "market_price": 1200},
    "onion":          {"base_yield": 200, "market_price": 1500},
    "tomato":         {"base_yield": 280, "market_price": 1200},
    "garlic":         {"base_yield": 100, "market_price": 5000},
    "ginger":         {"base_yield": 150, "market_price": 7000},
    "turmeric":       {"base_yield": 180, "market_price": 8000},
    "pepper":         {"base_yield": 20,  "market_price": 45000},
    "cardamom":       {"base_yield": 5,   "market_price": 120000},
    "coffee":         {"base_yield": 22,  "market_price": 8000},
    "tea":            {"base_yield": 18,  "market_price": 15000},
    "rubber":         {"base_yield": 150, "market_price": 18000},
}


def generate_rows(crop, params):
    n       = params["n"]
    soils   = params["soils"]
    season  = params["season"]
    rows    = []

    def sample(key):
        mu, sigma = params[key]
        return np.random.normal(mu, sigma, n)

    N    = np.clip(sample("N"),    0, 200)
    P    = np.clip(sample("P"),    0, 200)
    K    = np.clip(sample("K"),    0, 300)
    temp = np.clip(sample("temp"), -10, 60)
    hum  = np.clip(sample("hum"),  0, 100)
    ph   = np.clip(sample("ph"),   0, 14)
    rain = np.clip(sample("rain"), 0, 500)

    soil_choices = np.random.choice(soils, n)

    for i in range(n):
        rows.append({
            "N":           round(float(N[i]),    1),
            "P":           round(float(P[i]),    1),
            "K":           round(float(K[i]),    1),
            "temperature": round(float(temp[i]), 1),
            "humidity":    round(float(hum[i]),  1),
            "ph":          round(float(ph[i]),   2),
            "rainfall":    round(float(rain[i]), 1),
            "season":      season,
            "soil_type":   soil_choices[i],
            "crop":        crop,
        })
    return rows


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    all_rows = []
    total    = 0
    print("\n" + "=" * 60)
    print("  Farmer AI Dataset Generator v5.0 — India-Wide Coverage")
    print("=" * 60)
    for crop, params in CROP_PARAMS.items():
        rows = generate_rows(crop, params)
        all_rows.extend(rows)
        total += len(rows)
        print(f"  {crop:20s}: {len(rows):4d} samples  |  Season: {params['season']}")

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=2024).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] Dataset generated: {total} rows, {df['crop'].nunique()} unique crops")
    print(f"     Saved to: {OUT_PATH}")
    print(f"\nCrop distribution:")
    print(df["crop"].value_counts().to_string())
    print("\n[DONE] Now run: python train_model.py")


if __name__ == "__main__":
    main()
