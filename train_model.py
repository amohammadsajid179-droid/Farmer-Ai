# -*- coding: utf-8 -*-
"""
Farmer AI - Enhanced ML Model Training Script v5.0
Features:
  1. Random Forest Classifier for Crop Recommendation (48 crops)
  2. XGBoost Regressor for Yield Prediction
  3. Advanced Feature Engineering (interaction terms, ratios, polynomial)
  4. Stratified K-Fold Cross-Validation
  5. Detailed accuracy report saved to models/
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
import json
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "crop_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Crop catalogue (ALL 48 crops) ──────────────────────────────────────────────
CROP_INFO = {
    # Cereals
    "rice":           {"icon": "🌾", "season": "Kharif",  "base_yield": 45,   "market_price": 2000},
    "wheat":          {"icon": "🌿", "season": "Rabi",    "base_yield": 35,   "market_price": 2150},
    "maize":          {"icon": "🌽", "season": "Kharif",  "base_yield": 55,   "market_price": 1850},
    "barley":         {"icon": "🌾", "season": "Rabi",    "base_yield": 25,   "market_price": 1800},
    "sorghum":        {"icon": "🌾", "season": "Kharif",  "base_yield": 18,   "market_price": 1700},
    "pearl_millet":   {"icon": "🌾", "season": "Kharif",  "base_yield": 18,   "market_price": 2150},
    "finger_millet":  {"icon": "🌾", "season": "Kharif",  "base_yield": 20,   "market_price": 3200},
    # Pulses
    "chickpea":       {"icon": "🫘", "season": "Rabi",    "base_yield": 18,   "market_price": 5200},
    "lentil":         {"icon": "🫘", "season": "Rabi",    "base_yield": 11,   "market_price": 5400},
    "pigeonpeas":     {"icon": "🫛", "season": "Kharif",  "base_yield": 14,   "market_price": 6300},
    "mungbean":       {"icon": "🫛", "season": "Kharif",  "base_yield": 12,   "market_price": 6000},
    "blackgram":      {"icon": "🌿", "season": "Kharif",  "base_yield": 13,   "market_price": 5700},
    "kidneybeans":    {"icon": "🫘", "season": "Kharif",  "base_yield": 15,   "market_price": 5800},
    "mothbeans":      {"icon": "🌱", "season": "Kharif",  "base_yield": 10,   "market_price": 5500},
    "cowpea":         {"icon": "🫛", "season": "Kharif",  "base_yield": 14,   "market_price": 5000},
    "peas":           {"icon": "🫛", "season": "Rabi",    "base_yield": 12,   "market_price": 4500},
    # Oilseeds
    "mustard":        {"icon": "🌼", "season": "Rabi",    "base_yield": 18,   "market_price": 5650},
    "groundnut":      {"icon": "🥜", "season": "Kharif",  "base_yield": 28,   "market_price": 5850},
    "soybean":        {"icon": "🫘", "season": "Kharif",  "base_yield": 22,   "market_price": 3900},
    "sunflower":      {"icon": "🌻", "season": "Kharif",  "base_yield": 18,   "market_price": 5800},
    "sesame":         {"icon": "🌱", "season": "Kharif",  "base_yield": 8,    "market_price": 6800},
    "linseed":        {"icon": "🌱", "season": "Rabi",    "base_yield": 8,    "market_price": 6000},
    "castor":         {"icon": "🌱", "season": "Kharif",  "base_yield": 20,   "market_price": 5900},
    # Cash Crops
    "sugarcane":      {"icon": "🎋", "season": "Kharif",  "base_yield": 700,  "market_price": 315},
    "cotton":         {"icon": "☁️",  "season": "Kharif",  "base_yield": 20,   "market_price": 6600},
    "jute":           {"icon": "🌿", "season": "Kharif",  "base_yield": 25,   "market_price": 4200},
    "tobacco":        {"icon": "🍂", "season": "Rabi",    "base_yield": 18,   "market_price": 12000},
    # Fruits
    "banana":         {"icon": "🍌", "season": "Kharif",  "base_yield": 400,  "market_price": 2500},
    "mango":          {"icon": "🥭", "season": "Zaid",    "base_yield": 80,   "market_price": 4500},
    "grapes":         {"icon": "🍇", "season": "Rabi",    "base_yield": 200,  "market_price": 5500},
    "watermelon":     {"icon": "🍉", "season": "Zaid",    "base_yield": 300,  "market_price": 1800},
    "apple":          {"icon": "🍎", "season": "Rabi",    "base_yield": 120,  "market_price": 8500},
    "orange":         {"icon": "🍊", "season": "Zaid",    "base_yield": 150,  "market_price": 4000},
    "papaya":         {"icon": "🍈", "season": "Zaid",    "base_yield": 350,  "market_price": 2800},
    "coconut":        {"icon": "🥥", "season": "Kharif",  "base_yield": 100,  "market_price": 3200},
    "pomegranate":    {"icon": "🫐", "season": "Zaid",    "base_yield": 100,  "market_price": 8000},
    "guava":          {"icon": "🍈", "season": "Zaid",    "base_yield": 200,  "market_price": 3000},
    # Vegetables
    "potato":         {"icon": "🥔", "season": "Rabi",    "base_yield": 250,  "market_price": 1200},
    "onion":          {"icon": "🧅", "season": "Rabi",    "base_yield": 200,  "market_price": 1500},
    "tomato":         {"icon": "🍅", "season": "Kharif",  "base_yield": 280,  "market_price": 1200},
    "garlic":         {"icon": "🧄", "season": "Rabi",    "base_yield": 100,  "market_price": 5000},
    # Spices
    "ginger":         {"icon": "🫚", "season": "Kharif",  "base_yield": 150,  "market_price": 7000},
    "turmeric":       {"icon": "🟡", "season": "Kharif",  "base_yield": 180,  "market_price": 8000},
    "pepper":         {"icon": "🌶️",  "season": "Kharif",  "base_yield": 20,   "market_price": 45000},
    "cardamom":       {"icon": "🫛", "season": "Kharif",  "base_yield": 5,    "market_price": 120000},
    # Plantation
    "coffee":         {"icon": "☕", "season": "Kharif",  "base_yield": 22,   "market_price": 8000},
    "tea":            {"icon": "🍵", "season": "Kharif",  "base_yield": 18,   "market_price": 15000},
    "rubber":         {"icon": "🌳", "season": "Kharif",  "base_yield": 150,  "market_price": 18000},
}

CROP_TIPS = {
    "rice":           "Ensure standing water 5-10cm during seedling stage. Apply split doses of Nitrogen.",
    "wheat":          "Irrigate at CRI (21 days), tillering, jointing, and grain-filling stages.",
    "maize":          "Plant at 20cm spacing. Use 30kg/ha Zinc Sulphate for micronutrient boost.",
    "barley":         "Tolerates saline soils. Irrigate at CRI and boot stages.",
    "sorghum":        "Drought tolerant. Intercrop with pigeonpea for soil health.",
    "pearl_millet":   "Best suited for arid regions. Apply 40-60 kg N/ha in splits.",
    "finger_millet":  "Excellent nutritional value. Transplanting gives 20% higher yield.",
    "chickpea":       "No irrigation needed if rainfall is over 500mm. Avoid waterlogging.",
    "kidneybeans":    "Provide trellis support. Harvest pods when fully dry.",
    "pigeonpeas":     "Deep-rooted, drought tolerant. Good for intercropping with short-duration crops.",
    "mothbeans":      "Extremely drought tolerant. Grows well in sandy soils with low water.",
    "mungbean":       "Short duration (60-75 days). Excellent for crop rotation.",
    "blackgram":      "Sensitive to waterlogging. Ensure good field drainage.",
    "lentil":         "Inoculate seeds with Rhizobium before sowing for nitrogen fixation.",
    "cowpea":         "Dual purpose (grain + fodder). Fix nitrogen 80-100 kg/ha.",
    "peas":           "Sow in well-drained loamy soil. Apply Rhizobium inoculant.",
    "mustard":        "Irrigate at rosette and pod-filling stage. Aphid spray critical.",
    "groundnut":      "Apply gypsum at flowering for pod development. Requires well-drained soil.",
    "soybean":        "Inoculate with Bradyrhizobium. Avoid waterlogging at any stage.",
    "sunflower":      "Photo-insensitive crop. Apply boron for seed setting.",
    "sesame":         "Short duration. Sensitive to waterlogging. Harvest at capsule yellowing.",
    "linseed":        "Tolerates cool temperatures. Apply 40-60 kg N + 20-30 kg P/ha.",
    "castor":         "Deep-rooted. Tolerant to drought. Semi-arid zones ideal.",
    "sugarcane":      "Ratoon cropping for 2-3 seasons. Requires 150-200cm annual rainfall.",
    "cotton":         "Use drip irrigation. Monitor regularly for bollworm infestation.",
    "jute":           "Requires humid climate. Retting in water needed for fiber extraction.",
    "tobacco":        "Cured leaves fetch premium. Avoid continuous cropping in same field.",
    "banana":         "Apply 200g N, 60g P2O5, 300g K2O per plant annually.",
    "mango":          "Requires dry weather at flowering. Avoid irrigation during this period.",
    "grapes":         "Train on trellis. Prune annually for quality fruit production.",
    "watermelon":     "Use black plastic mulch to retain moisture and suppress weeds.",
    "apple":          "Requires 1000+ chilling hours below 7C. Hill areas are ideal.",
    "orange":         "Apply Zinc and Boron micronutrients for better fruit quality.",
    "papaya":         "Fast-growing (6-9 months to first harvest). Protect from frost.",
    "coconut":        "Plant at 7.5m x 7.5m spacing. Apply 1.3kg Urea per palm per year.",
    "pomegranate":    "Drought tolerant. Bacterial blight is the main threat.",
    "guava":          "Produces year-round in tropical conditions. Prune for shape.",
    "potato":         "Earthing up at 30 and 45 days critical. Irrigate every 10-12 days.",
    "onion":          "Transplant seedlings at 6 weeks. Stop irrigation 10 days before harvest.",
    "tomato":         "Staking improves yield 30%. Monitor for late blight in humid weather.",
    "garlic":         "Plant cloves 5cm deep. Requires cold period for bulb formation.",
    "ginger":         "Plant rhizome pieces with 2-3 buds. Mulch heavily.",
    "turmeric":       "9-month crop. Apply FYM 25 t/ha before planting.",
    "pepper":         "Grow on standards/trellises. Needs shade and high humidity.",
    "cardamom":       "Shade-loving. Grown under forest canopy. Premium spice.",
    "coffee":         "Shade-grown coffee has superior flavor. Requires acidic soil (pH 5-6).",
    "tea":            "Prune at 3-year cycles. Plucking two leaves and a bud is ideal.",
    "rubber":         "Tapping starts at 6-7 years. Requires consistent rainfall.",
}


def add_feature_engineering(df):
    """Create interaction & ratio features for better model accuracy."""
    df = df.copy()
    df["NPK_total"]    = df["N"] + df["P"] + df["K"]
    df["NP_ratio"]     = df["N"] / (df["P"] + 1)
    df["NK_ratio"]     = df["N"] / (df["K"] + 1)
    df["PK_ratio"]     = df["P"] / (df["K"] + 1)
    df["temp_hum"]     = df["temperature"] * df["humidity"] / 100
    df["rain_hum"]     = df["rainfall"] * df["humidity"] / 100
    df["ph_deviation"] = abs(df["ph"] - 6.5)  # distance from ideal pH
    return df


def generate_yield(row):
    """Estimate yield (quintals/ha) using a domain-based formula."""
    b = CROP_INFO.get(row["crop"], {"base_yield": 30})["base_yield"]
    noise  = np.random.normal(0, b * 0.08)
    factor = 1.0
    if 5.5 <= row["ph"] <= 7.5:           factor += 0.06
    if row["humidity"] > 65:              factor += 0.04
    if row["N"] > 60:                     factor += 0.03
    if 20 <= row["temperature"] <= 30:    factor += 0.03
    if row["rainfall"] > 100:             factor += 0.02
    return round(max(b * factor + noise, b * 0.4), 2)


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 65)
    print("  🌾 FARMER AI - Enhanced Model Training v5.0")
    print("=" * 65)

    # ── Step 1: Check & generate dataset ──────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\n[!] Dataset not found at {DATA_PATH}")
        print("    Run generate_dataset.py first to create the dataset.")
        return

    print("\n[1/5] Loading dataset from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["crop"] = df["crop"].str.strip().str.lower()
    df.dropna(inplace=True)

    if "season"    not in df.columns: df["season"]    = "Kharif"
    if "soil_type" not in df.columns: df["soil_type"] = "Loamy"
    df["season"]    = df["season"].str.strip()
    df["soil_type"] = df["soil_type"].str.strip()

    print(f"     Rows: {len(df)}, Crops: {df['crop'].nunique()}")
    print(f"     Crops: {sorted(df['crop'].unique())}")
    print(f"     Samples per crop:")
    for crop, count in df['crop'].value_counts().sort_index().items():
        print(f"       {crop:20s}: {count}")

    # ── Step 2: Encoders & Feature Engineering ────────────────────────────────
    print("\n[2/5] Encoding & engineering features...")

    le = LabelEncoder()
    le.fit(df["crop"])

    season_enc = OrdinalEncoder(categories=[["Kharif", "Rabi", "Zaid"]])
    soil_enc   = OrdinalEncoder(categories=[["Sandy", "Loamy", "Clayey"]])

    df["season_enc"] = season_enc.fit_transform(df[["season"]])
    df["soil_enc"]   = soil_enc.fit_transform(df[["soil_type"]])

    df = add_feature_engineering(df)

    base_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall",
                     "season_enc", "soil_enc"]
    eng_features  = ["NPK_total", "NP_ratio", "NK_ratio", "PK_ratio",
                     "temp_hum", "rain_hum", "ph_deviation"]
    features = base_features + eng_features
    print(f"     Features ({len(features)}): {features}")

    X = df[features].values
    y = le.transform(df["crop"])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Step 3: Crop Recommendation - Random Forest ───────────────────────────
    print("\n[3/5] Training Random Forest Classifier...")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,       # let trees grow fully
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=le.classes_, output_dict=True)
    report_text = classification_report(y_te, y_pred, target_names=le.classes_)

    print(f"\n     ✅ RANDOM FOREST ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(report_text)

    # Cross-validation
    print("     Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        rf, X_scaled, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=-1
    )
    print(f"     CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"     CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importances
    feat_imp = dict(zip(features, rf.feature_importances_))
    top_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n     Top-5 features: {[(f, round(v,3)) for f, v in top_feats]}")

    # Save crop model
    crop_model = {
        "model": rf,
        "scaler": scaler,
        "label_encoder": le,
        "features": features,
        "season_encoder": season_enc,
        "soil_encoder": soil_enc,
        "crop_info": CROP_INFO,
        "crop_tips": CROP_TIPS,
        "accuracy": round(acc, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "engineered_features": eng_features,
        "feature_importances": feat_imp,
    }
    joblib.dump(crop_model, os.path.join(MODELS_DIR, "crop_model.pkl"))
    print("     [SAVED] crop_model.pkl")

    # ── Step 4: Yield Prediction (XGBoost) ────────────────────────────────────
    print("\n[4/5] Training Yield Prediction (XGBoost)...")
    np.random.seed(42)
    df["yield_qty"]    = df.apply(generate_yield, axis=1)
    df["crop_encoded"] = le.transform(df["crop"])

    yield_features = features + ["crop_encoded"]
    X_y = df[yield_features].values
    y_y = df["yield_qty"].values

    yield_scaler = StandardScaler()
    X_y_scaled   = yield_scaler.fit_transform(X_y[:, :len(features)])
    X_y_full     = np.column_stack([X_y_scaled, X_y[:, -1]])

    Xt, Xv, yt, yv = train_test_split(X_y_full, y_y, test_size=0.20, random_state=42)
    xgb_reg = XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    xgb_reg.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    y_pred_yield = xgb_reg.predict(Xv)
    r2  = r2_score(yv, y_pred_yield)
    mae = mean_absolute_error(yv, y_pred_yield)
    print(f"     ✅ R² Score: {r2:.4f}")
    print(f"     MAE: {mae:.2f} quintals/ha")

    yield_model = {
        "model": xgb_reg,
        "scaler": yield_scaler,
        "label_encoder": le,
        "features": yield_features,
        "season_encoder": season_enc,
        "soil_encoder": soil_enc,
        "r2_score": round(r2, 4),
        "engineered_features": eng_features,
    }
    joblib.dump(yield_model, os.path.join(MODELS_DIR, "yield_model.pkl"))
    print("     [SAVED] yield_model.pkl")

    # ── Step 5: Soil metadata ──────────────────────────────────────────────────
    soil_meta = {
        "mean_N": float(df["N"].mean()), "mean_P": float(df["P"].mean()),
        "mean_K": float(df["K"].mean()), "mean_ph": float(df["ph"].mean()),
    }
    joblib.dump(soil_meta, os.path.join(MODELS_DIR, "soil_meta.pkl"))
    print("     [SAVED] soil_meta.pkl")

    # ── Save accuracy report ──────────────────────────────────────────────────
    print("\n[5/5] Saving accuracy report...")
    accuracy_report = {
        "crop_model": {
            "type": "Random Forest Classifier",
            "test_accuracy": round(acc * 100, 2),
            "cv_mean_accuracy": round(cv_scores.mean() * 100, 2),
            "cv_std": round(cv_scores.std() * 100, 2),
            "num_features": len(features),
            "feature_names": features,
            "num_crops": len(le.classes_),
            "crops": list(le.classes_),
            "per_crop": {
                k: {
                    "precision": round(v["precision"] * 100, 1),
                    "recall":    round(v["recall"] * 100, 1),
                    "f1":        round(v["f1-score"] * 100, 1)
                }
                for k, v in report.items() if k in le.classes_
            },
        },
        "yield_model": {
            "type": "XGBoost Regressor",
            "r2_score": round(r2 * 100, 2),
            "mae_quintals": round(mae, 2),
        },
        "dataset": {
            "total_rows": len(df),
            "num_crops": df["crop"].nunique(),
        }
    }
    with open(os.path.join(MODELS_DIR, "accuracy_report.json"), "w") as f:
        json.dump(accuracy_report, f, indent=2)
    print("     [SAVED] accuracy_report.json")

    print("\n" + "=" * 65)
    print(f"  ✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"  📊 Crop Accuracy: {acc*100:.2f}%  |  Yield R²: {r2*100:.1f}%")
    print(f"  📂 Models saved in: {MODELS_DIR}")
    print(f"  🌾 Supported crops: {list(le.classes_)}")
    print("=" * 65)


if __name__ == "__main__":
    train()
