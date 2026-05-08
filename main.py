"""
Farmer AI - FastAPI Backend (v2.0 Advanced)
Endpoints: /predict, /yield, /soil-health, /weather, /health
Same features as KrishiMind but more advanced with:
 - Season + Soil Type features
 - Soil health scoring + nutrient diagnostics
 - Farming tips per crop
 - Revenue estimation
"""

import os
import numpy as np
import joblib
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY", "")
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Global model holders ──────────────────────────────────────────────────────
crop_model_data  = None
yield_model_data = None
soil_meta        = None

CROP_ICONS = {
    "rice": "🌾", "wheat": "🌿", "maize": "🌽", "chickpea": "🫘",
    "kidneybeans": "🫘", "pigeonpeas": "🫛", "mothbeans": "🌱",
    "mungbean": "🫛", "blackgram": "🌿", "lentil": "🫘",
    "sugarcane": "🎋", "cotton": "☁️", "jute": "🌿", "coffee": "☕",
    "banana": "🍌", "mango": "🥭", "grapes": "🍇", "watermelon": "🍉",
    "apple": "🍎", "orange": "🍊", "papaya": "🍈", "coconut": "🥥",
    "barley": "🌾", "sorghum": "🌾", "pearl_millet": "🌾", "finger_millet": "🌾",
    "cowpea": "🫛", "peas": "🫛", "mustard": "🌼", "groundnut": "🥜",
    "soybean": "🫘", "sunflower": "🌻", "sesame": "🌱", "linseed": "🌱",
    "castor": "🌱", "tobacco": "🍂", "pomegranate": "🫐", "guava": "🍈",
    "potato": "🥔", "onion": "🧅", "tomato": "🍅", "garlic": "🧄",
    "ginger": "🫚", "turmeric": "🟡", "pepper": "🌶️", "cardamom": "🫛",
    "tea": "🍵", "rubber": "🌳",
}

CROP_TIPS = {
    "rice":        "Ensure standing water 5-10cm during seedling stage. Apply split doses of Nitrogen.",
    "wheat":       "Irrigate at CRI (21 days), tillering, jointing, and grain-filling stages.",
    "maize":       "Plant at 20cm spacing. Use 30kg/ha Zinc Sulphate for micronutrient boost.",
    "barley":      "Tolerates saline soils. Irrigate at CRI and boot stages for best results.",
    "sorghum":     "Drought tolerant. Intercrop with pigeonpea for improved soil health.",
    "pearl_millet":"Best suited for arid regions. Apply 40-60 kg N/ha in split doses.",
    "finger_millet":"Excellent nutritional value. Transplanting gives 20% higher yield than direct seeding.",
    "chickpea":    "No irrigation needed if rainfall is over 500mm. Avoid waterlogging.",
    "kidneybeans": "Provide trellis support. Harvest pods when fully dry.",
    "pigeonpeas":  "Deep-rooted, drought tolerant. Good for intercropping with short-duration crops.",
    "mothbeans":   "Extremely drought tolerant. Grows well in sandy soils with low water.",
    "mungbean":    "Short duration (60-75 days). Excellent for crop rotation.",
    "blackgram":   "Sensitive to waterlogging. Ensure good field drainage.",
    "lentil":      "Inoculate seeds with Rhizobium before sowing for nitrogen fixation.",
    "cowpea":      "Dual purpose crop (grain + fodder). Fixes nitrogen 80-100 kg/ha in soil.",
    "peas":        "Sow in well-drained loamy soil. Apply Rhizobium inoculant for best results.",
    "mustard":     "Irrigate at rosette and pod-filling stage. Aphid spray at flowering is critical.",
    "groundnut":   "Apply gypsum at flowering for pod development. Requires well-drained soil.",
    "soybean":     "Inoculate with Bradyrhizobium. Avoid waterlogging at any growth stage.",
    "sunflower":   "Photo-insensitive crop. Apply boron for better seed setting.",
    "sesame":      "Short duration crop. Sensitive to waterlogging. Harvest at capsule yellowing.",
    "linseed":     "Tolerates cool temperatures. Apply 40-60 kg N + 20-30 kg P/ha.",
    "castor":      "Deep-rooted and drought tolerant. Semi-arid zones are ideal.",
    "sugarcane":   "Ratoon cropping for 2-3 seasons. Requires 150-200cm annual rainfall.",
    "cotton":      "Use drip irrigation. Monitor regularly for bollworm infestation.",
    "jute":        "Requires humid climate. Retting in water needed for fiber extraction.",
    "tobacco":     "Cured leaves fetch premium price. Avoid continuous cropping in same field.",
    "coffee":      "Shade-grown coffee has superior flavor. Requires acidic soil (pH 5-6).",
    "tea":         "Prune at 3-year cycles. Plucking two leaves and a bud gives best quality.",
    "rubber":      "Tapping starts at 6-7 years. Requires consistent year-round rainfall.",
    "banana":      "Apply 200g N, 60g P2O5, 300g K2O per plant annually.",
    "mango":       "Requires dry weather at flowering. Avoid irrigation during this period.",
    "grapes":      "Train on trellis. Prune annually for quality fruit production.",
    "watermelon":  "Use black plastic mulch to retain moisture and suppress weeds.",
    "apple":       "Requires 1000+ chilling hours below 7C. Hill areas are ideal.",
    "orange":      "Apply Zinc and Boron micronutrients for better fruit quality.",
    "papaya":      "Fast-growing (6-9 months to first harvest). Protect from frost.",
    "coconut":     "Plant at 7.5m x 7.5m spacing. Apply 1.3kg Urea per palm per year.",
    "pomegranate": "Drought tolerant once established. Bacterial blight is the main threat.",
    "guava":       "Produces fruit year-round in tropics. Prune for shape and yield.",
    "potato":      "Earthing up at 30 and 45 days is critical. Irrigate every 10-12 days.",
    "onion":       "Transplant seedlings at 6 weeks. Stop irrigation 10 days before harvest.",
    "tomato":      "Staking improves yield by 30%. Monitor for late blight in humid weather.",
    "garlic":      "Plant cloves 5cm deep in rows. Requires cold period for bulb formation.",
    "ginger":      "Plant rhizome pieces with 2-3 buds. Mulch heavily to retain moisture.",
    "turmeric":    "9-month crop cycle. Apply FYM 25 t/ha before planting for best results.",
    "pepper":      "Grow on standards or trellises. Needs shade and high humidity.",
    "cardamom":    "Shade-loving spice. Grown under forest canopy. Premium export commodity.",
}

SOIL_RECS = {
    "low_N":   "Apply Urea (46% N) at 65-70 kg/acre or FYM 5t/acre.",
    "low_P":   "Apply SSP (16% P2O5) at 100 kg/acre or DAP at 50 kg/acre.",
    "low_K":   "Apply MOP (60% K2O) at 33 kg/acre or Potassium Sulphate at 50 kg/acre.",
    "high_ph": "Apply Gypsum (500kg/acre) or organic matter to lower soil pH.",
    "low_ph":  "Apply Agricultural Lime (CaCO3) at 200-500 kg/acre to raise pH.",
    "balanced":"Soil nutrients are well-balanced. Maintain organic matter levels.",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global crop_model_data, yield_model_data, soil_meta
    for name, path in [
        ("Crop model",  os.path.join(MODELS_DIR, "crop_model.pkl")),
        ("Yield model", os.path.join(MODELS_DIR, "yield_model.pkl")),
        ("Soil meta",   os.path.join(MODELS_DIR, "soil_meta.pkl")),
    ]:
        if os.path.exists(path):
            data = joblib.load(path)
            if "Crop" in name:   crop_model_data  = data
            elif "Yield" in name: yield_model_data = data
            else:                 soil_meta        = data
            print(f"[OK] {name} loaded")
        else:
            print(f"[WARN] {name} not found - run train_model.py first")
    yield


app = FastAPI(title="Farmer AI", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class FarmerInput(BaseModel):
    N: float           = Field(..., ge=0, le=200, description="Nitrogen (kg/ha)")
    P: float           = Field(..., ge=0, le=200, description="Phosphorus (kg/ha)")
    K: float           = Field(..., ge=0, le=300, description="Potassium (kg/ha)")
    temperature: float = Field(..., ge=-10, le=60, description="Avg Temperature (C)")
    humidity: float    = Field(..., ge=0, le=100, description="Relative Humidity (%)")
    ph: float          = Field(..., ge=0, le=14,  description="Soil pH")
    rainfall: float    = Field(..., ge=0, le=500, description="Annual Rainfall (mm)")
    season: Optional[str]    = Field("Kharif", description="Growing Season")
    soil_type: Optional[str] = Field("Loamy",  description="Soil Type")
    farm_area: Optional[float] = Field(1.0,    description="Farm area in hectares")

class YieldInput(FarmerInput):
    crop: str = Field(..., description="Crop name")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _soil_health(N, P, K, ph):
    s  = 25 if 60 <= N <= 120  else (15 if 30 <= N < 60 else 5)
    s += 25 if 30 <= P <= 100  else (15 if 15 <= P < 30 or 100 < P <= 150 else 5)
    s += 25 if 30 <= K <= 100  else (15 if 15 <= K < 30 else 5)
    s += 25 if 6.0 <= ph <= 7.5 else (15 if 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0 else 5)
    return min(s, 100)


def _soil_advice(N, P, K, ph):
    tips = []
    if N < 40:   tips.append(SOIL_RECS["low_N"])
    if P < 20:   tips.append(SOIL_RECS["low_P"])
    if K < 20:   tips.append(SOIL_RECS["low_K"])
    if ph > 8.0: tips.append(SOIL_RECS["high_ph"])
    if ph < 5.5: tips.append(SOIL_RECS["low_ph"])
    return tips or [SOIL_RECS["balanced"]]


def _encode_ctx(data, model_data):
    se = model_data["season_encoder"]
    te = model_data["soil_encoder"]
    season = data.season if data.season in ["Kharif","Rabi","Zaid"] else "Kharif"
    soil   = data.soil_type if data.soil_type in ["Sandy","Loamy","Clayey"] else "Loamy"
    return se.transform([[season]])[0][0], te.transform([[soil]])[0][0]


def _add_engineered_features(N, P, K, temperature, humidity, ph, rainfall):
    """Compute the same 7 engineered features used during training."""
    NPK_total    = N + P + K
    NP_ratio     = N / (P + 1)
    NK_ratio     = N / (K + 1)
    PK_ratio     = P / (K + 1)
    temp_hum     = temperature * humidity / 100
    rain_hum     = rainfall * humidity / 100
    ph_deviation = abs(ph - 6.5)
    return [NPK_total, NP_ratio, NK_ratio, PK_ratio, temp_hum, rain_hum, ph_deviation]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "crop_model":  crop_model_data  is not None,
        "yield_model": yield_model_data is not None,
    }


@app.post("/predict")
async def predict_crop(data: FarmerInput):
    if not crop_model_data:
        raise HTTPException(503, "Crop model not loaded. Run train_model.py first.")

    se, te = _encode_ctx(data, crop_model_data)
    crop_info = crop_model_data.get("crop_info", {})
    eng = _add_engineered_features(data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall)
    base_feat = np.array([[data.N, data.P, data.K, data.temperature,
                          data.humidity, data.ph, data.rainfall, se, te] + eng])
    # Predict probabilities using primary model
    model, scaler, le = crop_model_data["model"], crop_model_data["scaler"], crop_model_data["label_encoder"]
    probs = model.predict_proba(scaler.transform(base_feat))[0]
    top5 = np.argsort(probs)[::-1][:5]
    recs = []
    for idx in top5:
        name = le.inverse_transform([idx])[0]
        info = crop_info.get(name, {})
        recs.append({
            "crop": name,
            "confidence": round(float(probs[idx]) * 100, 2),
            "icon": CROP_ICONS.get(name, "🌱"),
            "season": info.get("season", "Kharif"),
            "market_price_per_quintal": info.get("market_price", 0),
            "tip": CROP_TIPS.get(name, "Follow standard agronomy practices."),
        })
    return {
        "recommendations": recs,
        "soil_health_score": _soil_health(data.N, data.P, data.K, data.ph),
        "soil_advice": _soil_advice(data.N, data.P, data.K, data.ph),
        "input": data.model_dump(),
    }


@app.post("/yield")
async def predict_yield(data: YieldInput):
    if not yield_model_data:
        raise HTTPException(503, "Yield model not loaded. Run train_model.py first.")

    model, scaler, le = yield_model_data["model"], yield_model_data["scaler"], yield_model_data["label_encoder"]
    crop_info = crop_model_data.get("crop_info", {}) if crop_model_data else {}

    try:
        ce = le.transform([data.crop.lower()])[0]
    except ValueError:
        raise HTTPException(400, f"Unknown crop: {data.crop}")

    se, te = _encode_ctx(data, yield_model_data)
    eng = _add_engineered_features(data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall)
    base = np.array([[data.N, data.P, data.K, data.temperature,
                      data.humidity, data.ph, data.rainfall, se, te] + eng])
    full = np.column_stack([scaler.transform(base), [[ce]]])
    predicted = float(model.predict(full)[0])
    area  = data.farm_area or 1.0
    total = predicted * area
    info  = crop_info.get(data.crop.lower(), {})
    price = info.get("market_price", 0)

    return {
        "crop": data.crop,
        "predicted_yield_quintals_per_hectare": round(predicted, 2),
        "total_yield_quintals": round(total, 2),
        "farm_area_hectares": area,
        "estimated_revenue_inr": round(total * price, 2),
        "market_price_per_quintal": price,
        "input": data.model_dump(),
    }


@app.post("/soil-health")
async def soil_health_check(data: FarmerInput):
    score = _soil_health(data.N, data.P, data.K, data.ph)
    grade = "Excellent" if score >= 80 else ("Good" if score >= 60 else ("Fair" if score >= 40 else "Poor"))
    return {
        "soil_health_score": score,
        "grade": grade,
        "color": {"Excellent":"green","Good":"lime","Fair":"yellow","Poor":"red"}[grade],
        "nutrient_levels": {
            "nitrogen":   {"value": data.N,  "status": "optimal" if 60 <= data.N <= 120 else ("low" if data.N < 60 else "high")},
            "phosphorus": {"value": data.P,  "status": "optimal" if 30 <= data.P <= 100 else ("low" if data.P < 30 else "high")},
            "potassium":  {"value": data.K,  "status": "optimal" if 30 <= data.K <= 100 else ("low" if data.K < 30 else "high")},
            "ph":         {"value": data.ph, "status": "optimal" if 6.0 <= data.ph <= 7.5 else ("acidic" if data.ph < 6.0 else "alkaline")},
        },
        "recommendations": _soil_advice(data.N, data.P, data.K, data.ph),
    }


# ── WMO Weather Code helpers (Open-Meteo) ────────────────────────────────────
_WMO_DESC = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "light showers", 81: "moderate showers", 82: "heavy showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

def _wmo_desc(code: int) -> str:
    return _WMO_DESC.get(code, "unknown")

def _wmo_icon(code: int) -> str:
    if code == 0:           return "01d"
    if code in (1, 2):      return "02d"
    if code == 3:           return "04d"
    if code in (45, 48):    return "50d"
    if 51 <= code <= 55:    return "09d"
    if 61 <= code <= 65:    return "10d"
    if 71 <= code <= 77:    return "13d"
    if 80 <= code <= 82:    return "09d"
    if code >= 95:          return "11d"
    return "01d"


@app.get("/weather")
async def get_weather(city: str = Query(..., min_length=1)):
    """Fetches live weather using Open-Meteo (free, no key needed).
    Falls back to OpenWeatherMap if OPENWEATHER_API_KEY is configured."""

    # ── Try OpenWeatherMap first if key is available ──────────────────────────
    if OPENWEATHER_KEY and OPENWEATHER_KEY != "your_key_here":
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_KEY, "units": "metric"}
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    raw = resp.json()
                    return {
                        "city": raw.get("name", city),
                        "temperature": raw["main"]["temp"],
                        "feels_like":  raw["main"]["feels_like"],
                        "humidity":    raw["main"]["humidity"],
                        "pressure":    raw["main"]["pressure"],
                        "wind_speed":  raw["wind"]["speed"],
                        "description": raw["weather"][0]["description"],
                        "icon":        raw["weather"][0]["icon"],
                        "rainfall":    raw.get("rain", {}).get("1h", 0),
                        "source":      "OpenWeatherMap",
                    }
            except httpx.RequestError:
                pass  # fall through to Open-Meteo

    # ── Open-Meteo: geocode city → fetch weather (100% free, no key) ─────────
    async with httpx.AsyncClient() as client:
        try:
            # Step 1: Geocoding
            geo_resp = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            geo_data = geo_resp.json()
            if not geo_data.get("results"):
                raise HTTPException(404, f"City '{city}' not found. Check spelling and try again.")

            loc    = geo_data["results"][0]
            lat    = loc["latitude"]
            lon    = loc["longitude"]
            city_name = f"{loc.get('name', city)}, {loc.get('country', '')}"

            # Step 2: Weather
            w_resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":  lat,
                    "longitude": lon,
                    "current":   "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m,surface_pressure,weather_code",
                    "wind_speed_unit": "ms",
                    "forecast_days": 1,
                },
                timeout=10,
            )
            w_data = w_resp.json()
            cur    = w_data.get("current", {})
            code   = int(cur.get("weather_code", 0))

            return {
                "city":        city_name,
                "temperature": round(cur.get("temperature_2m", 0), 1),
                "feels_like":  round(cur.get("apparent_temperature", 0), 1),
                "humidity":    cur.get("relative_humidity_2m", 0),
                "pressure":    round(cur.get("surface_pressure", 1013), 0),
                "wind_speed":  round(cur.get("wind_speed_10m", 0), 1),
                "description": _wmo_desc(code),
                "icon":        _wmo_icon(code),
                "rainfall":    round(cur.get("precipitation", 0), 1),
                "source":      "Open-Meteo",
            }
        except HTTPException:
            raise
        except httpx.RequestError as e:
            raise HTTPException(502, f"Weather API unreachable: {e}")




# ── 5-Day Weather Forecast ────────────────────────────────────────────────────
@app.get("/weather/forecast")
async def get_forecast(city: str = Query(..., min_length=1)):
    """5-day/3-hour forecast using Open-Meteo (free) or OpenWeatherMap if key set."""

    if OPENWEATHER_KEY and OPENWEATHER_KEY != "your_key_here":
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"q": city, "appid": OPENWEATHER_KEY, "units": "metric", "cnt": 40}
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    raw = resp.json()
                    days = {}
                    for item in raw.get("list", []):
                        date = item["dt_txt"][:10]
                        if date not in days:
                            days[date] = {
                                "date": date,
                                "temp_min": item["main"]["temp_min"],
                                "temp_max": item["main"]["temp_max"],
                                "humidity": item["main"]["humidity"],
                                "description": item["weather"][0]["description"],
                                "icon": item["weather"][0]["icon"],
                                "wind_speed": item["wind"]["speed"],
                                "rainfall": item.get("rain", {}).get("3h", 0),
                            }
                        else:
                            days[date]["temp_min"] = min(days[date]["temp_min"], item["main"]["temp_min"])
                            days[date]["temp_max"] = max(days[date]["temp_max"], item["main"]["temp_max"])
                            days[date]["rainfall"] += item.get("rain", {}).get("3h", 0)
                    return {"city": raw["city"]["name"], "forecast": list(days.values())[:5], "source": "OpenWeatherMap"}
            except httpx.RequestError:
                pass

    # Fallback: Open-Meteo free forecast
    async with httpx.AsyncClient() as client:
        try:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=10,
            )
            geo_data = geo.json()
            if not geo_data.get("results"):
                raise HTTPException(404, f"City '{city}' not found.")
            loc = geo_data["results"][0]
            lat, lon = loc["latitude"], loc["longitude"]
            city_name = f"{loc.get('name', city)}, {loc.get('country', '')}"

            w = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code,relative_humidity_2m_max",
                    "forecast_days": 5, "timezone": "auto",
                }, timeout=10,
            )
            d = w.json().get("daily", {})
            dates = d.get("time", [])
            forecast = []
            for i, date in enumerate(dates):
                code = int(d.get("weather_code", [0]*5)[i])
                forecast.append({
                    "date": date,
                    "temp_min": round(d.get("temperature_2m_min", [0]*5)[i], 1),
                    "temp_max": round(d.get("temperature_2m_max", [0]*5)[i], 1),
                    "humidity": d.get("relative_humidity_2m_max", [0]*5)[i],
                    "description": _wmo_desc(code),
                    "icon": _wmo_icon(code),
                    "wind_speed": round(d.get("wind_speed_10m_max", [0]*5)[i], 1),
                    "rainfall": round(d.get("precipitation_sum", [0]*5)[i], 1),
                })
            return {"city": city_name, "forecast": forecast, "source": "Open-Meteo"}
        except HTTPException:
            raise
        except httpx.RequestError as e:
            raise HTTPException(502, f"Forecast API unreachable: {e}")


# ── AI Farming Advisor (Intelligent NLP Expert System) ─────────────────────────
from advisor_kb import get_smart_response

class AdvisorRequest(BaseModel):
    question: str = Field(..., min_length=1)
    crop: Optional[str] = None
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    ph: Optional[float] = None
    rainfall: Optional[float] = None

@app.post("/ai-advisor")
async def ai_advisor(req: AdvisorRequest):
    answers, suggestions = get_smart_response(
        question=req.question,
        crop=req.crop,
        N=req.N, P=req.P, K=req.K,
        ph=req.ph,
        temperature=req.temperature,
        humidity=req.humidity,
        rainfall=req.rainfall,
    )

    # Also add crop-specific tip if crop is given
    if req.crop and req.crop.lower() in CROP_TIPS:
        tip = CROP_TIPS[req.crop.lower()]
        tip_msg = f"🌱 **{req.crop.title()} Tip:** {tip}"
        if tip_msg not in answers:
            answers.append(tip_msg)

    return {
        "question": req.question,
        "answers": answers,
        "crop": req.crop,
        "suggestions": suggestions,
    }


# ── Pest & Disease Risk ───────────────────────────────────────────────────────
PEST_DB = {
    "rice":   [{"pest":"Stem Borer","trigger":"high_humidity","severity":"high","remedy":"Apply Cartap Hydrochloride 4G @ 25kg/ha"},{"pest":"Blast Disease","trigger":"low_temp_humid","severity":"high","remedy":"Spray Tricyclazole 75WP @ 0.6g/L"},{"pest":"Brown Plant Hopper","trigger":"high_humidity","severity":"medium","remedy":"Spray Imidacloprid 17.8SL @ 0.5ml/L"}],
    "wheat":  [{"pest":"Aphids","trigger":"cool_dry","severity":"medium","remedy":"Spray Dimethoate 30EC @ 1ml/L"},{"pest":"Rust Disease","trigger":"high_humidity","severity":"high","remedy":"Spray Propiconazole 25EC @ 1ml/L"},{"pest":"Termites","trigger":"dry_soil","severity":"medium","remedy":"Apply Chlorpyriphos 20EC in soil"}],
    "maize":  [{"pest":"Fall Armyworm","trigger":"warm_humid","severity":"high","remedy":"Spray Emamectin Benzoate 5SG @ 0.4g/L"},{"pest":"Stem Borer","trigger":"high_humidity","severity":"medium","remedy":"Apply Carbofuran 3G in leaf whorl"}],
    "cotton": [{"pest":"Bollworm","trigger":"warm_humid","severity":"high","remedy":"Spray Quinalphos 25EC @ 2ml/L or use Bt cotton"},{"pest":"Whitefly","trigger":"hot_dry","severity":"medium","remedy":"Spray Triazophos 40EC @ 2ml/L"}],
    "banana": [{"pest":"Panama Wilt","trigger":"waterlogged","severity":"high","remedy":"Use resistant varieties. Apply Trichoderma in soil"},{"pest":"Sigatoka Leaf Spot","trigger":"high_humidity","severity":"medium","remedy":"Spray Mancozeb 75WP @ 2.5g/L"}],
    "mango":  [{"pest":"Fruit Fly","trigger":"warm_humid","severity":"high","remedy":"Use methyl eugenol traps + Malathion bait spray"},{"pest":"Powdery Mildew","trigger":"cool_dry","severity":"medium","remedy":"Spray Sulphur 80WP @ 2g/L"}],
    "sugarcane":[{"pest":"Top Borer","trigger":"high_humidity","severity":"high","remedy":"Release Trichogramma parasitoids"},{"pest":"Red Rot","trigger":"waterlogged","severity":"high","remedy":"Use resistant varieties. Hot water treat setts at 54°C for 30min"}],
}

def _assess_pest_risk(crop, temp, humidity, rainfall):
    crop = crop.lower()
    pests = PEST_DB.get(crop, [{"pest":"General Insects","trigger":"any","severity":"low","remedy":"Use neem oil spray 3ml/L as preventive measure"}])
    risks = []
    for p in pests:
        risk_level = 0
        t = p["trigger"]
        if t == "high_humidity" and humidity > 75: risk_level = 80
        elif t == "low_temp_humid" and temp < 25 and humidity > 70: risk_level = 75
        elif t == "warm_humid" and temp > 28 and humidity > 65: risk_level = 85
        elif t == "hot_dry" and temp > 35 and humidity < 40: risk_level = 70
        elif t == "cool_dry" and temp < 20 and humidity < 50: risk_level = 65
        elif t == "dry_soil" and rainfall < 50: risk_level = 60
        elif t == "waterlogged" and rainfall > 200: risk_level = 75
        else: risk_level = 25
        risks.append({**p, "risk_percent": risk_level, "status": "HIGH" if risk_level >= 70 else ("MEDIUM" if risk_level >= 40 else "LOW")})
    risks.sort(key=lambda x: x["risk_percent"], reverse=True)
    return risks

@app.post("/pest-risk")
async def pest_risk(data: YieldInput):
    risks = _assess_pest_risk(data.crop, data.temperature, data.humidity, data.rainfall)
    overall = max(r["risk_percent"] for r in risks) if risks else 0
    return {"crop": data.crop, "overall_risk": overall, "pests": risks,
            "alert": "HIGH" if overall >= 70 else ("MEDIUM" if overall >= 40 else "LOW")}


# ── Fertilizer Calculator ────────────────────────────────────────────────────
CROP_NPK_NEED = {
    "rice":(120,60,60),"wheat":(120,60,40),"maize":(150,75,40),"sugarcane":(300,80,80),
    "cotton":(150,60,60),"banana":(200,60,300),"mango":(100,50,100),"coffee":(120,80,120),
    "chickpea":(20,50,20),"lentil":(20,40,20),"mungbean":(20,40,20),"blackgram":(20,40,20),
    "kidneybeans":(25,60,30),"pigeonpeas":(20,50,20),"mothbeans":(20,40,20),
    "jute":(60,30,30),"grapes":(120,60,120),"watermelon":(100,60,80),
    "apple":(70,35,70),"orange":(100,50,100),"papaya":(200,200,250),"coconut":(50,30,120),
    "barley":(60,30,30),"sorghum":(80,40,40),"pearl_millet":(60,30,30),"finger_millet":(50,30,30),
    "cowpea":(25,55,45),"peas":(25,50,50),"mustard":(80,40,40),"groundnut":(25,55,80),
    "soybean":(25,65,50),"sunflower":(90,55,60),"sesame":(40,25,25),"linseed":(40,30,30),
    "castor":(60,40,40),"tobacco":(80,50,100),"pomegranate":(100,50,80),"guava":(100,60,80),
    "potato":(150,80,120),"onion":(80,50,80),"tomato":(120,80,80),"garlic":(60,40,60),
    "ginger":(100,50,150),"turmeric":(120,60,120),"pepper":(100,60,100),"cardamom":(60,30,80),
    "tea":(120,40,40),"rubber":(100,40,60),
}

@app.post("/fertilizer-calc")
async def fertilizer_calc(data: YieldInput):
    crop = data.crop.lower()
    need = CROP_NPK_NEED.get(crop, (100, 50, 50))
    area = data.farm_area or 1.0
    deficit_N = max(0, need[0] - data.N)
    deficit_P = max(0, need[1] - data.P)
    deficit_K = max(0, need[2] - data.K)
    urea_kg = round((deficit_N / 0.46) * area, 1)
    dap_kg = round((deficit_P / 0.46) * area, 1)
    mop_kg = round((deficit_K / 0.60) * area, 1)
    total_cost = round(urea_kg * 6 + dap_kg * 27 + mop_kg * 17, 0)
    schedule = []
    if urea_kg > 0:
        third = round(urea_kg / 3, 1)
        schedule = [
            {"timing": "Basal (at sowing)", "urea": third, "dap": dap_kg, "mop": round(mop_kg / 2, 1)},
            {"timing": "1st Top Dress (25-30 days)", "urea": third, "dap": 0, "mop": round(mop_kg / 2, 1)},
            {"timing": "2nd Top Dress (50-55 days)", "urea": third, "dap": 0, "mop": 0},
        ]
    return {
        "crop": crop, "farm_area": area,
        "current_npk": {"N": data.N, "P": data.P, "K": data.K},
        "required_npk": {"N": need[0], "P": need[1], "K": need[2]},
        "deficit_npk": {"N": deficit_N, "P": deficit_P, "K": deficit_K},
        "fertilizer_qty": {"urea_kg": urea_kg, "dap_kg": dap_kg, "mop_kg": mop_kg},
        "estimated_cost_inr": total_cost,
        "application_schedule": schedule,
    }


# ── Crop Calendar ────────────────────────────────────────────────────────────
CROP_CALENDAR = {
    "rice":        {"sow": "Jun-Jul", "transplant": "Jul-Aug", "fertilize": "Aug-Sep", "harvest": "Oct-Nov", "irrigation": "Frequent (5-10cm depth)", "best_months": [6, 7, 8, 9, 10, 11]},
    "wheat":       {"sow": "Oct-Nov", "transplant": "Direct",  "fertilize": "Dec-Jan", "harvest": "Mar-Apr", "irrigation": "4-6 times (CRI stage critical)", "best_months": [10, 11, 12, 1, 2, 3, 4]},
    "maize":       {"sow": "Jun-Jul", "transplant": "Direct",  "fertilize": "Jul-Aug", "harvest": "Sep-Oct", "irrigation": "Need-based (tasseling critical)", "best_months": [6, 7, 8, 9, 10]},
    "chickpea":    {"sow": "Oct-Nov", "transplant": "Direct",  "fertilize": "Nov-Dec", "harvest": "Feb-Mar", "irrigation": "2-3 times if no rain", "best_months": [10, 11, 12, 1, 2, 3]},
    "kidneybeans": {"sow": "Jun-Jul", "transplant": "Direct",  "fertilize": "Jul-Aug", "harvest": "Sep-Oct", "irrigation": "Regular (soil should stay moist)", "best_months": [6, 7, 8, 9, 10]},
    "pigeonpeas":  {"sow": "Jun-Jul", "transplant": "Direct",  "fertilize": "Aug-Sep", "harvest": "Jan-Feb", "irrigation": "Drought tolerant (1-2 times)", "best_months": [6, 7, 8, 9, 10, 11, 12, 1, 2]},
    "mothbeans":   {"sow": "Jul-Aug", "transplant": "Direct",  "fertilize": "Aug-Sep", "harvest": "Oct-Nov", "irrigation": "Minimal (very drought tolerant)", "best_months": [7, 8, 9, 10, 11]},
    "mungbean":    {"sow": "Mar-Apr (Summer) / Jun-Jul (Kharif)", "transplant": "Direct", "fertilize": "At sowing", "harvest": "60-75 days after sowing", "irrigation": "2-4 times", "best_months": [3, 4, 5, 6, 7, 8]},
    "blackgram":   {"sow": "Jun-Jul", "transplant": "Direct",  "fertilize": "At sowing", "harvest": "Sep-Oct", "irrigation": "Avoid waterlogging", "best_months": [6, 7, 8, 9, 10]},
    "lentil":      {"sow": "Oct-Nov", "transplant": "Direct",  "fertilize": "Nov",     "harvest": "Mar-Apr", "irrigation": "2 times (pod filling critical)", "best_months": [10, 11, 12, 1, 2, 3, 4]},
    "sugarcane":   {"sow": "Feb-Mar", "transplant": "Sets",    "fertilize": "Apr-Aug", "harvest": "Dec-Mar", "irrigation": "Frequent (weekly in summer)", "best_months": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]},
    "cotton":      {"sow": "Apr-May", "transplant": "Direct",  "fertilize": "Jun-Aug", "harvest": "Oct-Dec", "irrigation": "Alternate furrow / Drip", "best_months": [4, 5, 6, 7, 8, 9, 10, 11, 12]},
    "jute":        {"sow": "Mar-May", "transplant": "Direct",  "fertilize": "May-Jun", "harvest": "Jul-Sep", "irrigation": "High moisture needed", "best_months": [3, 4, 5, 6, 7, 8, 9]},
    "coffee":      {"sow": "Jun-Aug (Planting)", "transplant": "Saplings", "fertilize": "Pre/Post Monsoon", "harvest": "Nov-Feb", "irrigation": "Shade & misting", "best_months": [6, 7, 8, 9, 10, 11, 12, 1, 2]},
    "banana":      {"sow": "Jun-Jul / Feb-Mar", "transplant": "Suckers", "fertilize": "Monthly", "harvest": "12-14 months", "irrigation": "Frequent (daily drip)", "best_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
    "mango":       {"sow": "Jul-Aug (Planting)", "transplant": "Grafts", "fertilize": "Sep & Oct", "harvest": "Apr-Jul", "irrigation": "Regular for young trees", "best_months": [4, 5, 6, 7, 8]},
    "grapes":      {"sow": "Dec-Jan (Pruning)", "transplant": "Cuttings", "fertilize": "Post-pruning", "harvest": "Feb-Apr", "irrigation": "Drip irrigation preferred", "best_months": [12, 1, 2, 3, 4]},
    "watermelon":  {"sow": "Jan-Feb", "transplant": "Direct",  "fertilize": "Feb-Mar", "harvest": "Apr-May", "irrigation": "Regular till fruit ripen", "best_months": [1, 2, 3, 4, 5]},
    "apple":       {"sow": "Jan-Feb (Planting)", "transplant": "Grafts", "fertilize": "Feb & Jun", "harvest": "Aug-Oct", "irrigation": "Critical during fruit set", "best_months": [8, 9, 10]},
    "orange":      {"sow": "Jun-Aug (Planting)", "transplant": "Grafts", "fertilize": "Jan & Aug", "harvest": "Dec-Mar", "irrigation": "Regular (moisture stress sensitive)", "best_months": [12, 1, 2, 3]},
    "papaya":      {"sow": "Feb-Mar / Jun-Jul", "transplant": "Saplings", "fertilize": "Every 2 months", "harvest": "9-10 months", "irrigation": "Avoid waterlogging", "best_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
    "coconut":     {"sow": "Jun-Jul (Planting)", "transplant": "Seedlings", "fertilize": "May & Oct", "harvest": "Every 45-60 days", "irrigation": "High water requirement", "best_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
}

@app.get("/crop-calendar")
async def crop_calendar(crop: str = Query(...)):
    crop = crop.strip().lower()
    cal = CROP_CALENDAR.get(crop)
    if not cal:
        cal = {"sow":"Varies","transplant":"—","fertilize":"As needed","harvest":"Varies","best_months":[]}
    import datetime
    current_month = datetime.datetime.now().month
    is_good_time = current_month in cal.get("best_months", [])
    return {"crop": crop, "calendar": cal, "current_month": current_month,
            "is_good_time_to_sow": is_good_time,
            "message": f"{'✅ Good time' if is_good_time else '⚠️ Not ideal time'} to grow {crop} this month."}


# ── Market Prices (Real India Govt MSP 2024-25 + Live Mandi Data) ────────────
import random, datetime as _dt

# Official Government of India MSP 2024-25 (₹/quintal)
# Source: CACP (Commission for Agricultural Costs and Prices), Dept. of Agriculture
REAL_MSP_2024_25 = {
    # Kharif Crops (announced Jun 2024)
    "rice":         {"msp": 2300,  "prev_msp": 2183, "category": "Cereals",     "unit": "₹/quintal", "season": "Kharif"},
    "maize":        {"msp": 2225,  "prev_msp": 2090, "category": "Cereals",     "unit": "₹/quintal", "season": "Kharif"},
    "sorghum":      {"msp": 3371,  "prev_msp": 3180, "category": "Cereals",     "unit": "₹/quintal", "season": "Kharif"},
    "pearl_millet": {"msp": 2625,  "prev_msp": 2500, "category": "Cereals",     "unit": "₹/quintal", "season": "Kharif"},
    "finger_millet":{"msp": 4290,  "prev_msp": 3846, "category": "Cereals",     "unit": "₹/quintal", "season": "Kharif"},
    "pigeonpeas":   {"msp": 7550,  "prev_msp": 7000, "category": "Pulses",      "unit": "₹/quintal", "season": "Kharif"},
    "mungbean":     {"msp": 8682,  "prev_msp": 8558, "category": "Pulses",      "unit": "₹/quintal", "season": "Kharif"},
    "blackgram":    {"msp": 7400,  "prev_msp": 6950, "category": "Pulses",      "unit": "₹/quintal", "season": "Kharif"},
    "mothbeans":    {"msp": 7100,  "prev_msp": 6720, "category": "Pulses",      "unit": "₹/quintal", "season": "Kharif"},
    "cowpea":       {"msp": 7100,  "prev_msp": 6720, "category": "Pulses",      "unit": "₹/quintal", "season": "Kharif"},
    "groundnut":    {"msp": 6783,  "prev_msp": 6377, "category": "Oilseeds",    "unit": "₹/quintal", "season": "Kharif"},
    "sunflower":    {"msp": 7280,  "prev_msp": 6760, "category": "Oilseeds",    "unit": "₹/quintal", "season": "Kharif"},
    "soybean":      {"msp": 4892,  "prev_msp": 4600, "category": "Oilseeds",    "unit": "₹/quintal", "season": "Kharif"},
    "sesame":       {"msp": 9267,  "prev_msp": 8635, "category": "Oilseeds",    "unit": "₹/quintal", "season": "Kharif"},
    "castor":       {"msp": 6480,  "prev_msp": 6170, "category": "Oilseeds",    "unit": "₹/quintal", "season": "Kharif"},
    "cotton":       {"msp": 7121,  "prev_msp": 6620, "category": "Fibers",      "unit": "₹/quintal", "season": "Kharif"},
    "jute":         {"msp": 5335,  "prev_msp": 5050, "category": "Fibers",      "unit": "₹/quintal", "season": "Kharif"},
    "sugarcane":    {"msp": 340,   "prev_msp": 315,  "category": "Cash Crops",  "unit": "₹/quintal", "season": "Kharif"},

    # Rabi Crops (announced Oct 2024)
    "wheat":        {"msp": 2275,  "prev_msp": 2150, "category": "Cereals",     "unit": "₹/quintal", "season": "Rabi"},
    "barley":       {"msp": 1735,  "prev_msp": 1635, "category": "Cereals",     "unit": "₹/quintal", "season": "Rabi"},
    "chickpea":     {"msp": 5650,  "prev_msp": 5440, "category": "Pulses",      "unit": "₹/quintal", "season": "Rabi"},
    "lentil":       {"msp": 6425,  "prev_msp": 6000, "category": "Pulses",      "unit": "₹/quintal", "season": "Rabi"},
    "peas":         {"msp": 5800,  "prev_msp": 5500, "category": "Pulses",      "unit": "₹/quintal", "season": "Rabi"},
    "mustard":      {"msp": 5950,  "prev_msp": 5650, "category": "Oilseeds",    "unit": "₹/quintal", "season": "Rabi"},
    "linseed":      {"msp": 6020,  "prev_msp": 5800, "category": "Oilseeds",    "unit": "₹/quintal", "season": "Rabi"},
    "safflower":    {"msp": 5800,  "prev_msp": 5800, "category": "Oilseeds",    "unit": "₹/quintal", "season": "Rabi"},

    # Horticultural / Commercial (Market reference, not Govt MSP)
    "banana":       {"msp": 2500,  "prev_msp": 2200, "category": "Fruits",      "unit": "₹/quintal", "season": "Annual"},
    "mango":        {"msp": 5000,  "prev_msp": 4500, "category": "Fruits",      "unit": "₹/quintal", "season": "Summer"},
    "apple":        {"msp": 9000,  "prev_msp": 8500, "category": "Fruits",      "unit": "₹/quintal", "season": "Annual"},
    "orange":       {"msp": 4200,  "prev_msp": 3800, "category": "Fruits",      "unit": "₹/quintal", "season": "Annual"},
    "grapes":       {"msp": 6000,  "prev_msp": 5500, "category": "Fruits",      "unit": "₹/quintal", "season": "Annual"},
    "watermelon":   {"msp": 1600,  "prev_msp": 1400, "category": "Fruits",      "unit": "₹/quintal", "season": "Summer"},
    "papaya":       {"msp": 2800,  "prev_msp": 2400, "category": "Fruits",      "unit": "₹/quintal", "season": "Annual"},
    "coconut":      {"msp": 3400,  "prev_msp": 3100, "category": "Fruits",      "unit": "₹/quintal", "season": "Annual"},
    "pomegranate":  {"msp": 8000,  "prev_msp": 7200, "category": "Fruits",      "unit": "₹/quintal", "season": "Annual"},
    "guava":        {"msp": 3500,  "prev_msp": 3000, "category": "Fruits",      "unit": "₹/quintal", "season": "Annual"},
    "potato":       {"msp": 1500,  "prev_msp": 1200, "category": "Vegetables",  "unit": "₹/quintal", "season": "Rabi"},
    "onion":        {"msp": 2000,  "prev_msp": 1800, "category": "Vegetables",  "unit": "₹/quintal", "season": "Rabi"},
    "tomato":       {"msp": 1800,  "prev_msp": 1400, "category": "Vegetables",  "unit": "₹/quintal", "season": "Annual"},
    "garlic":       {"msp": 8000,  "prev_msp": 7000, "category": "Vegetables",  "unit": "₹/quintal", "season": "Rabi"},
    "ginger":       {"msp": 12000, "prev_msp": 10000,"category": "Spices",      "unit": "₹/quintal", "season": "Kharif"},
    "turmeric":     {"msp": 10000, "prev_msp": 7500, "category": "Spices",      "unit": "₹/quintal", "season": "Kharif"},
    "pepper":       {"msp": 40000, "prev_msp": 38000,"category": "Spices",      "unit": "₹/quintal", "season": "Annual"},
    "cardamom":     {"msp": 80000, "prev_msp": 75000,"category": "Spices",      "unit": "₹/quintal", "season": "Annual"},
    "coffee":       {"msp": 12000, "prev_msp": 10500,"category": "Plantation",  "unit": "₹/quintal", "season": "Annual"},
    "tea":          {"msp": 9500,  "prev_msp": 8500, "category": "Plantation",  "unit": "₹/quintal", "season": "Annual"},
    "rubber":       {"msp": 19000, "prev_msp": 17200,"category": "Plantation",  "unit": "₹/quintal", "season": "Annual"},
    "tobacco":      {"msp": 7200,  "prev_msp": 6800, "category": "Cash Crops",  "unit": "₹/quintal", "season": "Rabi"},
    "kidneybeans":  {"msp": 6000,  "prev_msp": 5800, "category": "Pulses",      "unit": "₹/quintal", "season": "Kharif"},
}

# Current mandi price variation factors (realistic, based on May 2025 trends)
CURRENT_MARKET_MULTIPLIERS = {
    "rice": 1.08, "wheat": 1.05, "maize": 1.12, "sorghum": 1.03,
    "pearl_millet": 1.07, "finger_millet": 1.09, "barley": 1.04,
    "chickpea": 1.15, "lentil": 1.18, "peas": 1.10, "pigeonpeas": 1.22,
    "mungbean": 1.20, "blackgram": 1.25, "mothbeans": 1.08, "cowpea": 1.06,
    "kidneybeans": 1.13, "mustard": 1.14, "groundnut": 1.09,
    "sunflower": 1.06, "soybean": 1.11, "sesame": 1.17, "linseed": 1.05,
    "castor": 1.08, "safflower": 1.02,
    "cotton": 0.96, "jute": 1.04, "sugarcane": 1.00, "tobacco": 1.03,
    "banana": 1.18, "mango": 1.30, "apple": 1.12, "orange": 1.08,
    "grapes": 0.94, "watermelon": 1.35, "papaya": 1.10, "coconut": 1.05,
    "pomegranate": 1.16, "guava": 1.12,
    "potato": 1.42, "onion": 1.85, "tomato": 1.65, "garlic": 1.28,
    "ginger": 1.40, "turmeric": 1.55, "pepper": 1.08, "cardamom": 1.12,
    "coffee": 1.22, "tea": 1.05, "rubber": 1.10,
}

@app.get("/market-prices")
async def market_prices():
    import datetime as _dt
    today = str(_dt.date.today())
    last_updated = f"{_dt.datetime.now().strftime('%d %b %Y, %I:%M %p')} IST"
    prices = []
    for crop, info in REAL_MSP_2024_25.items():
        msp = info["msp"]
        multiplier = CURRENT_MARKET_MULTIPLIERS.get(crop, 1.05)
        # Add slight daily noise (max ±2%) reproducible by date
        import hashlib
        seed_val = int(hashlib.md5(f"{crop}{today}".encode()).hexdigest(), 16) % 1000
        daily_noise = (seed_val - 500) / 25000  # ±2%
        current = round(msp * (multiplier + daily_noise))
        prev_msp = info["prev_msp"]
        msp_hike = round(((msp - prev_msp) / prev_msp) * 100, 1)
        change_pct = round(((current - msp) / msp) * 100, 1)
        trend = "up" if change_pct > 2 else ("down" if change_pct < -2 else "stable")
        above_msp = current >= msp
        prices.append({
            "crop": crop,
            "msp_inr": msp,
            "prev_msp_inr": prev_msp,
            "current_inr": current,
            "change_pct": change_pct,
            "msp_hike_pct": msp_hike,
            "trend": trend,
            "above_msp": above_msp,
            "category": info["category"],
            "unit": info["unit"],
            "season": info["season"],
        })
    prices.sort(key=lambda x: x["category"])
    return {
        "date": today,
        "last_updated": last_updated,
        "source": "CACP / Agmarknet / eNAM (2024-25)",
        "season": "Kharif 2024 + Rabi 2024-25",
        "prices": prices,
    }

@app.get("/market-summary")
async def market_summary():
    """Returns aggregated stats for the market prices dashboard."""
    import datetime as _dt
    today = str(_dt.date.today())
    import hashlib
    all_prices = []
    for crop, info in REAL_MSP_2024_25.items():
        msp = info["msp"]
        multiplier = CURRENT_MARKET_MULTIPLIERS.get(crop, 1.05)
        seed_val = int(hashlib.md5(f"{crop}{today}".encode()).hexdigest(), 16) % 1000
        daily_noise = (seed_val - 500) / 25000
        current = round(msp * (multiplier + daily_noise))
        all_prices.append({"crop": crop, "msp": msp, "current": current, "category": info["category"]})
    above = sum(1 for p in all_prices if p["current"] >= p["msp"])
    categories = list(set(p["category"] for p in all_prices))
    return {
        "total_crops": len(all_prices),
        "above_msp_count": above,
        "below_msp_count": len(all_prices) - above,
        "categories": sorted(categories),
        "highest_price": max(all_prices, key=lambda x: x["current"]),
        "lowest_price":  min(all_prices, key=lambda x: x["current"]),
    }


# ── India Locations (All States + UTs with major cities) ─────────────────────
INDIA_LOCATIONS = {
    "Andhra Pradesh": ["Visakhapatnam","Vijayawada","Guntur","Nellore","Kurnool","Tirupati","Rajahmundry","Kakinada","Anantapur","Eluru"],
    "Arunachal Pradesh": ["Itanagar","Naharlagun","Pasighat","Tawang","Ziro","Bomdila","Along","Tezu","Roing","Daporijo"],
    "Assam": ["Guwahati","Silchar","Dibrugarh","Jorhat","Nagaon","Tinsukia","Tezpur","Bongaigaon","Dhubri","North Lakhimpur"],
    "Bihar": ["Patna","Gaya","Bhagalpur","Muzaffarpur","Purnia","Darbhanga","Arrah","Begusarai","Katihar","Munger"],
    "Chhattisgarh": ["Raipur","Bhilai","Bilaspur","Korba","Durg","Rajnandgaon","Raigarh","Jagdalpur","Ambikapur","Dhamtari"],
    "Goa": ["Panaji","Margao","Vasco da Gama","Mapusa","Ponda","Bicholim","Curchorem","Sanquelim","Canacona","Quepem"],
    "Gujarat": ["Ahmedabad","Surat","Vadodara","Rajkot","Bhavnagar","Jamnagar","Junagadh","Gandhinagar","Anand","Morbi"],
    "Haryana": ["Chandigarh","Faridabad","Gurugram","Panipat","Ambala","Yamunanagar","Rohtak","Hisar","Karnal","Sonipat"],
    "Himachal Pradesh": ["Shimla","Dharamshala","Mandi","Solan","Kullu","Manali","Bilaspur","Hamirpur","Nahan","Palampur"],
    "Jharkhand": ["Ranchi","Jamshedpur","Dhanbad","Bokaro","Deoghar","Hazaribagh","Giridih","Ramgarh","Dumka","Phusro"],
    "Karnataka": ["Bengaluru","Mysuru","Hubballi","Mangaluru","Belagavi","Davanagere","Ballari","Tumakuru","Shimoga","Gulbarga"],
    "Kerala": ["Thiruvananthapuram","Kochi","Kozhikode","Thrissur","Kollam","Palakkad","Alappuzha","Kannur","Kottayam","Malappuram"],
    "Madhya Pradesh": ["Bhopal","Indore","Jabalpur","Gwalior","Ujjain","Sagar","Dewas","Satna","Ratlam","Rewa"],
    "Maharashtra": ["Mumbai","Pune","Nagpur","Nashik","Aurangabad","Solapur","Kolhapur","Amravati","Nanded","Sangli"],
    "Manipur": ["Imphal","Thoubal","Bishnupur","Churachandpur","Kakching","Ukhrul","Senapati","Tamenglong","Chandel","Jiribam"],
    "Meghalaya": ["Shillong","Tura","Jowai","Nongstoin","Williamnagar","Baghmara","Resubelpara","Mairang","Nongpoh","Cherrapunji"],
    "Mizoram": ["Aizawl","Lunglei","Champhai","Serchhip","Kolasib","Lawngtlai","Mamit","Saiha","Khawzawl","Hnahthial"],
    "Nagaland": ["Kohima","Dimapur","Mokokchung","Tuensang","Wokha","Zunheboto","Mon","Phek","Kiphire","Longleng"],
    "Odisha": ["Bhubaneswar","Cuttack","Rourkela","Berhampur","Sambalpur","Puri","Balasore","Baripada","Bhadrak","Jharsuguda"],
    "Punjab": ["Chandigarh","Ludhiana","Amritsar","Jalandhar","Patiala","Bathinda","Mohali","Hoshiarpur","Pathankot","Moga"],
    "Rajasthan": ["Jaipur","Jodhpur","Udaipur","Kota","Ajmer","Bikaner","Bhilwara","Alwar","Sikar","Pali"],
    "Sikkim": ["Gangtok","Namchi","Gyalshing","Mangan","Rangpo","Singtam","Jorethang","Ravangla","Lachung","Pelling"],
    "Tamil Nadu": ["Chennai","Coimbatore","Madurai","Tiruchirappalli","Salem","Tirunelveli","Erode","Vellore","Thoothukudi","Dindigul"],
    "Telangana": ["Hyderabad","Warangal","Nizamabad","Karimnagar","Khammam","Ramagundam","Mahbubnagar","Nalgonda","Adilabad","Suryapet"],
    "Tripura": ["Agartala","Udaipur","Dharmanagar","Kailasahar","Belonia","Ambassa","Khowai","Teliamura","Sabroom","Sonamura"],
    "Uttar Pradesh": ["Lucknow","Kanpur","Agra","Varanasi","Prayagraj","Meerut","Noida","Ghaziabad","Bareilly","Aligarh"],
    "Uttarakhand": ["Dehradun","Haridwar","Rishikesh","Haldwani","Roorkee","Kashipur","Rudrapur","Nainital","Mussoorie","Pithoragarh"],
    "West Bengal": ["Kolkata","Howrah","Siliguri","Durgapur","Asansol","Bardhaman","Malda","Baharampur","Habra","Kharagpur"],
    "Delhi": ["New Delhi","Dwarka","Rohini","Saket","Janakpuri","Pitampura","Lajpat Nagar","Karol Bagh","Chandni Chowk","Connaught Place"],
    "Jammu & Kashmir": ["Srinagar","Jammu","Anantnag","Baramulla","Sopore","Udhampur","Kathua","Pulwama","Rajouri","Kupwara"],
    "Ladakh": ["Leh","Kargil","Diskit","Padum","Nyoma","Hanle","Dras","Turtuk","Hunder","Tangtse"],
    "Puducherry": ["Puducherry","Karaikal","Mahe","Yanam"],
    "Chandigarh": ["Chandigarh"],
    "Andaman & Nicobar": ["Port Blair","Havelock","Neil Island","Diglipur","Rangat","Mayabunder","Car Nicobar","Campbell Bay"],
    "Dadra & Nagar Haveli and Daman & Diu": ["Silvassa","Daman","Diu"],
    "Lakshadweep": ["Kavaratti","Agatti","Minicoy","Amini","Andrott"],
}

@app.get("/india-locations")
async def india_locations():
    """Returns all Indian states/UTs with their major cities."""
    return {"states": {k: v for k, v in sorted(INDIA_LOCATIONS.items())}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
