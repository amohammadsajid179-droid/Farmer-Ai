"""
Farmer AI - Intelligent Advisor Knowledge Base v2.0
Comprehensive rule-based NLP chatbot for agriculture
"""
import re

# ── Crop-specific detailed knowledge ─────────────────────────────────────────
CROP_KB = {
    "rice": {
        "growing": "Rice needs 20-35°C, standing water 5-10cm during seedling stage. Transplant 25-day-old seedlings. Use SRI method for 30% higher yield.",
        "fertilizer": "Apply 120-150 kg N/ha in 3 splits: 50% basal, 25% at tillering, 25% at panicle. DAP 60 kg/ha basal. MOP 60 kg/ha.",
        "irrigation": "Maintain 5cm standing water from transplanting to flowering. Drain 15 days before harvest. Total water need: 1200-1400mm.",
        "pest": "Major pests: Stem borer (Cartap 4G 25kg/ha), BPH (Imidacloprid 0.5ml/L), Blast (Tricyclazole 0.6g/L). Scout weekly.",
        "harvest": "Harvest when 80% grains are golden. Moisture should be 20-22%. Dry to 14% for safe storage.",
        "diseases": "Blast, Bacterial Leaf Blight, Sheath Rot. Preventive: Tricyclazole spray at boot stage. Avoid excess nitrogen.",
    },
    "wheat": {
        "growing": "Wheat needs 10-25°C, well-drained loamy soil. Sow in Nov (North India). Seed rate 100 kg/ha. Row spacing 20-22.5cm.",
        "fertilizer": "Apply 120 kg N, 60 kg P2O5, 40 kg K2O per ha. N in 3 splits: half basal, 1/4 at CRI, 1/4 at boot stage.",
        "irrigation": "Critical stages: CRI (21 days), tillering (40-45), jointing (60-65), flowering (80-85), grain filling (100-105). Total 4-6 irrigations.",
        "pest": "Aphids: Dimethoate 30EC 1ml/L. Termites: Chlorpyriphos 20EC soil drench. Rust: Propiconazole 25EC 1ml/L.",
        "harvest": "Harvest at golden yellow stage. Grain moisture 12-14%. Thresh within 2-3 days. Store in dry conditions.",
        "diseases": "Yellow rust, Brown rust, Karnal bunt. Spray Propiconazole at flag leaf stage preventively.",
    },
    "potato": {
        "growing": "Plant in Oct-Nov (Rabi). Seed rate 25-30 q/ha. Spacing 60x20cm. Earthing up at 30 and 45 days critical.",
        "fertilizer": "Apply 150-200 kg N, 80 kg P2O5, 120 kg K2O per ha. FYM 25 t/ha. Split N in 3 doses.",
        "irrigation": "Irrigate every 10-12 days. Critical at stolon formation and tuber bulking. Avoid waterlogging.",
        "pest": "Late blight: Mancozeb 2.5g/L preventive. Cutworm: Chlorpyriphos soil application. Aphids: Imidacloprid.",
        "harvest": "Harvest 15-20 days after haulm cutting. Cure tubers 10-15 days at 15°C. Store in cold storage 2-4°C.",
    },
    "tomato": {
        "growing": "Transplant 4-week seedlings. Spacing 60x45cm. Staking increases yield 30%. Needs 20-30°C.",
        "fertilizer": "Apply 120 kg N, 80 kg P2O5, 80 kg K2O per ha. Calcium application prevents blossom end rot.",
        "irrigation": "Drip irrigation best. Critical at flowering and fruit development. Mulching reduces water need 40%.",
        "pest": "Fruit borer: Emamectin 0.4g/L. Whitefly: Yellow sticky traps + Imidacloprid. Late blight: Metalaxyl+Mancozeb.",
    },
    "onion": {
        "growing": "Transplant 6-week seedlings. Spacing 15x10cm. Kharif (Jun-Jul) or Rabi (Dec-Jan). Needs 15-25°C.",
        "fertilizer": "Apply 100 kg N, 50 kg P2O5, 80 kg K2O per ha. Sulphur 40 kg/ha improves pungency and storage.",
        "irrigation": "Light frequent irrigation. Stop 10 days before harvest for better curing and storage life.",
        "pest": "Thrips: Fipronil 5SC 1ml/L. Purple blotch: Mancozeb 2.5g/L. Stemphylium blight: Chlorothalonil.",
    },
}

# ── Topic knowledge base with fuzzy matching ─────────────────────────────────
TOPIC_KB = {
    "irrigation": {
        "keywords": ["water", "irrigat", "drought", "dry", "moisture", "drip", "flood", "sprinkler", "canal", "pani", "sinchai"],
        "response": "💧 **Irrigation Best Practices:**\n- Water early morning (6-8 AM) to minimize evaporation\n- **Drip irrigation** saves 40-60% water vs flood irrigation\n- Check soil moisture 5cm deep — irrigate when dry to touch\n- Sandy soils: irrigate every 3-4 days\n- Clayey soils: irrigate every 7-10 days\n- **Mulching** (straw/plastic) reduces water need by 25-35%\n- Rainwater harvesting reduces dependency by 30%\n- Avoid irrigation during peak afternoon heat\n- Critical stages vary by crop — always irrigate at flowering stage"
    },
    "fertilizer": {
        "keywords": ["fertil", "npk", "nitrogen", "phosphor", "potassium", "urea", "dap", "mop", "manure", "compost", "nutrient", "khad", "micronutrient", "zinc", "boron", "sulphur"],
        "response": "🧪 **Fertilizer Application Guide:**\n- Always do **soil testing** before applying fertilizers\n- Apply in **split doses** (basal + 2-3 top dressings)\n- **Urea** (46% N): Never apply in standing water — use LCC method\n- **DAP** (18N + 46P): Best as basal at sowing\n- **MOP** (60% K2O): Split between basal and 1st top dressing\n- **FYM/Compost** 5-10 t/acre improves soil structure long-term\n- Avoid fertilizers before heavy rain (nutrient loss by leaching)\n- **Micronutrients**: Zinc Sulphate 25 kg/ha, Borax 5 kg/ha as needed\n- Apply organic + inorganic together for best results (INM approach)"
    },
    "pest": {
        "keywords": ["pest", "insect", "disease", "fungal", "virus", "worm", "borer", "aphid", "bug", "caterpillar", "mite", "beetle", "fly", "rot", "blight", "wilt", "rust", "smut", "mildew", "keeda", "rog"],
        "response": "🐛 **Integrated Pest Management (IPM):**\n- **Scout** fields every 3-5 days during critical growth stages\n- Use **yellow sticky traps** for whiteflies, **pheromone traps** for borers\n- **Neem oil** 3ml/L is effective organic preventive spray\n- **Trichogramma** cards for borer control in rice/sugarcane\n- Chemical sprays: apply early morning or evening only\n- **Rotate** insecticides to prevent resistance buildup\n- Maintain field hygiene — remove crop debris post harvest\n- **Seed treatment** with Thiram/Carbendazim prevents seed-borne diseases\n- Use disease-resistant varieties when available"
    },
    "soil": {
        "keywords": ["soil", "ph", "acid", "alkali", "organic", "clay", "sandy", "loam", "mitti", "health", "testing", "erosion", "saline"],
        "response": "🌱 **Soil Health Management:**\n- **Ideal pH**: 6.0-7.5 for most crops\n- **Acidic (pH < 5.5)**: Apply agricultural lime 200-500 kg/acre\n- **Alkaline (pH > 8.0)**: Apply gypsum 500 kg/acre\n- **Add organic matter** (FYM/compost) every season — at least 5 t/acre\n- Do **soil testing** every 2-3 years at your nearest KVK\n- **Green manuring** with dhaincha/sunhemp improves nitrogen naturally\n- Practice **crop rotation** to maintain soil fertility\n- Avoid soil compaction — use minimal tillage where possible\n- **Vermicompost** is excellent for micronutrient supply"
    },
    "harvest": {
        "keywords": ["harvest", "yield", "reap", "cut", "collect", "matur", "storage", "store", "post-harvest", "drying", "threshing"],
        "response": "🌾 **Harvesting & Post-Harvest:**\n- Harvest at **physiological maturity** for maximum yield\n- Avoid harvesting wet crops — wait 2-3 days after rain\n- Use sharp, clean implements to minimize crop damage\n- **Thresh** within 2-3 days to prevent storage losses\n- Maintain **grain moisture below 12-14%** for safe storage\n- Store in moisture-proof bags in ventilated rooms\n- Use **hermetic storage** bags (PICS bags) for longer storage\n- Treat stored grain with Aluminium Phosphide for insect control\n- **Grade and sort** produce for better market price"
    },
    "season": {
        "keywords": ["season", "kharif", "rabi", "zaid", "when to sow", "planting time", "calendar", "month", "sowing"],
        "response": "📅 **Crop Season Guide (India):**\n- **Kharif (Jun-Nov):** Rice, Maize, Cotton, Soybean, Groundnut, Sorghum, Pearl Millet, Jute, Sugarcane\n- **Rabi (Oct-Mar):** Wheat, Barley, Chickpea, Mustard, Lentil, Peas, Potato, Onion, Garlic\n- **Zaid (Mar-Jun):** Watermelon, Muskmelon, Cucumber, Moong, Vegetables\n- **Perennial:** Sugarcane, Banana, Coconut, Mango, Coffee, Tea, Rubber\n- Sow 2-3 weeks before optimal date for germination buffer\n- Check your **district agriculture calendar** for local timings"
    },
    "weather": {
        "keywords": ["weather", "rain", "temperature", "climate", "frost", "heat wave", "flood", "cyclone", "hail", "mausam"],
        "response": "🌤️ **Weather-Smart Farming Tips:**\n- Use our **Weather Dashboard** for live conditions\n- **Heat wave (>40°C)**: irrigate, apply mulch, shade sensitive crops\n- **Frost risk (<5°C)**: cover seedlings with straw, avoid night irrigation\n- **Heavy rain forecast**: delay fertilizer application by 3-4 days\n- **High humidity (>85%)**: apply preventive fungicide spray\n- **Hailstorm**: insure crops under PMFBY scheme\n- Monitor **IMD forecasts** before major field operations"
    },
    "market": {
        "keywords": ["market", "price", "sell", "mandi", "msp", "income", "profit", "revenue", "cost", "economics", "loan", "insurance", "subsidy"],
        "response": "💰 **Market & Finance Tips:**\n- Check **eNAM** portal for real-time mandi prices across India\n- Sell at **APMC/Regulated markets** for fair price\n- **MSP (Minimum Support Price)** is guaranteed by govt for 23 crops\n- Use our **Market Prices** page for current rates\n- **PMFBY** crop insurance: premium 1.5-5% only, covers all risks\n- **Kisan Credit Card (KCC)**: loan at 4% interest for farmers\n- **PM-KISAN**: ₹6000/year direct benefit for all farmers\n- Grade and sort produce to get 15-25% higher price\n- Consider **Farmer Producer Organizations (FPO)** for collective bargaining"
    },
    "organic": {
        "keywords": ["organic", "natural", "chemical-free", "bio", "jaivik", "green manure", "vermicompost", "biofertilizer"],
        "response": "🌿 **Organic Farming Guide:**\n- Use **FYM** (10-15 t/ha), **vermicompost** (5 t/ha), **green manure**\n- **Biofertilizers**: Rhizobium (pulses), Azotobacter (cereals), PSB (all crops)\n- **Neem cake** 250 kg/ha acts as both fertilizer and pest repellent\n- **Trichoderma** for soil-borne disease management\n- **Panchagavya** (3% spray) promotes growth naturally\n- **Jeevamrutha**: fermented cow dung preparation for soil health\n- Organic certification takes 3 years but premium price is 25-50% higher\n- Practice **crop rotation** and **intercropping** for pest suppression"
    },
    "seed": {
        "keywords": ["seed", "variety", "hybrid", "beej", "germination", "sowing", "transplant", "nursery"],
        "response": "🌱 **Seed & Sowing Guide:**\n- Always use **certified/truthful seeds** from authorized dealers\n- **Seed treatment**: Thiram/Carbendazim 2-3g/kg prevents seed-borne diseases\n- **Germination test**: Place 100 seeds on wet cloth — count sprouted after 7 days\n- **Hybrid seeds** give 20-30% higher yield but cannot be reused\n- Maintain **proper seed rate** — excess reduces individual plant yield\n- **Raised nursery beds** for transplanted crops (rice, tomato, onion)\n- Store seeds in cool dry place — viability decreases with heat/moisture"
    },
    "technology": {
        "keywords": ["technology", "drone", "sensor", "app", "digital", "precision", "gps", "satellite", "iot", "smart"],
        "response": "📱 **AgriTech & Precision Farming:**\n- **Soil moisture sensors** optimize irrigation timing\n- **Drone spraying** covers 1 acre in 10 min vs 3 hours manual\n- **Satellite imagery** (NDVI) detects crop stress early\n- **Weather stations** on-farm for microclimate monitoring\n- **GPS-guided** tractors reduce overlap and save fuel 15-20%\n- **IoT drip systems** automate irrigation based on soil moisture\n- Use **Farmer AI** for crop recommendations and yield prediction!"
    },
    "livestock": {
        "keywords": ["cattle", "cow", "buffalo", "goat", "poultry", "dairy", "milk", "animal", "pashu", "murgi", "egg"],
        "response": "🐄 **Integrated Farming with Livestock:**\n- **Dairy**: HF cow gives 15-20 L/day, Murrah buffalo 8-12 L/day\n- **FYM from livestock** reduces fertilizer cost by 30-40%\n- **Biogas** from cow dung: 1 cow produces enough gas for family cooking\n- **Azolla** cultivation: excellent feed supplement, doubles milk yield\n- **Poultry** (100 birds): ₹3000-5000/month income with low investment\n- **Goat farming**: 3 kiddings in 2 years, high demand for meat\n- Integrate **fish farming** in farm ponds for additional income"
    },
}

# ── Greetings and conversational patterns ─────────────────────────────────────
GREETING_PATTERNS = ["hello", "hi", "hey", "namaste", "good morning", "good evening", "good afternoon", "help", "start"]
THANKS_PATTERNS = ["thank", "thanks", "dhanyavad", "shukriya", "great", "awesome", "helpful", "good answer"]

GREETING_RESPONSE = "👋 **Namaste! Welcome to Farmer AI Advisor!**\n\nI'm your expert agricultural assistant. I can help with:\n\n- 🌾 **Crop advice** — growing tips for 48+ crops\n- 💧 **Irrigation** — when and how much to water\n- 🧪 **Fertilizers** — NPK recommendations & schedules\n- 🐛 **Pest & Disease** — identification and treatment\n- 🌱 **Soil Health** — pH correction and amendments\n- 📅 **Crop Calendar** — best sowing and harvest times\n- 🌤️ **Weather** — climate-smart farming tips\n- 💰 **Market & Finance** — prices, subsidies, loans\n- 🌿 **Organic farming** — natural methods\n\nJust type your question! For example:\n*'How to grow rice?'* or *'Best fertilizer for wheat?'*"

THANKS_RESPONSE = "🙏 **You're welcome!** Happy to help. Feel free to ask more farming questions anytime. Good farming leads to good harvests! 🌾"


def find_crop_in_query(query):
    """Extract crop name from query."""
    all_crops = list(CROP_KB.keys()) + [
        "maize", "barley", "sorghum", "pearl_millet", "finger_millet",
        "chickpea", "lentil", "pigeonpeas", "mungbean", "blackgram",
        "kidneybeans", "mothbeans", "cowpea", "peas", "mustard",
        "groundnut", "soybean", "sunflower", "sesame", "linseed", "castor",
        "sugarcane", "cotton", "jute", "tobacco", "banana", "mango",
        "grapes", "watermelon", "apple", "orange", "papaya", "coconut",
        "pomegranate", "guava", "garlic", "ginger", "turmeric",
        "pepper", "cardamom", "coffee", "tea", "rubber",
    ]
    query_lower = query.lower()
    for crop in all_crops:
        if crop in query_lower or crop.replace("_", " ") in query_lower:
            return crop
    return None


def get_smart_response(question, crop=None, N=None, P=None, K=None, ph=None, temperature=None, humidity=None, rainfall=None):
    """Generate intelligent contextual response."""
    q = question.lower().strip()
    answers = []

    # Check greetings
    if any(g in q for g in GREETING_PATTERNS) and len(q.split()) <= 5:
        return [GREETING_RESPONSE], ["How to grow rice?", "Best fertilizer for wheat?", "When to sow potato?"]

    # Check thanks
    if any(t in q for t in THANKS_PATTERNS) and len(q.split()) <= 6:
        return [THANKS_RESPONSE], ["Tell me about organic farming", "How to control pests?"]

    # Find crop in query or from context
    detected_crop = find_crop_in_query(q) or (crop.lower() if crop else None)

    # Match topics
    for topic, data in TOPIC_KB.items():
        if any(kw in q for kw in data["keywords"]):
            answers.append(data["response"])

    # Crop-specific knowledge
    if detected_crop and detected_crop in CROP_KB:
        kb = CROP_KB[detected_crop]
        crop_name = detected_crop.replace("_", " ").title()
        # Find which aspect they're asking about
        if any(w in q for w in ["grow", "cultivat", "plant", "sow", "how to", "tips", "guide", "tell me about"]):
            answers.append(f"🌱 **{crop_name} — Growing Guide:**\n{kb.get('growing', 'Follow standard practices.')}")
        if any(w in q for w in ["fertil", "npk", "nutrient", "urea", "dap", "khad"]):
            answers.append(f"🧪 **{crop_name} — Fertilizer:**\n{kb.get('fertilizer', 'Apply balanced NPK.')}")
        if any(w in q for w in ["water", "irrigat", "pani", "drip"]):
            answers.append(f"💧 **{crop_name} — Irrigation:**\n{kb.get('irrigation', 'Irrigate at critical stages.')}")
        if any(w in q for w in ["pest", "disease", "insect", "spray", "keeda", "rog", "blight", "wilt"]):
            answers.append(f"🐛 **{crop_name} — Pest & Disease:**\n{kb.get('pest', 'Use IPM practices.')}")
        if any(w in q for w in ["harvest", "storage", "store", "yield", "matur"]):
            answers.append(f"🌾 **{crop_name} — Harvest:**\n{kb.get('harvest', 'Harvest at physiological maturity.')}")
        # If no specific aspect matched, give growing guide
        if not answers:
            answers.append(f"🌱 **{crop_name} — Complete Guide:**\n{kb.get('growing', '')}\n\n🧪 **Fertilizer:** {kb.get('fertilizer', '')}\n\n💧 **Irrigation:** {kb.get('irrigation', '')}")

    # Contextual soil/weather advice from provided values
    ctx_tips = []
    if N is not None and N < 40:
        ctx_tips.append(f"⚠️ **Low Nitrogen ({N:.0f} kg/ha)**: Apply Urea 65-70 kg/acre or FYM 5t/acre.")
    if N is not None and N > 150:
        ctx_tips.append(f"⚠️ **Very High Nitrogen ({N:.0f} kg/ha)**: Risk of lodging and delayed maturity. Reduce N application.")
    if P is not None and P < 20:
        ctx_tips.append(f"⚠️ **Low Phosphorus ({P:.0f} kg/ha)**: Apply DAP 50 kg/acre or SSP 100 kg/acre.")
    if K is not None and K < 20:
        ctx_tips.append(f"⚠️ **Low Potassium ({K:.0f} kg/ha)**: Apply MOP 33 kg/acre.")
    if ph is not None and ph < 5.5:
        ctx_tips.append(f"⚠️ **Acidic soil (pH {ph:.1f})**: Apply Lime 200-500 kg/acre to raise pH.")
    if ph is not None and ph > 8.0:
        ctx_tips.append(f"⚠️ **Alkaline soil (pH {ph:.1f})**: Apply Gypsum 500 kg/acre to lower pH.")
    if temperature is not None and temperature > 38:
        ctx_tips.append(f"🌡️ **Heat stress ({temperature:.1f}°C)**: Irrigate early morning, apply mulch, consider shade nets for sensitive crops.")
    if temperature is not None and temperature < 5:
        ctx_tips.append(f"❄️ **Frost risk ({temperature:.1f}°C)**: Cover seedlings with straw/cloth. Avoid night irrigation.")
    if humidity is not None and humidity > 85:
        ctx_tips.append(f"💧 **High humidity ({humidity:.0f}%)**: Elevated fungal disease risk. Apply preventive fungicide.")
    if humidity is not None and humidity < 30:
        ctx_tips.append(f"🏜️ **Very low humidity ({humidity:.0f}%)**: Increase irrigation frequency. Use mulching.")
    if rainfall is not None and rainfall > 300:
        ctx_tips.append(f"🌧️ **Heavy rainfall ({rainfall:.0f}mm)**: Ensure drainage. Avoid fertilizer application.")

    if ctx_tips:
        answers.append("📊 **Based on your farm data:**\n" + "\n".join(ctx_tips))

    # Fallback if nothing matched
    if not answers:
        # Try to give a helpful response based on any word match
        answers = ["I can help with many farming topics! Here are some things you can ask:\n\n"
                   "🌾 **Crop growing**: *'How to grow rice?'* or *'Tips for wheat cultivation'*\n"
                   "💧 **Irrigation**: *'When to water potato?'* or *'Drip irrigation tips'*\n"
                   "🧪 **Fertilizers**: *'Best fertilizer for tomato?'* or *'NPK for sugarcane'*\n"
                   "🐛 **Pest control**: *'How to control aphids?'* or *'Rice blast treatment'*\n"
                   "🌱 **Soil health**: *'How to improve soil pH?'* or *'Organic matter tips'*\n"
                   "📅 **Seasons**: *'When to sow wheat?'* or *'Kharif crops list'*\n"
                   "💰 **Market**: *'Current MSP rates'* or *'How to sell at mandi?'*\n"
                   "🌿 **Organic**: *'How to make vermicompost?'*\n\n"
                   "Type your question and I'll give you expert advice!"]

    # Generate smart suggestions
    suggestions = ["How to improve soil pH?", "Best fertilizer for rice?", "When to harvest wheat?", "How to control aphids?"]
    if detected_crop:
        cn = detected_crop.replace("_", " ").title()
        suggestions = [f"Best fertilizer for {cn}?", f"Irrigation schedule for {cn}?", f"Pest control for {cn}?", f"When to harvest {cn}?"]

    return answers, suggestions
