# 🌾 Farmer AI — Professional Smart Crop Intelligence Platform

Farmer AI is a high-performance, professional-grade agricultural intelligence platform designed to empower Indian farmers with data-driven decision-making. It combines advanced machine learning models, real-time data integration, and an intelligent NLP-based advisory system within a premium, SaaS-style user interface.

![Farmer AI UI](https://img.shields.io/badge/UI-Premium_Glassmorphism-emerald)
![ML Accuracy](https://img.shields.io/badge/ML_Accuracy-88%25-blue)
![Yield R2](https://img.shields.io/badge/Yield_R2-97.8%25-gold)

## 🚀 Key Features

### 1. Smart Crop Recommendation
- **Predictive Engine:** Powered by a Random Forest Classifier trained on **5,430+ rows** of high-fidelity agronomic data.
- **Coverage:** Supports **48 diverse Indian crops** including cereals, pulses, oilseeds, fruits, vegetables, spices, and plantation crops.
- **Accuracy:** Achieves **~88% accuracy** with 5-fold cross-validation.

### 2. Precision Yield Estimation
- **XGBoost Pipeline:** High-precision yield forecasting with an **R² score of 97.8%**.
- **Actionable Insights:** Helps farmers estimate production before sowing based on NPK, pH, and climate data.

### 3. Intelligent AI Advisor (NLP)
- **Context-Aware Assistance:** Built-in knowledge base (`advisor_kb.py`) covering 12+ critical agricultural topics.
- **Real-time Advice:** Detailed guidance on irrigation, fertilizers, pest control, organic farming, and harvest schedules.
- **Crop-Specific Logic:** Deep technical data for major Indian staples like rice, wheat, potato, tomato, and onion.

### 4. Weather & Market Intelligence
- **Live Weather Dashboard:** Real-time local weather data with AI-generated farming advisories (heat stress, frost alerts, rainfall management).
- **Market Trends:** Live tracking of crop prices and MSP comparison with trend analysis.

### 5. Professional SaaS UI
- **Refined Aesthetics:** Clean, dark-mode interface with emerald/gold accents and high-end glassmorphism.
- **Cinematic Backgrounds:** Contextual, high-resolution agricultural photography backgrounds for every module (Weather, Market, Advisor, Predict).
- **Responsive Design:** Optimized for field use across devices.

## 🛠️ Technology Stack

- **Backend:** 
  - [FastAPI](https://fastapi.tiangolo.com/) (High-performance Python web framework)
  - [Scikit-learn](https://scikit-learn.org/) & [XGBoost](https://xgboost.readthedocs.io/) (Machine Learning)
  - [Joblib](https://joblib.readthedocs.io/) (Model Serialization)
- **Frontend:** 
  - [React](https://reactjs.org/) with [Vite](https://vitejs.dev/)
  - [Vanilla CSS](https://developer.mozilla.org/en-US/docs/Web/CSS) (Custom Premium Design System)
  - [Lucide React](https://lucide.dev/) (Icons)

## 📦 Installation & Setup

### Backend
1. Navigate to the `backend` directory.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend
1. Navigate to the `frontend` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## 📊 Dataset Information
The models are trained on a comprehensive dataset tailored for Indian agro-climatic conditions, featuring:
- **Features:** Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall.
- **Targets:** Crop Type (Classification), Yield (Regression).

---
*Developed with a focus on empowering the next generation of digital-first farmers.*
