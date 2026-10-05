# ERFlow — Emergency Department Demand Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://er-flow-pateint-flow-prediction-5qk.vercel.app/)

> 🚀 **Live Web Application**: [https://er-flow-pateint-flow-prediction-5qk.vercel.app/](https://er-flow-pateint-flow-prediction-5qk.vercel.app/)

ERFlow is a full-stack emergency department demand forecasting, crowding risk classification, anomaly surge detection, and retrieval-augmented domain guidance system. It combines **5 machine learning models**, an **in-memory & ChromaDB RAG vector store**, a **Direct NLP AI Assistant**, and a **custom-styled React dashboard**.

---


## 🌟 Key Capabilities & Features

### 1. Multi-Model ML Inference Engine
- **Supervised XGBoost Regressor**: Predicts patient waiting times (in minutes) and feature importances using SHAP values.
- **Supervised XGBoost Classifier**: Classifies emergency department crowding risk (*LOW*, *MODERATE*, *HIGH*, *CRITICAL*) with class probability distributions.
- **Unsupervised K-Means + PCA**: Clusters ER operational regimes (*Low Demand*, *Medium Demand*, *High Demand*) and maps 2D PCA coordinates.
- **Unsupervised DBSCAN Anomaly Engine**: Detects operational arrival surges and deviation percentages from historical baselines.
- **Deep Learning LSTM Neural Network**: Generates multi-horizon patient arrival forecasts (*1h*, *3h*, *6h*, *24h*) using continuous historical dataset trends.

### 2. Retrieval-Augmented Generation (RAG) & Direct NLP Assistant
- **ChromaDB Vector Store**: Persists document passages from clinical triage guidelines, ESI protocols, and hospital surge management rules.
- **Resilient Fallback Engine**: High-performance in-memory TF-IDF vector similarity search if ChromaDB native embeddings are offline.
- **Direct Domain NLP Engine**: Instantly answers queries on ESI Triage Levels (1–5), Surge Protocols (*Code Yellow / Code Red*), Door-to-Provider targets, LWBS limits, and 5 proven wait-time reduction strategies (Fast-track, PIT, POCT, DBN, Hallway beds).

### 3. Real-Time Emergency Operations & Scenario Simulator
- **Live System Synchronization**: Updating any operational variable (Occupancy %, Patients Waiting, Arrival Velocity, Active Physicians/Nurses) automatically triggers real-time predictions across all 5 ML models.
- **Interactive Scenario Simulator**: Compare baseline ER conditions against custom operational scenarios or presets (*Quiet Shift*, *Busy Evening*, *Surge Scenario*).
- **Dynamic Date-Aware Forecast Timelines**: Switch between 24-hour, 7-day, and 30-day forecast ranges built dynamically from the live system clock.

### 4. Custom 5-Color Light Theme Palette
Designed with a clean, high-contrast light theme color scheme:
- 🩵 **Scooter** (`#2F9D94`): Secondary accents, icons, and highlights.
- 🤍 **Alabaster** (`#F7F6F2`): Primary background and subtle container surfaces.
- 🩶 **Heather** (`#BCC5CC`): Borders, neutral dividers, and muted text.
- 🌊 **Blue Lagoon** (`#025F67`): Primary brand color, headers, and trend lines.
- 💙 **Sapphire** (`#063154`): Deep navy text and high-contrast typography.

---

## 📐 Project Architecture

```
ERFlow-Pateint-Flow-Prediction/
├── backend/                        # FastAPI ML Inference Backend
│   ├── main.py                     # Primary FastAPI application entry point & routes
│   ├── rag/                        # RAG Subsystem
│   │   ├── config.py               # Path definitions & RAG settings
│   │   ├── document_loader.py      # Multi-format document ingester (.md, .txt, .pdf)
│   │   ├── embeddings.py           # SentenceTransformers embedding engine
│   │   ├── text_splitter.py        # Section-aware sliding window document splitter
│   │   ├── vector_store.py         # ChromaDB & TF-IDF fallback vector store
│   │   └── retriever.py            # Semantic retrieval & context formatter
│   └── knowledge_base/             # Clinical guidelines & operational protocol documents
├── ml_model/                       # Trained ML model artifacts (.pkl, .joblib, .h5, .json)
├── chatbot/                        # Chatbot Microservice & NLP Engine
│   └── app/
│       ├── chatbot/
│       │   ├── nlp_responder.py    # Direct domain answer engine (ESI, surge, metrics)
│       │   ├── chatbot_service.py  # Orchestrator (Safety -> Direct Answer -> RAG -> Fallback)
│       │   └── safety_guard.py     # Medical diagnosis refusal safety gate
│       └── main.py                 # Chatbot standalone FastAPI app
├── src/                            # React + Vite Frontend
│   ├── context/
│   │   ├── ERContext.jsx           # Central operational state & multi-model prediction dispatcher
│   │   └── ModeContext.jsx         # REAL ML vs DEMO mode toggle
│   ├── dashboard/
│   │   ├── pages/                  # Overview, PatientForecast, CrowdingRisk, ScenarioSimulator, AIAssistant
│   │   └── components/             # EROperationsControlPanel, TrendChart, MetricCard, PageHeader
│   └── index.css                   # Tailwind utilities & 5-color palette tokens
└── start-all.ps1                   # One-click startup script for PowerShell
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: Version 3.11
- **Node.js**: Version 18.0 or higher (`npm` included)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/smurtiranikhadanga/ERFlow-Pateint-Flow-Prediction.git
cd ERFlow-Pateint-Flow-Prediction

# Copy environment template
cp .env.example .env

# Create and activate Python virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install backend & chatbot dependencies
pip install -r backend/requirements.txt
pip install -r chatbot/requirements.txt

# Install frontend dependencies
npm ci
```

### 2. Launching Services

#### Option A: One-Click Startup Script (Windows PowerShell)
```powershell
.\start-all.ps1
```

#### Option B: Manual Service Startup
Run each process in a separate terminal:

**Terminal 1 — Backend FastAPI ML Engine (Port 8000)**:
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --port 8000 --reload
```

**Terminal 2 — Chatbot Microservice (Port 8001)**:
```powershell
$env:PYTHONPATH="chatbot"
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --port 8001 --reload
```

**Terminal 3 — React Frontend (Port 5173)**:
```bash
npm run dev
```

Access the dashboard in your web browser at **`http://localhost:5173`**.

---

## 🛰️ Core API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Health check for FastAPI backend & loaded ML models |
| `/api/dashboard/overview` | `POST` | Single request returning full predictions across all 5 ML models |
| `/api/predict/waiting-time` | `POST` | XGBoost Regressor predicted wait times & SHAP feature importances |
| `/api/predict/crowding-risk` | `POST` | XGBoost Classifier crowding risk levels & probability breakdown |
| `/api/patterns/flow` | `POST` | K-Means clustering regime & 2D coordinate projections |
| `/api/surge/detect` | `POST` | DBSCAN anomaly surge detection & historical baseline deviations |
| `/api/chat` | `POST` | AI Assistant query endpoint (Direct NLP + RAG Knowledge Retrieval) |

---

## 🛠️ Build & Verification

To verify production bundle build integrity:
```bash
npm run build
```
Target production artifacts will compile cleanly into the `dist/` directory.
