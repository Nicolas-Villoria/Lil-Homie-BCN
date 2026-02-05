# Barcelona Property Price Predictor

A production-grade machine learning system for predicting property sale prices in Barcelona. The system processes real estate data from Idealista combined with socioeconomic indicators from Open Data BCN to deliver accurate price predictions through a REST API and interactive web interface.

**Live Demo:** [Streamlit App](https://bcn-housing-price-predictor.streamlit.app) | **API:** [Render](https://bcn-housing-price-predictor.onrender.com)

---

## Project Overview

This project implements an end-to-end ML pipeline covering data engineering, model training, deployment, and monitoring. It demonstrates practical MLOps patterns suitable for production environments.

### Features

- Property characteristics: size, rooms, bathrooms, property type
- Location: neighborhood and district
- Socioeconomic context: average income index, population density
- Training data: 70,000+ property listings from Barcelona (2020-2023)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Data Sources                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   Idealista     │  │  OpenData BCN   │  │  OpenData BCN   │          │
│  │   Listings      │  │  Income Index   │  │  Density Data   │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
└───────────┼────────────────────┼────────────────────┼───────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Lake (Medallion)                           │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │
│  │   Bronze    │  ──▶ │   Silver    │  ──▶ │    Gold     │              │
│  │  Raw JSON   │      │  Cleaned    │      │  Enriched   │              │
│  └─────────────┘      └─────────────┘      └─────────────┘              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ML Pipeline                                     │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │
│  │  Training   │  ──▶ │ Validation  │  ──▶ │  Registry   │              │
│  │  sklearn    │      │  Metrics    │      │  Versioned  │              │
│  └─────────────┘      └─────────────┘      └─────────────┘              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Deployment                                      │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │
│  │  FastAPI    │  ◀── │   Docker    │  ◀── │  CI/CD      │              │
│  │  (Render)   │      │  Container  │      │  GitHub     │              │
│  └──────┬──────┘      └─────────────┘      └─────────────┘              │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐      ┌─────────────┐                                   │
│  │  Streamlit  │      │    S3       │                                   │
│  │  Frontend   │      │  Artifacts  │                                   │
│  └─────────────┘      └─────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── api/                    # FastAPI REST service
│   ├── main.py            # API endpoints
│   ├── model_loader.py    # Model loading and inference
│   └── schemas.py         # Pydantic request/response models
│
├── app/                    # Streamlit web application
│   ├── main.py            # Entry point
│   ├── components/        # UI components
│   ├── services/          # API client and model services
│   └── utils/             # Formatting utilities
│
├── scripts/                # ML operations
│   ├── train_sklearn.py   # Training pipeline
│   ├── validate_model.py  # Quality gate validation
│   ├── model_versioning.py # Semantic versioning
│   ├── rollback_model.py  # Production rollback
│   ├── s3_storage.py      # Artifact backup to AWS S3
│   └── feature_transformer.py
│
├── models/                 # Serialized artifacts
│   ├── champion_model.pkl
│   ├── feature_transformer.pkl
│   └── model_metadata.json
│
├── data_lake/              # Medallion architecture
│   ├── bronze/            # Raw ingested data
│   ├── silver/            # Cleaned and validated
│   └── gold/              # Feature-engineered, ML-ready
│
├── tests/                  # Test suite
├── legacy/                 # Historical Spark/Airflow ETL (reference)
│
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile             # Container definition
├── docker-compose.yml     # Local development
└── render.yaml            # Cloud deployment config
```

---

## Technical Stack

| Layer | Technology |
|-------|------------|
| ML Framework | scikit-learn |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker |
| Cloud Hosting | Render (API), Streamlit Cloud (UI) |
| CI/CD | GitHub Actions |
| Artifact Storage | AWS S3 |
| Data Processing | pandas (production), PySpark (historical ETL) |

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/model/info` | Model version and metadata |
| POST | `/predict` | Single property prediction |
| POST | `/predict/batch` | Batch predictions |

### Example Request

```bash
curl -X POST https://bcn-housing-price-predictor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "size": 85,
    "rooms": 3,
    "bathrooms": 2,
    "neighborhood": "la Vila de Gràcia",
    "district": "Gràcia",
    "propertyType": "flat"
  }'
```

### Response

```json
{
  "predicted_price": 425000.00,
  "model_version": "1.0.0",
  "prediction_id": "abc123"
}
```

---

## Local Development

### Prerequisites

- Python 3.11+
- Docker (optional)

### Setup

```bash
git clone https://github.com/yourusername/barcelona-property-predictor.git
cd barcelona-property-predictor

python -m venv .venv
source .venv/bin/activate

pip install -r requirements-api.txt
```

### Run API

```bash
# Direct
uvicorn api.main:app --reload --port 8000

# Docker
docker compose up
```

### Run Streamlit

```bash
pip install -r requirements-streamlit.txt
streamlit run app/main.py
```

---

## MLOps Workflow

### Training

```bash
python scripts/train_sklearn.py
```

Outputs:
- Trained model with hyperparameter tuning
- Validation metrics and visualizations
- Versioned artifacts in `models/`

### Validation

```bash
python scripts/validate_model.py
```

Enforces quality gates:
- R² ≥ 0.70
- RMSE ≤ €150,000

### Versioning

```bash
python scripts/model_versioning.py info
python scripts/model_versioning.py bump minor
python scripts/model_versioning.py deploy --env production
```

### Rollback

```bash
python scripts/rollback_model.py list
python scripts/rollback_model.py rollback 1.0.0
```

### S3 Backup

```bash
python scripts/s3_storage.py upload
python scripts/s3_storage.py download --version 1.0.0
```

---

## CI/CD Pipeline

The GitHub Actions workflow executes on every push to `main`:

1. **Lint**: Code quality checks with Ruff and Black
2. **Test**: Unit tests with pytest
3. **Validate**: Model quality gate enforcement
4. **Build**: Docker image construction and verification
5. **Deploy**: Automatic deployment to Render
6. **Backup**: Artifact upload to S3

Pull requests trigger stages 1-3 for validation without deployment.

---

## Data Sources

| Source | Description | Update Frequency |
|--------|-------------|------------------|
| Idealista | Property listings with prices, sizes, features | Historical dataset |
| Open Data BCN | Household income index by neighborhood | Annual |
| Open Data BCN | Population density by neighborhood | Annual |

Data is joined on neighborhood identifiers using lookup tables that map between data source schemas.

---

## Model Performance

The Random Forest model was selected after comparative evaluation against Ridge Regression and Gradient Boosting. Cross-validation with 5 folds was used for hyperparameter tuning.

| Model | R² | RMSE |
|-------|-----|------|
| Random Forest | 0.906 | €116,745 |
| Gradient Boosting | 0.891 | €124,320 |
| Ridge Regression | 0.712 | €198,450 |

Feature importance analysis shows property size and neighborhood as the strongest predictors, followed by income index and number of rooms.

---

## Future Improvements

- SHAP explainability for individual predictions
- Automated data refresh pipeline with Idealista API integration
- Data drift detection and alerting
- A/B testing infrastructure for model comparison
- Prediction monitoring dashboard

---

## License

MIT License
