# 🏢 MLOps Sentiment Analysis - Company Reputation Monitoring

[![CI](https://github.com/pdimarcodev/sentiment-monitoring-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/pdimarcodev/sentiment-monitoring-mlops/actions/workflows/ci.yml)

An end-to-end MLOps solution for monitoring company reputation through automated sentiment analysis of social media content.

## 🎯 Project Overview

This project implements a comprehensive MLOps pipeline for **MachineInnovators Inc.** to monitor online reputation by analyzing sentiment in social media content. The system provides automated sentiment classification, continuous monitoring, and model retraining capabilities.

### Key Features

- **🤖 Advanced Sentiment Analysis**: HuggingFace RoBERTa model with 3-class classification
- **🚀 Production API**: FastAPI service with comprehensive endpoints
- **📊 Real-time Monitoring**: Grafana dashboards with Prometheus metrics
- **🔄 Automated Retraining**: Airflow DAG for model updates
- **🧪 Comprehensive Testing**: Unit, integration, and API tests
- **🐳 Container Ready**: Docker and Docker Compose deployment
- **⚡ CI/CD Pipeline**: Automated testing and deployment

## 📁 Project Structure

```
sentiment-monitoring-mlops/
├── src/sentiment_analyzer/          # Core application code
│   ├── model.py                     # Sentiment analysis model
│   └── api.py                       # FastAPI application
├── tests/                           # Test suite
│   ├── test_model.py               # Model tests
│   └── test_api.py                 # API tests
├── monitoring/                      # Monitoring configuration
│   ├── grafana/                    # Grafana dashboards
│   ├── airflow/                    # Airflow DAGs
│   └── prometheus.yml              # Prometheus config
├── .github/workflows/              # CI/CD pipelines
│   ├── ci.yml                      # Continuous Integration
│   └── cd.yml                      # Continuous Deployment
├── docker-compose.yml              # Multi-service deployment
├── Dockerfile                      # Container definition
├── app.py                          # HuggingFace Spaces app
└── main.py                         # Application entry point
```

## 🚀 Quick Start

### 1. Local Development

```bash
# Clone repository
git clone https://github.com/pdimarcodev/sentiment-monitoring-mlops.git
cd sentiment-monitoring-mlops

# Install dependencies
pip install -r requirements.txt

# Start the API server
python main.py
```

The API will be available at `http://localhost:8000`

### 2. Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Access services:
# - API: http://localhost:8000
# - Grafana: http://localhost:3000 (admin/admin123)
# - Prometheus: http://localhost:9090
```

### 3. Google Colab Demo

Open the provided notebook: `Sentiment_Analysis_MLOps_Demo.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pdimarcodev/sentiment-monitoring-mlops/blob/main/Sentiment_Analysis_MLOps_Demo.ipynb)

## 📖 API Documentation

### Endpoints

- **POST `/predict`** - Single text sentiment analysis
- **POST `/predict/batch`** - Batch text processing
- **POST `/predict/dataset`** - Process text arrays with optional validation labels
- **POST `/predict/csv`** - Upload and process CSV files
- **POST `/predict/json`** - Upload and process JSON files
- **GET `/health`** - Service health check
- **GET `/metrics`** - Prometheus metrics
- **GET `/model/info`** - Model information

### Example Usage

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "I love this product!"}
)
print(response.json())
# Output: {"sentiment": "positive", "confidence": 0.9234, ...}

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["Great service!", "Poor quality"]}
)

# Dataset processing with validation
response = requests.post(
    "http://localhost:8000/predict/dataset",
    json={
        "texts": ["Amazing product!", "Terrible service"],
        "labels": ["positive", "negative"]  # Optional for validation
    }
)

# CSV file upload
with open("dataset.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/csv",
        files={"file": f},
        data={
            "text_column": "text",
            "label_column": "sentiment",  # Optional
            "export_format": "json"
        }
    )
```

## 🔧 Configuration

### Environment Variables

```env
# Model Configuration
MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
MODEL_CACHE_DIR=/tmp/model_cache

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### GitHub Secrets (for CI/CD)

- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password
- `HF_TOKEN`: HuggingFace access token

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_model.py -v    # Model tests
pytest tests/test_api.py -v      # API tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Monitoring and Metrics

### Grafana Dashboards

Access Grafana at `http://localhost:3000` with credentials `admin/admin123`:

- **API Performance**: Request rate, response times, error rates
- **Sentiment Distribution**: Real-time sentiment classification trends
- **Model Performance**: Confidence scores and prediction accuracy

### Key Metrics

- `sentiment_requests_total`: Total API requests by endpoint
- `sentiment_request_duration_seconds`: Request processing time
- `sentiment_predictions_total`: Predictions by sentiment class

## 🔄 Model Retraining

The Airflow DAG (`monitoring/airflow/sentiment_retraining_dag.py`) handles:

1. **Performance Monitoring**: Check model accuracy metrics
2. **Data Collection**: Gather new labeled data
3. **Model Training**: Retrain with updated dataset
4. **Validation**: Compare new vs. current model
5. **Deployment**: Auto-deploy if performance improves

## 🚢 Deployment Options

### 1. Docker Hub

```bash
# Build and push
docker build -t sentiment-analyzer .
docker tag sentiment-analyzer pdimarcodev/sentiment-analyzer
docker push pdimarcodev/sentiment-analyzer
```

### 2. HuggingFace Spaces

Deploy the included `app.py` to HuggingFace Spaces for a public web interface.

### 3. Cloud Platforms

The containerized application can be deployed to:
- **AWS ECS/EKS**
- **Google Cloud Run**
- **Azure Container Instances**

## 🛡️ Security

- Non-root container user
- Input validation and sanitization
- Rate limiting (configurable)
- Security scanning in CI/CD pipeline
- Secrets management for sensitive data

## 🔧 Development

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Run test suite: `pytest tests/ -v`
4. Commit and push: `git push origin feature/new-feature`
5. Create pull request

### Code Quality

- **Linting**: flake8 for code style
- **Testing**: pytest with >90% coverage
- **Security**: bandit security scanning
- **Dependencies**: safety vulnerability checks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Context

This project was developed as part of the MLOps course requirements, demonstrating:

- **Phase 1**: Model implementation with HuggingFace
- **Phase 2**: CI/CD pipeline development
- **Phase 3**: Monitoring and retraining automation

**Institution**: ProfessionAI MLOps Course  
**Project**: Company Online Reputation Monitoring System  
**Technologies**: HuggingFace, FastAPI, Grafana, Airflow, Docker, GitHub Actions

---

**📞 Support**: Create an issue for questions or bug reports  
**⭐ Star**: If this project helps you, please star it!  
**🔗 Links**: [Documentation](docs/) | [API Docs](http://localhost:8000/docs) | [Monitoring](http://localhost:3000)