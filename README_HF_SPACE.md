---
title: Sentiment Analysis - Company Reputation Monitor
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.7.1
app_file: app_hf.py
pinned: false
---

# 🏢 Company Reputation Monitoring System

MLOps-powered sentiment analysis tool for monitoring online reputation through social media content analysis.

## 🎯 Features

- **Real-time Sentiment Analysis**: Classify text as positive, negative, or neutral
- **Batch Processing**: Analyze multiple texts at once
- **Confidence Scores**: Get detailed probability scores for each sentiment
- **Production-Ready**: Part of a complete MLOps pipeline

## 🤖 Model

- **Name**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Type**: RoBERTa fine-tuned for sentiment analysis
- **Classes**: Negative, Neutral, Positive
- **Evaluation**: Tested on Tweet Eval public dataset

## 📊 Performance Metrics

The model has been evaluated on the Tweet Eval sentiment dataset with:
- Accuracy, Precision, Recall, F1-score metrics
- Confusion matrix analysis
- Sample-level validation

## 🏗️ Complete MLOps Pipeline

This Space is part of a comprehensive MLOps project that includes:

- ✅ **Model Evaluation**: Public dataset testing with metrics
- ✅ **FastAPI Service**: RESTful API for predictions
- ✅ **CI/CD Pipeline**: GitHub Actions for automated testing
- ✅ **Monitoring**: Grafana + Prometheus metrics
- ✅ **Containerization**: Docker deployment
- ✅ **Testing**: Comprehensive unit and integration tests

## 🔗 Links

- **GitHub Repository**: [pdimarcodev/sentiment-monitoring-mlops](https://github.com/pdimarcodev/sentiment-monitoring-mlops)
- **CI/CD Pipeline**: [GitHub Actions](https://github.com/pdimarcodev/sentiment-monitoring-mlops/actions)
- **Documentation**: [README](https://github.com/pdimarcodev/sentiment-monitoring-mlops#readme)

## 💡 Usage

### Single Text Analysis
Enter any text and get instant sentiment classification with confidence scores.

### Batch Analysis
Paste multiple texts (one per line) to analyze them all at once with summary statistics.

## 📖 About

This project was developed as part of the MLOps course requirements for ProfessionAI, demonstrating:
- End-to-end MLOps best practices
- Model deployment and monitoring
- Automated testing and CI/CD
- Production-ready infrastructure

---

**Made with ❤️ for MLOps Course - ProfessionAI**
