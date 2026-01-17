from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from contextlib import asynccontextmanager
import time
import logging
import tempfile
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, FileResponse

from .model import SentimentAnalyzer
from .dataset_processor import DatasetProcessor

# Prometheus metrics
REQUEST_COUNT = Counter('sentiment_requests_total', 'Total sentiment analysis requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('sentiment_request_duration_seconds', 'Request duration')
PREDICTION_COUNT = Counter('sentiment_predictions_total', 'Total predictions by sentiment', ['sentiment'])

# Initialize the sentiment analyzer and dataset processor at module level
sentiment_analyzer = SentimentAnalyzer()
dataset_processor = DatasetProcessor(sentiment_analyzer)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup: model already initialized at module level
    logging.info("Sentiment Analysis API started")
    yield
    # Shutdown
    logging.info("Sentiment Analysis API shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Sentiment Analysis API",
    description="MLOps Sentiment Analysis Service for Company Reputation Monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Text to analyze")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to analyze")

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    all_scores: Dict[str, float]

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processed: int

class DatasetProcessingResponse(BaseModel):
    total_processed: int
    results: List[SentimentResponse]
    statistics: Dict
    has_true_labels: bool

class DatasetRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    labels: Optional[List[str]] = Field(None, description="Optional true labels for validation")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sentiment-analysis", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_info": sentiment_analyzer.get_model_info(),
        "timestamp": time.time()
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    """
    Predict sentiment for a single text
    """
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Track request metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
        
        # Make prediction
        result = sentiment_analyzer.predict(request.text)
        
        # Track prediction metrics
        PREDICTION_COUNT.labels(sentiment=result['sentiment']).inc()
        
        # Track request duration
        REQUEST_DURATION.observe(time.time() - start_time)
        
        return SentimentResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(request: BatchRequest):
    """
    Predict sentiment for multiple texts
    """
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Track request metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch").inc()
        
        # Make predictions
        results = sentiment_analyzer.predict_batch(request.texts)
        
        # Track prediction metrics
        for result in results:
            PREDICTION_COUNT.labels(sentiment=result['sentiment']).inc()
        
        # Track request duration
        REQUEST_DURATION.observe(time.time() - start_time)
        
        response_results = [SentimentResponse(**result) for result in results]
        
        return BatchSentimentResponse(
            results=response_results,
            total_processed=len(results)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return sentiment_analyzer.get_model_info()

@app.post("/predict/dataset", response_model=DatasetProcessingResponse)
async def predict_dataset(request: DatasetRequest):
    """
    Process a dataset with texts and optional labels
    """
    if dataset_processor is None:
        raise HTTPException(status_code=503, detail="Dataset processor not loaded")
    
    start_time = time.time()
    
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/dataset").inc()
        
        # Process the dataset
        result = dataset_processor.process_text_list(request.texts, request.labels)
        
        # Track prediction metrics
        for res in result['results']:
            PREDICTION_COUNT.labels(sentiment=res['sentiment']).inc()
        
        REQUEST_DURATION.observe(time.time() - start_time)
        
        # Convert results to response format
        response_results = [SentimentResponse(**res) for res in result['results']]
        
        return DatasetProcessingResponse(
            total_processed=result['total_processed'],
            results=response_results,
            statistics=result['statistics'],
            has_true_labels=result['has_true_labels']
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Dataset processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/csv")
async def predict_csv_file(
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    label_column: Optional[str] = Form(None),
    export_format: str = Form("json")
):
    """
    Upload and process a CSV file for sentiment analysis
    
    Parameters:
    - file: CSV file to upload
    - text_column: Name of the column containing text (default: 'text')
    - label_column: Name of the column containing true labels (optional)
    - export_format: Format for results export ('json' or 'csv')
    """
    if dataset_processor is None:
        raise HTTPException(status_code=503, detail="Dataset processor not loaded")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    start_time = time.time()
    
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/csv").inc()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the CSV file
            result = dataset_processor.process_csv_file(
                temp_file.name, 
                text_column, 
                label_column
            )
        
        # Track prediction metrics
        for res in result['results']:
            PREDICTION_COUNT.labels(sentiment=res['sentiment']).inc()
        
        REQUEST_DURATION.observe(time.time() - start_time)
        
        # Export results to temporary file
        export_path = tempfile.mktemp(suffix=f'.{export_format}')
        dataset_processor.export_results(result, export_path, export_format)
        
        # Clean up uploaded file
        os.unlink(temp_file.name)
        
        # Return the processed file
        media_type = "application/json" if export_format == "json" else "text/csv"
        return FileResponse(
            export_path,
            media_type=media_type,
            filename=f"sentiment_results.{export_format}",
            background=None  # File will be cleaned up after response
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"CSV processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/json")
async def predict_json_file(
    file: UploadFile = File(...),
    text_field: str = Form("text"),
    label_field: Optional[str] = Form(None),
    export_format: str = Form("json")
):
    """
    Upload and process a JSON file for sentiment analysis
    
    Parameters:
    - file: JSON file to upload
    - text_field: Name of the field containing text (default: 'text')
    - label_field: Name of the field containing true labels (optional)
    - export_format: Format for results export ('json' or 'csv')
    """
    if dataset_processor is None:
        raise HTTPException(status_code=503, detail="Dataset processor not loaded")
    
    # Validate file type
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="File must be a JSON file")
    
    start_time = time.time()
    
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/json").inc()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.json', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the JSON file
            result = dataset_processor.process_json_file(
                temp_file.name, 
                text_field, 
                label_field
            )
        
        # Track prediction metrics
        for res in result['results']:
            PREDICTION_COUNT.labels(sentiment=res['sentiment']).inc()
        
        REQUEST_DURATION.observe(time.time() - start_time)
        
        # Export results to temporary file
        export_path = tempfile.mktemp(suffix=f'.{export_format}')
        dataset_processor.export_results(result, export_path, export_format)
        
        # Clean up uploaded file
        os.unlink(temp_file.name)
        
        # Return the processed file
        media_type = "application/json" if export_format == "json" else "text/csv"
        return FileResponse(
            export_path,
            media_type=media_type,
            filename=f"sentiment_results.{export_format}",
            background=None
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"JSON processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)