import uvicorn
from src.sentiment_analyzer.api import app

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run(
        "src.sentiment_analyzer.api:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )