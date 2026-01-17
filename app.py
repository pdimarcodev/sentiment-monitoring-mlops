import gradio as gr
import requests
import json
from typing import List

# This would be your deployed API endpoint
API_BASE_URL = "http://localhost:8000"  # Change to your actual deployed URL

def analyze_sentiment(text: str) -> dict:
    """Call the sentiment analysis API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def batch_analyze_sentiment(texts: str) -> dict:
    """Call the batch sentiment analysis API"""
    try:
        # Split texts by newlines and clean them
        text_list = [text.strip() for text in texts.split('\n') if text.strip()]
        
        if not text_list:
            return {"error": "No valid texts provided"}
        
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"texts": text_list},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def format_single_result(result: dict) -> str:
    """Format single prediction result for display"""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    sentiment = result['sentiment']
    confidence = result['confidence']
    
    # Add emoji based on sentiment
    emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
    
    output = f"{emoji.get(sentiment, '')} **Sentiment**: {sentiment.upper()}\n"
    output += f"**Confidence**: {confidence:.2%}\n\n"
    output += "**All Scores**:\n"
    
    for label, score in result['all_scores'].items():
        bar = "█" * int(score * 20)  # Simple bar visualization
        output += f"- {label}: {score:.2%} {bar}\n"
    
    return output

def format_batch_results(results: dict) -> str:
    """Format batch prediction results for display"""
    if "error" in results:
        return f"❌ Error: {results['error']}"
    
    output = f"**Processed {results['total_processed']} texts**\n\n"
    
    # Summary statistics
    sentiments = [r['sentiment'] for r in results['results']]
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    
    output += "**Summary**:\n"
    for sentiment, count in sentiment_counts.items():
        percentage = count / len(sentiments) * 100
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        output += f"{emoji.get(sentiment, '')} {sentiment}: {count} ({percentage:.1f}%)\n"
    
    output += "\n**Individual Results**:\n"
    for i, result in enumerate(results['results'], 1):
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        sentiment_emoji = emoji.get(result['sentiment'], '')
        output += f"{i}. {sentiment_emoji} {result['sentiment']} ({result['confidence']:.2%}): \"{result['text'][:50]}...\"\n"
    
    return output

# Create Gradio interface
with gr.Blocks(title="Sentiment Analysis - Company Reputation Monitor") as demo:
    gr.Markdown("""
    # 🏢 Company Reputation Monitoring System
    
    This MLOps-powered sentiment analysis tool helps monitor online reputation by analyzing social media content.
    Built with HuggingFace RoBERTa, FastAPI, and comprehensive monitoring.
    """)
    
    with gr.Tab("Single Text Analysis"):
        with gr.Row():
            with gr.Column(scale=2):
                single_input = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="Type your text here... (e.g., 'I love this product!')",
                    lines=3
                )
                single_button = gr.Button("Analyze Sentiment", variant="primary")
            
            with gr.Column(scale=3):
                single_output = gr.Markdown(label="Results")
        
        single_button.click(
            fn=lambda text: format_single_result(analyze_sentiment(text)),
            inputs=single_input,
            outputs=single_output
        )
        
        # Example inputs
        gr.Examples(
            examples=[
                ["I absolutely love this product! Best purchase ever!"],
                ["This service is terrible and I hate it."],
                ["The product is okay, nothing special."],
                ["Amazing customer service and fast delivery! Highly recommend!"],
                ["Poor quality and overpriced. Very disappointed."]
            ],
            inputs=single_input
        )
    
    with gr.Tab("Batch Analysis"):
        with gr.Row():
            with gr.Column(scale=2):
                batch_input = gr.Textbox(
                    label="Enter multiple texts (one per line)",
                    placeholder="Enter each text on a new line...",
                    lines=8
                )
                batch_button = gr.Button("Analyze All", variant="primary")
            
            with gr.Column(scale=3):
                batch_output = gr.Markdown(label="Results")
        
        batch_button.click(
            fn=lambda texts: format_batch_results(batch_analyze_sentiment(texts)),
            inputs=batch_input,
            outputs=batch_output
        )
        
        # Example batch input
        gr.Examples(
            examples=[
                ["""I love this company's products!
Their customer service is amazing.
The delivery was fast and efficient.
Product quality could be better.
Overall satisfied with my purchase."""]
            ],
            inputs=batch_input
        )
    
    with gr.Tab("API Information"):
        gr.Markdown("""
        ## 🔗 API Endpoints
        
        This interface connects to a FastAPI backend with the following endpoints:
        
        - `POST /predict` - Single text sentiment analysis
        - `POST /predict/batch` - Batch text sentiment analysis  
        - `GET /health` - Service health check
        - `GET /metrics` - Prometheus metrics
        
        ## 📊 Model Information
        
        - **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
        - **Labels**: Positive, Negative, Neutral
        - **Framework**: HuggingFace Transformers
        - **Monitoring**: Grafana + Prometheus
        - **CI/CD**: GitHub Actions
        
        ## 🏗️ MLOps Pipeline
        
        This system includes:
        - Automated testing and deployment
        - Real-time monitoring and alerting
        - Model performance tracking
        - Automated retraining capabilities
        """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )