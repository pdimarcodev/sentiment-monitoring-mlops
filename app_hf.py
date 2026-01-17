import gradio as gr
from transformers import pipeline
from typing import List

# Initialize sentiment analysis pipeline
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)
print("Model loaded successfully!")

# Label mapping
label_map = {
    'LABEL_0': 'negative',
    'LABEL_1': 'neutral',
    'LABEL_2': 'positive'
}

def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of a single text"""
    if not text or not text.strip():
        return "❌ Error: Please enter some text to analyze"

    try:
        # Get predictions
        results = sentiment_pipeline(text[:512])[0]  # Truncate to 512 chars

        # Find best prediction
        best_result = max(results, key=lambda x: x['score'])
        sentiment = label_map.get(best_result['label'], best_result['label'])
        confidence = best_result['score']

        # Format output
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        sentiment_emoji = emoji.get(sentiment, '')

        output = f"{sentiment_emoji} **Sentiment**: {sentiment.upper()}\n"
        output += f"**Confidence**: {confidence:.2%}\n\n"
        output += "**All Scores**:\n"

        for result in results:
            label = label_map.get(result['label'], result['label'])
            score = result['score']
            bar = "█" * int(score * 20)  # Simple bar visualization
            output += f"- {label}: {score:.2%} {bar}\n"

        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"

def batch_analyze_sentiment(texts: str) -> str:
    """Analyze sentiment of multiple texts"""
    if not texts or not texts.strip():
        return "❌ Error: Please enter some texts to analyze (one per line)"

    try:
        # Split texts by newlines
        text_list = [text.strip() for text in texts.split('\n') if text.strip()]

        if not text_list:
            return "❌ Error: No valid texts provided"

        # Get predictions for all texts
        all_predictions = []
        for text in text_list:
            results = sentiment_pipeline(text[:512])[0]
            best_result = max(results, key=lambda x: x['score'])
            all_predictions.append({
                'text': text,
                'sentiment': label_map.get(best_result['label'], best_result['label']),
                'confidence': best_result['score']
            })

        # Summary statistics
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for pred in all_predictions:
            sentiment_counts[pred['sentiment']] += 1

        # Format output
        output = f"**Processed {len(all_predictions)} texts**\n\n"
        output += "**Summary**:\n"

        for sentiment, count in sentiment_counts.items():
            percentage = count / len(all_predictions) * 100
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
            output += f"{emoji[sentiment]} {sentiment}: {count} ({percentage:.1f}%)\n"

        output += "\n**Individual Results**:\n"
        for i, pred in enumerate(all_predictions, 1):
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
            sentiment_emoji = emoji.get(pred['sentiment'], '')
            text_preview = pred['text'][:60] + "..." if len(pred['text']) > 60 else pred['text']
            output += f"{i}. {sentiment_emoji} {pred['sentiment']} ({pred['confidence']:.2%}): \"{text_preview}\"\n"

        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Sentiment Analysis - Company Reputation Monitor") as demo:
    gr.Markdown("""
    # 🏢 Company Reputation Monitoring System

    This MLOps-powered sentiment analysis tool helps monitor online reputation by analyzing social media content.
    Built with HuggingFace RoBERTa, FastAPI, and comprehensive monitoring.

    **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
    **GitHub**: [pdimarcodev/sentiment-monitoring-mlops](https://github.com/pdimarcodev/sentiment-monitoring-mlops)
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
            fn=analyze_sentiment,
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
            fn=batch_analyze_sentiment,
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

    with gr.Tab("About"):
        gr.Markdown("""
        ## 📊 Model Information

        - **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
        - **Labels**: Positive, Negative, Neutral
        - **Framework**: HuggingFace Transformers
        - **Evaluation**: Tested on Tweet Eval public dataset

        ## 🏗️ MLOps Pipeline

        This system includes:
        - ✅ Model evaluation on public dataset (accuracy, precision, recall, F1)
        - ✅ FastAPI service with comprehensive endpoints
        - ✅ Automated testing and CI/CD (GitHub Actions)
        - ✅ Real-time monitoring (Grafana + Prometheus)
        - ✅ Docker containerization
        - ✅ Model retraining capabilities

        ## 🔗 Links

        - [GitHub Repository](https://github.com/pdimarcodev/sentiment-monitoring-mlops)
        - [CI/CD Pipeline](https://github.com/pdimarcodev/sentiment-monitoring-mlops/actions)
        - [Documentation](https://github.com/pdimarcodev/sentiment-monitoring-mlops#readme)

        ---

        **Made with ❤️ for MLOps Course - ProfessionAI**
        """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
