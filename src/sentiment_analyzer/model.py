from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from typing import Dict, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analysis model using RoBERTa from HuggingFace
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Initialize the sentiment analyzer with the specified model"""
        self.model_name = model_name
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.pipeline = pipeline(
                "sentiment-analysis", 
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for a single text
        Returns: dict with label and confidence score
        """
        if not self.pipeline:
            raise ValueError("Model not loaded")
        
        # Clean and validate input
        text = self._preprocess_text(text)
        
        try:
            # Get predictions for all labels
            results = self.pipeline(text)[0]
            
            # Find the highest scoring label
            best_result = max(results, key=lambda x: x['score'])
            
            # Map labels to human-readable format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            mapped_label = label_mapping.get(best_result['label'], best_result['label'])
            
            return {
                'text': text,
                'sentiment': mapped_label,
                'confidence': round(best_result['score'], 4),
                'all_scores': {
                    label_mapping.get(result['label'], result['label']): round(result['score'], 4) 
                    for result in results
                }
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for a batch of texts
        """
        return [self.predict(text) for text in texts]
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Basic cleaning
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty")
        
        # Truncate if too long (RoBERTa has token limits)
        if len(text) > 512:
            text = text[:512]
            logger.warning("Text truncated to 512 characters")
        
        return text
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_type': 'sentiment-analysis',
            'labels': ['negative', 'neutral', 'positive']
        }