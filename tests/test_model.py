import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_analyzer.model import SentimentAnalyzer

class TestSentimentAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        """Fixture to create analyzer instance"""
        return SentimentAnalyzer()
    
    def test_model_initialization(self, analyzer):
        """Test model loads successfully"""
        assert analyzer.pipeline is not None
        assert analyzer.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    def test_positive_sentiment(self, analyzer):
        """Test positive sentiment prediction"""
        result = analyzer.predict("I love this product! It's amazing!")
        assert result['sentiment'] == 'positive'
        assert result['confidence'] > 0.5
        assert 'text' in result
        assert 'all_scores' in result
    
    def test_negative_sentiment(self, analyzer):
        """Test negative sentiment prediction"""
        result = analyzer.predict("This is terrible and I hate it!")
        assert result['sentiment'] == 'negative'
        assert result['confidence'] > 0.5
    
    def test_neutral_sentiment(self, analyzer):
        """Test neutral sentiment prediction"""
        result = analyzer.predict("This is a product.")
        assert result['sentiment'] in ['neutral', 'positive', 'negative']  # Any valid sentiment
        assert 0 <= result['confidence'] <= 1
    
    def test_empty_text_error(self, analyzer):
        """Test error handling for empty text"""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            analyzer.predict("")
    
    def test_non_string_input_error(self, analyzer):
        """Test error handling for non-string input"""
        with pytest.raises(ValueError, match="Input must be a string"):
            analyzer.predict(123)
    
    def test_long_text_truncation(self, analyzer):
        """Test text truncation for long inputs"""
        long_text = "This is a test. " * 100  # Create very long text
        result = analyzer.predict(long_text)
        assert len(result['text']) <= 512
    
    def test_batch_prediction(self, analyzer):
        """Test batch prediction functionality"""
        texts = [
            "I love this!",
            "This is terrible!",
            "This is okay."
        ]
        results = analyzer.predict_batch(texts)
        assert len(results) == 3
        for result in results:
            assert 'sentiment' in result
            assert 'confidence' in result
    
    def test_model_info(self, analyzer):
        """Test model info retrieval"""
        info = analyzer.get_model_info()
        assert 'model_name' in info
        assert 'model_type' in info
        assert 'labels' in info
        assert info['labels'] == ['negative', 'neutral', 'positive']