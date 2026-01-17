import pytest
import sys
import os
import tempfile
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_analyzer.model import SentimentAnalyzer
from src.sentiment_analyzer.dataset_processor import DatasetProcessor

class TestDatasetProcessor:
    
    @pytest.fixture
    def processor(self):
        """Fixture to create dataset processor"""
        analyzer = SentimentAnalyzer()
        return DatasetProcessor(analyzer)
    
    def test_process_text_list(self, processor):
        """Test processing list of texts"""
        texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay."
        ]
        
        result = processor.process_text_list(texts)
        
        assert result['total_processed'] == 3
        assert len(result['results']) == 3
        assert 'statistics' in result
        assert result['has_true_labels'] == False
        
        # Check all results have required fields
        for res in result['results']:
            assert 'sentiment' in res
            assert 'confidence' in res
            assert 'text' in res
    
    def test_process_text_list_with_labels(self, processor):
        """Test processing list of texts with validation labels"""
        texts = ["Great product!", "Bad service"]
        labels = ["positive", "negative"]
        
        result = processor.process_text_list(texts, labels)
        
        assert result['has_true_labels'] == True
        assert 'validation_metrics' in result['statistics']
        assert 'accuracy' in result['statistics']['validation_metrics']
    
    def test_process_csv_file(self, processor):
        """Test processing CSV file"""
        # Create temporary CSV file
        csv_data = """text,sentiment
"I love this!",positive
"This is bad.",negative
"It's average.",neutral"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            f.flush()
            
            result = processor.process_csv_file(f.name, 'text', 'sentiment')
        
        os.unlink(f.name)
        
        assert result['total_processed'] == 3
        assert result['has_true_labels'] == True
        assert 'validation_metrics' in result['statistics']
    
    def test_process_json_file(self, processor):
        """Test processing JSON file"""
        # Create temporary JSON file
        json_data = [
            {"text": "Great service!", "label": "positive"},
            {"text": "Poor quality.", "label": "negative"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            f.flush()
            
            result = processor.process_json_file(f.name, 'text', 'label')
        
        os.unlink(f.name)
        
        assert result['total_processed'] == 2
        assert result['has_true_labels'] == True
    
    def test_statistics_calculation(self, processor):
        """Test statistics calculation"""
        texts = [
            "I love this!",
            "Amazing product!",
            "This is terrible!",
            "It's okay."
        ]
        
        result = processor.process_text_list(texts)
        stats = result['statistics']
        
        assert 'sentiment_distribution' in stats
        assert 'sentiment_percentages' in stats
        assert 'average_confidence' in stats
        assert 'total_texts' in stats
        assert stats['total_texts'] == 4
        
        # Check percentages sum to 100
        total_percentage = sum(stats['sentiment_percentages'].values())
        assert abs(total_percentage - 100.0) < 0.1
    
    def test_export_results_csv(self, processor):
        """Test exporting results to CSV"""
        texts = ["Great!", "Bad!"]
        result = processor.process_text_list(texts)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = processor.export_results(result, f.name, 'csv')
            
            # Verify file exists and has content
            assert os.path.exists(output_path)
            
            # Read back and verify structure
            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert 'text' in df.columns
            assert 'predicted_sentiment' in df.columns
            assert 'confidence' in df.columns
        
        os.unlink(f.name)
    
    def test_export_results_json(self, processor):
        """Test exporting results to JSON"""
        texts = ["Good!", "Bad!"]
        result = processor.process_text_list(texts)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = processor.export_results(result, f.name, 'json')
            
            # Verify file exists and has content
            assert os.path.exists(output_path)
            
            # Read back and verify structure
            with open(output_path, 'r') as read_f:
                data = json.load(read_f)
                assert 'results' in data
                assert 'statistics' in data
                assert len(data['results']) == 2
        
        os.unlink(f.name)
    
    def test_error_handling_invalid_column(self, processor):
        """Test error handling for invalid column names"""
        csv_data = "content,label\nTest text,positive"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            f.flush()
            
            with pytest.raises(ValueError, match="Text column 'text' not found"):
                processor.process_csv_file(f.name, 'text', 'label')
        
        os.unlink(f.name)
    
    def test_error_handling_empty_texts(self, processor):
        """Test error handling for empty text list"""
        with pytest.raises(ValueError, match="No valid texts found"):
            processor.process_text_list([])
    
    def test_csv_string_processing(self, processor):
        """Test processing CSV content from string"""
        csv_content = """text,sentiment
"Good product!",positive
"Poor service.",negative"""
        
        result = processor.process_csv_string(csv_content, 'text', 'sentiment')
        
        assert result['total_processed'] == 2
        assert result['has_true_labels'] == True