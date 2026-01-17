import pandas as pd
import json
import csv
from typing import List, Dict, Union, Optional
from io import StringIO
import logging

logger = logging.getLogger(__name__)

class DatasetProcessor:
    """
    Handle various dataset formats for sentiment analysis
    """
    
    def __init__(self, sentiment_analyzer):
        self.analyzer = sentiment_analyzer
    
    def process_csv_file(self, file_path: str, text_column: str = 'text', 
                        label_column: Optional[str] = None) -> Dict:
        """
        Process CSV file with text data
        
        Args:
            file_path: Path to CSV file
            text_column: Name of the text column
            label_column: Name of the label column (if available for validation)
        
        Returns:
            Dict with results and statistics
        """
        try:
            df = pd.read_csv(file_path)
            return self._process_dataframe(df, text_column, label_column)
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise
    
    def process_json_file(self, file_path: str, text_field: str = 'text',
                         label_field: Optional[str] = None) -> Dict:
        """
        Process JSON file with text data
        
        Args:
            file_path: Path to JSON file
            text_field: Name of the text field
            label_field: Name of the label field (if available)
        
        Returns:
            Dict with results and statistics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            return self._process_dataframe(df, text_field, label_field)
        except Exception as e:
            logger.error(f"Error processing JSON file: {str(e)}")
            raise
    
    def process_text_list(self, texts: List[str], labels: Optional[List[str]] = None) -> Dict:
        """
        Process a list of texts
        
        Args:
            texts: List of text strings
            labels: Optional list of true labels for validation
        
        Returns:
            Dict with results and statistics
        """
        try:
            # Create dataframe from lists
            data = {'text': texts}
            if labels:
                data['label'] = labels
            
            df = pd.DataFrame(data)
            return self._process_dataframe(df, 'text', 'label' if labels else None)
        except Exception as e:
            logger.error(f"Error processing text list: {str(e)}")
            raise
    
    def process_csv_string(self, csv_content: str, text_column: str = 'text',
                          label_column: Optional[str] = None) -> Dict:
        """
        Process CSV content from string
        
        Args:
            csv_content: CSV content as string
            text_column: Name of the text column
            label_column: Name of the label column (if available)
        
        Returns:
            Dict with results and statistics
        """
        try:
            df = pd.read_csv(StringIO(csv_content))
            return self._process_dataframe(df, text_column, label_column)
        except Exception as e:
            logger.error(f"Error processing CSV string: {str(e)}")
            raise
    
    def _process_dataframe(self, df: pd.DataFrame, text_column: str,
                          label_column: Optional[str] = None) -> Dict:
        """
        Internal method to process pandas DataFrame
        """
        # Validate columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        if label_column and label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
        
        # Clean and prepare texts
        texts = df[text_column].dropna().astype(str).tolist()
        true_labels = None
        if label_column:
            true_labels = df[label_column].dropna().astype(str).tolist()
        
        if not texts:
            raise ValueError("No valid texts found in dataset")
        
        logger.info(f"Processing {len(texts)} texts...")
        
        # Make predictions using batch processing
        results = self.analyzer.predict_batch(texts)
        
        # Calculate statistics
        stats = self._calculate_statistics(results, true_labels)
        
        return {
            'total_processed': len(results),
            'results': results,
            'statistics': stats,
            'has_true_labels': true_labels is not None
        }
    
    def _calculate_statistics(self, results: List[Dict], 
                            true_labels: Optional[List[str]] = None) -> Dict:
        """
        Calculate statistics from prediction results
        """
        # Basic sentiment distribution
        sentiments = [r['sentiment'] for r in results]
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        
        total = len(sentiments)
        sentiment_percentages = {
            k: round(v / total * 100, 2) for k, v in sentiment_counts.items()
        }
        
        # Average confidence scores
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        stats = {
            'sentiment_distribution': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'average_confidence': round(avg_confidence, 4),
            'total_texts': total
        }
        
        # If true labels are available, calculate accuracy metrics
        if true_labels and len(true_labels) == len(results):
            predicted_labels = [r['sentiment'] for r in results]
            accuracy = sum(1 for p, t in zip(predicted_labels, true_labels) 
                          if p == t) / len(true_labels)
            
            stats['validation_metrics'] = {
                'accuracy': round(accuracy, 4),
                'correct_predictions': sum(1 for p, t in zip(predicted_labels, true_labels) if p == t),
                'total_comparisons': len(true_labels)
            }
        
        return stats
    
    def export_results(self, results_dict: Dict, output_path: str, 
                      format: str = 'csv') -> str:
        """
        Export results to file
        
        Args:
            results_dict: Results from processing
            output_path: Path for output file
            format: Output format ('csv', 'json')
        
        Returns:
            Path to saved file
        """
        try:
            results = results_dict['results']
            
            if format.lower() == 'csv':
                # Create DataFrame from results
                df = pd.DataFrame([
                    {
                        'text': r['text'],
                        'predicted_sentiment': r['sentiment'],
                        'confidence': r['confidence'],
                        'positive_score': r['all_scores']['positive'],
                        'negative_score': r['all_scores']['negative'],
                        'neutral_score': r['all_scores']['neutral']
                    }
                    for r in results
                ])
                df.to_csv(output_path, index=False)
            
            elif format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Results exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise