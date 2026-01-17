#!/usr/bin/env python3
"""
Dataset Processing Examples for Sentiment Analysis

This script demonstrates various ways to pass datasets for sentiment analysis:
1. Direct text lists
2. CSV files
3. JSON files
4. API endpoints
"""

import requests
import json
import pandas as pd
import sys
import os

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sentiment_analyzer.model import SentimentAnalyzer
from src.sentiment_analyzer.dataset_processor import DatasetProcessor

def example_1_text_list():
    """Example 1: Process a list of texts directly"""
    print("🔥 Example 1: Processing Text Lists")
    print("=" * 50)
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    processor = DatasetProcessor(analyzer)
    
    # Sample texts (could come from any source)
    texts = [
        "I love this company's innovative approach!",
        "Poor customer service experience.",
        "The product quality is acceptable.",
        "Amazing features and great value!",
        "Disappointed with the recent update."
    ]
    
    # Optional: true labels for validation
    true_labels = ["positive", "negative", "neutral", "positive", "negative"]
    
    # Process the dataset
    results = processor.process_text_list(texts, true_labels)
    
    print(f"📊 Processed {results['total_processed']} texts")
    print(f"📈 Statistics: {results['statistics']}")
    print(f"✅ Has validation labels: {results['has_true_labels']}")
    
    # Show individual results
    print("\\n🔍 Individual Results:")
    for i, result in enumerate(results['results'][:3], 1):  # Show first 3
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        sentiment_emoji = emoji.get(result['sentiment'], '')
        print(f"{i}. {sentiment_emoji} {result['sentiment'].upper()} ({result['confidence']:.2%}): \"{result['text'][:50]}...\"")
    
    print("\\n" + "=" * 50 + "\\n")

def example_2_csv_file():
    """Example 2: Process CSV file"""
    print("📊 Example 2: Processing CSV Files")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    processor = DatasetProcessor(analyzer)
    
    # Process the sample CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'sample_dataset.csv')
    
    try:
        results = processor.process_csv_file(
            csv_path, 
            text_column='text', 
            label_column='sentiment'  # For validation
        )
        
        print(f"📊 Processed {results['total_processed']} texts from CSV")
        print(f"📈 Statistics: {results['statistics']}")
        
        # Export results
        output_path = os.path.join(os.path.dirname(__file__), 'csv_results.json')
        processor.export_results(results, output_path, 'json')
        print(f"💾 Results exported to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
    
    print("\\n" + "=" * 50 + "\\n")

def example_3_json_file():
    """Example 3: Process JSON file"""
    print("📄 Example 3: Processing JSON Files")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    processor = DatasetProcessor(analyzer)
    
    # Process the sample JSON file
    json_path = os.path.join(os.path.dirname(__file__), 'sample_dataset.json')
    
    try:
        results = processor.process_json_file(
            json_path,
            text_field='text',
            label_field='sentiment'  # For validation
        )
        
        print(f"📊 Processed {results['total_processed']} texts from JSON")
        print(f"📈 Statistics: {results['statistics']}")
        
        # Export results as CSV
        output_path = os.path.join(os.path.dirname(__file__), 'json_results.csv')
        processor.export_results(results, output_path, 'csv')
        print(f"💾 Results exported to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing JSON: {e}")
    
    print("\\n" + "=" * 50 + "\\n")

def example_4_api_usage():
    """Example 4: Using API endpoints for dataset processing"""
    print("🌐 Example 4: Using API Endpoints")
    print("=" * 50)
    
    BASE_URL = "http://localhost:8000"
    
    # Test if API is running
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API not running. Start with: python main.py")
            return
        print("✅ API is running")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API. Start with: python main.py")
        return
    
    # Example 4a: Dataset endpoint with JSON payload
    print("\\n4a. Using /predict/dataset endpoint:")
    
    payload = {
        "texts": [
            "Outstanding product quality!",
            "Terrible customer experience.",
            "Average performance overall."
        ],
        "labels": ["positive", "negative", "neutral"]  # For validation
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/dataset",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 API processed {result['total_processed']} texts")
            print(f"📈 Statistics: {result['statistics']}")
        else:
            print(f"❌ API Error: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Example 4b: CSV file upload
    print("\\n4b. Uploading CSV file to API:")
    
    csv_path = os.path.join(os.path.dirname(__file__), 'sample_dataset.csv')
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'rb') as f:
                files = {'file': ('sample_dataset.csv', f, 'text/csv')}
                data = {
                    'text_column': 'text',
                    'label_column': 'sentiment',
                    'export_format': 'json'
                }
                
                response = requests.post(
                    f"{BASE_URL}/predict/csv",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    # Save the returned results
                    with open('api_csv_results.json', 'wb') as output_file:
                        output_file.write(response.content)
                    print("✅ CSV processed and results saved to api_csv_results.json")
                else:
                    print(f"❌ CSV Upload Error: {response.text}")
        except Exception as e:
            print(f"❌ CSV upload failed: {e}")
    
    print("\\n" + "=" * 50 + "\\n")

def example_5_custom_datasets():
    """Example 5: Creating custom datasets from various sources"""
    print("🛠️ Example 5: Custom Dataset Creation")
    print("=" * 50)
    
    # Simulate data from different sources
    social_media_data = [
        {"platform": "twitter", "text": "Love the new features! #innovation", "user_id": 123},
        {"platform": "facebook", "text": "Customer support was very helpful", "user_id": 456},
        {"platform": "instagram", "text": "Beautiful product design 📱✨", "user_id": 789}
    ]
    
    review_data = [
        {"rating": 5, "comment": "Excellent quality and fast shipping!"},
        {"rating": 2, "comment": "Product arrived damaged and support was slow"},
        {"rating": 4, "comment": "Good value for money, would recommend"}
    ]
    
    # Extract texts from different data structures
    texts_from_social = [item['text'] for item in social_media_data]
    texts_from_reviews = [item['comment'] for item in review_data]
    
    # Combine all texts
    all_texts = texts_from_social + texts_from_reviews
    
    # Process combined dataset
    analyzer = SentimentAnalyzer()
    processor = DatasetProcessor(analyzer)
    
    results = processor.process_text_list(all_texts)
    
    print(f"📊 Processed {results['total_processed']} texts from mixed sources")
    print(f"📈 Sentiment Distribution: {results['statistics']['sentiment_percentages']}")
    
    # Show source breakdown
    print("\\n🔍 Results by Source:")
    for i, result in enumerate(results['results']):
        source = "Social Media" if i < len(texts_from_social) else "Reviews"
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        sentiment_emoji = emoji.get(result['sentiment'], '')
        print(f"{source}: {sentiment_emoji} {result['sentiment']} - \"{result['text'][:40]}...\"")
    
    print("\\n" + "=" * 50 + "\\n")

def main():
    """Run all examples"""
    print("🚀 SENTIMENT ANALYSIS DATASET PROCESSING EXAMPLES")
    print("=" * 60)
    print("This script demonstrates various ways to process datasets for sentiment analysis.\\n")
    
    try:
        example_1_text_list()
        example_2_csv_file()
        example_3_json_file()
        example_4_api_usage()
        example_5_custom_datasets()
        
        print("🎉 All examples completed successfully!")
        print("\\n📝 Summary of Dataset Processing Options:")
        print("  1. Direct text lists with optional validation labels")
        print("  2. CSV files with configurable column names")
        print("  3. JSON files with flexible field mapping")
        print("  4. RESTful API endpoints for remote processing")
        print("  5. Custom data extraction from various sources")
        print("\\n🔗 Available API Endpoints:")
        print("  • POST /predict/dataset - Process text arrays")
        print("  • POST /predict/csv - Upload CSV files")
        print("  • POST /predict/json - Upload JSON files")
        print("  • GET /docs - Interactive API documentation")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()