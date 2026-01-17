from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create the DAG
dag = DAG(
    'sentiment_model_retraining',
    default_args=default_args,
    description='Automated sentiment model retraining pipeline',
    schedule_interval='@weekly',  # Run weekly
    catchup=False,
    tags=['ml', 'sentiment', 'retraining']
)

# Configuration
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_DIR = "/tmp/sentiment_models"
PERFORMANCE_THRESHOLD = 0.80
IMPROVEMENT_THRESHOLD = 0.02  # New model must be 2% better

def check_model_performance(**context):
    """Check if model needs retraining based on performance metrics"""
    logging.info("Checking model performance...")

    # Load test dataset to evaluate current model
    dataset = load_dataset("tweet_eval", "sentiment")
    test_data = dataset['test'].select(range(200))  # Sample for quick check

    # Load current model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Evaluate
    correct = 0
    total = len(test_data)
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    for sample in test_data:
        inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        if predicted_class == sample['label']:
            correct += 1

    current_accuracy = correct / total
    logging.info(f"Current model accuracy: {current_accuracy:.4f}")

    # Store for comparison later
    context['task_instance'].xcom_push(key='current_accuracy', value=current_accuracy)

    if current_accuracy < PERFORMANCE_THRESHOLD:
        logging.warning(f"Model performance below threshold: {current_accuracy} < {PERFORMANCE_THRESHOLD}")
        return True
    else:
        logging.info(f"Model performance acceptable: {current_accuracy} >= {PERFORMANCE_THRESHOLD}")
        # Still retrain to potentially improve
        return True

def collect_new_data(**context):
    """Collect new data for retraining"""
    logging.info("Collecting new training data...")

    # Load fresh training data from public dataset
    dataset = load_dataset("tweet_eval", "sentiment")

    # In production, this would:
    # 1. Collect new labeled data from production logs
    # 2. Query human-labeled feedback
    # 3. Pull from data lake/warehouse

    # For this implementation, we use a subset of the dataset
    train_data = dataset['train'].select(range(1000))  # Sample for faster training
    validation_data = dataset['validation'].select(range(200))

    logging.info(f"Collected {len(train_data)} training samples and {len(validation_data)} validation samples")

    # Store data for next task
    context['task_instance'].xcom_push(key='train_data', value=train_data)
    context['task_instance'].xcom_push(key='validation_data', value=validation_data)

    return len(train_data)

def validate_new_data(**context):
    """Validate the collected data"""
    logging.info("Validating new data...")

    train_count = context['task_instance'].xcom_pull(task_ids='collect_data')

    if train_count is None or train_count == 0:
        raise ValueError("No new data collected")

    if train_count < 100:
        raise ValueError(f"Insufficient data for retraining: {train_count} < 100")

    logging.info(f"Data validation passed: {train_count} samples available")
    return True

def retrain_model(**context):
    """Retrain the sentiment model with new data"""
    logging.info("Starting model retraining...")

    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Get training data from XCom
    train_data = context['task_instance'].xcom_pull(task_ids='collect_data', key='train_data')
    validation_data = context['task_instance'].xcom_pull(task_ids='collect_data', key='validation_data')

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    train_dataset = Dataset.from_dict({
        'text': [item['text'] for item in train_data],
        'label': [item['label'] for item in train_data]
    })

    val_dataset = Dataset.from_dict({
        'text': [item['text'] for item in validation_data],
        'label': [item['label'] for item in validation_data]
    })

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{MODEL_DIR}/training_output",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,  # Small number for quick retraining
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{MODEL_DIR}/logs",
        logging_steps=50,
    )

    # Metric computation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    logging.info("Fine-tuning model...")
    trainer.train()

    # Save the retrained model
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{MODEL_DIR}/sentiment_model_{model_version}"

    os.makedirs(MODEL_DIR, exist_ok=True)
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    logging.info(f"Model saved to {model_path}")

    # Store model path for evaluation
    context['task_instance'].xcom_push(key='new_model_path', value=model_path)
    context['task_instance'].xcom_push(key='model_version', value=model_version)

    return model_version

def evaluate_model(**context):
    """Evaluate the retrained model and compare with current"""
    logging.info("Evaluating retrained model...")

    # Get paths
    new_model_path = context['task_instance'].xcom_pull(task_ids='retrain_model', key='new_model_path')
    current_accuracy = context['task_instance'].xcom_pull(task_ids='check_performance', key='current_accuracy')

    # Load test dataset
    dataset = load_dataset("tweet_eval", "sentiment")
    test_data = dataset['test'].select(range(200))

    # Load new model
    tokenizer = AutoTokenizer.from_pretrained(new_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(new_model_path)

    # Evaluate new model
    correct = 0
    total = len(test_data)

    for sample in test_data:
        inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        if predicted_class == sample['label']:
            correct += 1

    new_accuracy = correct / total

    logging.info(f"New model accuracy: {new_accuracy:.4f}")
    logging.info(f"Current model accuracy: {current_accuracy:.4f}")
    logging.info(f"Improvement: {(new_accuracy - current_accuracy):.4f}")

    # Store metrics
    context['task_instance'].xcom_push(key='new_accuracy', value=new_accuracy)

    # Decide if we should deploy
    should_deploy = new_accuracy > (current_accuracy + IMPROVEMENT_THRESHOLD)

    if should_deploy:
        logging.info(f"✅ New model shows improvement > {IMPROVEMENT_THRESHOLD:.2%}. Proceeding to deploy.")
    else:
        logging.info(f"❌ New model does not show sufficient improvement. Keeping current model.")

    return should_deploy

def deploy_model(**context):
    """Deploy the new model if evaluation passed"""
    logging.info("Deploying new model...")

    new_model_path = context['task_instance'].xcom_pull(task_ids='retrain_model', key='new_model_path')
    model_version = context['task_instance'].xcom_pull(task_ids='retrain_model', key='model_version')
    new_accuracy = context['task_instance'].xcom_pull(task_ids='evaluate_model', key='new_accuracy')

    # In production, this would:
    # 1. Upload model to model registry (MLflow, HuggingFace Hub, etc.)
    # 2. Update production API to use new model
    # 3. Create a backup of the old model
    # 4. Update monitoring dashboards

    production_model_path = f"{MODEL_DIR}/production"
    os.makedirs(production_model_path, exist_ok=True)

    # Simulate deployment by copying to production directory
    import shutil
    if os.path.exists(production_model_path):
        shutil.rmtree(production_model_path)
    shutil.copytree(new_model_path, production_model_path)

    logging.info(f"✅ Model {model_version} deployed to production")
    logging.info(f"   Accuracy: {new_accuracy:.4f}")
    logging.info(f"   Path: {production_model_path}")

    return model_version

# Define tasks
check_performance = PythonOperator(
    task_id='check_performance',
    python_callable=check_model_performance,
    dag=dag
)

collect_data = PythonOperator(
    task_id='collect_data',
    python_callable=collect_new_data,
    dag=dag
)

validate_data = PythonOperator(
    task_id='validate_data',
    python_callable=validate_new_data,
    dag=dag
)

retrain = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag
)

evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

deploy = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Send notification
notify = BashOperator(
    task_id='send_notification',
    bash_command='echo "Model retraining pipeline completed successfully. New model deployed."',
    dag=dag
)

# Define task dependencies
check_performance >> collect_data >> validate_data >> retrain >> evaluate >> deploy >> notify
