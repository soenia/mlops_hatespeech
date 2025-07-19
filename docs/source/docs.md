# MLOps Hate Speech Project Documentation
Documentation of core classes and functions used in this project.

## Data Preparation

### load_and_prepare_dataset
::: src.mlops_hatespeech.data.load_and_prepare_dataset

## Model Training
### get_config
::: src.mlops_hatespeech.train.get_config
### train_model
::: src.mlops_hatespeech.train.train_model
### train
::: src.mlops_hatespeech.train.train

## Model Evaluation
### find_latest_checkpoint
::: src.mlops_hatespeech.evaluate.find_latest_checkpoint
### compute_metrics
::: src.mlops_hatespeech.evaluate.compute_metrics
### main
::: src.mlops_hatespeech.evaluate.main


## Drift Detection
### get_bert_embeddings
::: src.mlops_hatespeech.drift_detector.get_bert_embeddings
### download_predictions_from_gcs
::: src.mlops_hatespeech.drift_detector.download_predictions_from_gcs
### upload_report_to_gcs
::: src.mlops_hatespeech.drift_detector.upload_report_to_gcs
### main
::: src.mlops_hatespeech.drift_detector.main
