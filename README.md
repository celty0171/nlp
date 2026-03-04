# NLP Coursework: PCL Detection

This repository trains a RoBERTa-based classifier for PCL detection and supports test-set inference with configurable output paths.

## Project Structure
- `BestModel/roberta_large.py`: main training and inference script
- `dont_patronize_me.py`: dataset loader
- `train_semeval_parids-labels.csv`, `dev_semeval_parids-labels.csv`: ID lists
- `dontpatronizeme_pcl.tsv`: raw dataset
- `train_dontpatronizeme_pcl.tsv`, `dev_dontpatronizeme_pcl.tsv`: training and dev sets reconstructed by ID matching with raw dataset
- `task4_test.tsv`: test set
- `outputs/best_model_task1`: training checkpoints and best model
- `Predicted_output/`: prediction outputs on the dev and test datasets

## Setup

```bash
conda create -n env python=3.10 -y
conda activate env
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start (dev + test)

```bash
python BestModel/roberta_large.py \
  --dev-result Predicted_output/dev.txt \
  --test-result Predicted_output/test.txt
```

This runs training first (dev output), then test inference, and writes both files to the paths you specify.


## Training (dev predictions)

```bash
python BestModel/roberta_large.py
```

This trains the model and writes dev predictions to `prediction/results/dev.txt` by default.

## Test Inference Only

```bash
python BestModel/roberta_large.py --mode test
```

This loads the best available checkpoint and writes test predictions to `prediction/results/test.txt`.


## Notes
- Please use python 3.10 to run the code.
- CUDA is used automatically if available.
- Best checkpoint is saved under `outputs/best_model_task1`.
