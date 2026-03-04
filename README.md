# NLP Coursework: PCL Detection (Task 1)

This repository trains a RoBERTa-based classifier for PCL detection and supports test-set inference with configurable output paths.

## Project Structure
- `BestModel/roberta_baseline.py`: main training/inference script
- `dont_patronize_me.py`: dataset loader
- `train_semeval_parids-labels.csv`, `dev_semeval_parids-labels.csv`: ID lists
- `train_dontpatronizeme_pcl.tsv`, `dev_dontpatronizeme_pcl.tsv`, `task4_test.tsv`: raw datasets
- `outputs/`: training checkpoints and best model
- `Predicted_output/` or `prediction/results/`: prediction outputs

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training (dev predictions)

```bash
python BestModel/roberta_baseline.py
```

This trains the model and writes dev predictions to `prediction/results/dev.txt` by default.

## Test Inference Only

```bash
python BestModel/roberta_baseline.py --mode test
```

This loads the best available checkpoint and writes test predictions to `prediction/results/test.txt`.

## Custom Output Paths (dev + test)

```bash
python BestModel/roberta_baseline.py \
  --dev-result Predicted_output/dev.txt \
  --test-result Predicted_output/test.txt
```

This runs training first (dev output), then test inference, and writes both files to the paths you specify.

## Notes
- CUDA is used automatically if available.
- Best checkpoint is saved under `outputs/best_model_task1`.
