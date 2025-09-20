# PlaszymerV1

## Overview

PlaszymerV1 is a machine learning framework designed for the classification and analysis of plastic-degrading enzymes (Plaszymes). 
It leverages protein sequence embeddings and supervised learning models to predict degradation potential across different plastic types.

The repository provides utilities for data preprocessing, feature embedding generation, model training, evaluation, 
and prediction on new protein sequences.

---

## Repository Structure

```
PlaszymerV1/
├── data/               # (Optional) Dataset folder for input sequences and labels
├── models/             # Saved models (e.g., classifier.pkl, label_encoder.pkl)
├── results/            # Output predictions and evaluation metrics
├── utils.py            # Helper functions (sequence cleaning, embedding extraction, etc.)
├── train.py            # Training script for model building
├── test.py             # Testing/inference script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Installation

esm1b model and configs are needed while using this model. You can download from https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt

It is recommended to create a virtual environment (e.g., Conda or venv).

```bash
# Example using conda
conda create -n plaszymer python=3.9
conda activate plaszymer

# Install required dependencies
pip install -r requirements.txt
```

---

## Usage

### Data Preparation

- Input should be provided as a CSV file containing at least two columns:
  - `protein_id`: Unique identifier for each sequence
  - `sequence`: Raw protein sequence (string of amino acids)

Example:

```csv
protein_id,sequence,degradable_plastics
seq1,MAADQLTAR...,PET
seq2,MKVLWAALL...,PE
```

### Training

Run the training script with your dataset:

```bash
python train.py --train_csv path/to/train.csv --out_dir models/ --batch_size 4 --epochs 50
```

This will save the trained classifier and label encoder to the `models/` folder.

### Testing / Prediction

Run the test script with a new dataset:

```bash
python test.py --test_csv path/to/test.csv --model_dir models/ --out_csv results/predictions.csv --batch_size 4
```

Output will include:
- Prediction probabilities for each plastic class
- Top hit per sequence 

### Example Output

```csv
protein_id,PET,PE,PU,PHA,PCL,PBAT,PLA,TopHit
seq1,0.92,0.03,0.01,0.01,0.00,0.02,0.01,PET
seq2,0.05,0.80,0.03,0.01,0.02,0.06,0.03,PE
```

---

## Results

Model performance is evaluated using standard classification metrics (accuracy, precision, recall, F1-score).  
Results will be stored in the `results/` directory after evaluation.

