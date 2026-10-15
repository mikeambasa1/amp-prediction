# amp-prediction

Predicts whether a peptide is antimicrobial (AMP) from its sequence. Evaluated on unseen sequence clusters, with dataset bias checks, and served in a Streamlit app.

## Result
On test peptides from clusters the model never saw, the final random forest (31 features) scores **ROC AUC 0.931 and MCC 0.721**.

| Model | ROC AUC | MCC |
|---|---|---|
| Logistic regression | 0.859 | 0.578 |
| Random forest, simple features (leading M removed) | 0.933 | 0.715 |
| Random forest, compact features (final) | 0.931 | 0.721 |

## What I found
- Non-AMPs started with M far more often than AMPs (37.2% vs 2.0%), a database artifact. Removing the leading M changed ROC AUC by only 0.005.
- Peptide length was matched between classes, so it is not a shortcut.
- 400 extra dipeptide features added nothing (cross-validated ROC AUC 0.903 vs 0.898 for the 31-feature set), so I kept the compact set.

## Data
- AMPs: DRAMP
- Non-AMPs: UniProt
- `data/processed/peptides_clean.csv` is the cleaned data, and `data/processed/split.csv` is the train/test split.

## How to run
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```
Notebooks, in order: `01_Data_cleaning_and_clustering`, `02_features_and_baseline_model`, `03_richer_features`.

## Limitations
Trained on peptides of 10 to 50 amino acids only. The non-AMP set comes from a different database than the AMPs, so the results may not carry over to other peptide sources. A research tool, not a diagnostic.
