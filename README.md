
# FraudShield

**An Explainable, Cost-Aware Fraud Triage System with Uncertainty-Based Deferral to Human Review**

FraudShield is a machine-learning-based credit card fraud detection and triage application. It analyses transactions, estimates fraud probabilities, and recommends whether a transaction should be approved, blocked, or deferred for human review.

The system combines supervised machine learning, probability calibration, explainable AI, cost-sensitive decision-making, and a persistent SQLite database.

## 1. Features

- **Fraud detection:** Predicts the probability that a transaction is fraudulent.
- **Cost-sensitive decisions:** Uses configurable cost assumptions to compare approval, blocking, and human review.
- **Human review queue:** Stores deferred transactions for manual review.
- **Explainable AI:** Uses SHAP to display feature contributions for reviewed transactions.
- **Transaction history:** Maintains previously scored transactions using SQLite.
- **Batch triage:** Processes multiple transactions and assigns decisions.
- **Performance analytics:** Displays model evaluation metrics and experiment results.
- **Reports:** Provides downloadable transaction CSV reports and evaluation summaries.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| Database | SQLite |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, LightGBM |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib |
| Model persistence | Joblib |

## 3. Project Structure

```text
fraudshield/
├── app.py
├── requirements.txt
├── README.md
├── fraudshield/
│   ├── __init__.py
│   ├── costs.py
│   ├── data.py
│   ├── database.py
│   ├── decision.py
│   └── models.py
├── data/
│   ├── creditcard.csv
│   └── fraudshield.db
├── artifacts/
│   ├── fraudshield.joblib
│   ├── eval.npz
│   └── test_sample.csv
├── results/
│   ├── summary.csv
│   ├── headline.json
│   ├── expected_cost_per_seed.csv
│   └── figures/
├── tests/
│   └── test_fraudshield.py
├── run_experiments.py
├── run_cost_sensitivity.py
└── plot_cost_sensitivity.py
```

## 4. Dataset

FraudShield uses the Credit Card Fraud Detection dataset from Kaggle, provided by the Machine Learning Group of Université Libre de Bruxelles (ULB).

Dataset characteristics:

- 284,807 transactions
- 492 fraudulent transactions
- 284,315 legitimate transactions
- 30 input features and one target variable
- Fraud prevalence of approximately 0.173%

The dataset contains anonymised PCA-transformed features (`V1` to `V28`), along with transaction time, amount, and class label.

## 5. System Architecture

The system follows this workflow:

1. Load and validate transaction data.
2. Preprocess the dataset and create chronological train, calibration, tuning, and test splits.
3. Train and compare machine learning models.
4. Select a model using validation performance.
5. Calibrate predicted fraud probabilities.
6. Apply cost-sensitive decision-making.
7. Assign one of three decisions:
   - Approve
   - Block
   - Defer to human review
8. Generate SHAP explanations for deferred transactions.
9. Store transaction information and review outcomes in SQLite.
10. Display analytics and downloadable reports through Streamlit.

## 6. Cost-Sensitive Decision-Making

FraudShield evaluates three possible actions:

- **Approve:** Allow the transaction.
- **Block:** Reject the transaction.
- **Defer:** Send the transaction to a human reviewer.

The expected-cost policy considers:

- False-block cost
- Human review cost
- Reviewer accuracy
- Transaction amount
- Calibrated fraud probability
- Review capacity

The default cost values are hypothetical assumptions and can be adjusted. They are not measured real-world financial costs.

## 7. Explainability

FraudShield uses SHAP (SHapley Additive exPlanations) to show which input features contribute most to the selected model's fraud score.

Positive SHAP contributions push the raw model output towards fraud, while negative contributions push it towards legitimate classification.

These contributions describe model behaviour and should not be interpreted as causal explanations.

## 8. Installation

### Prerequisites

- Python 3.10 or later
- pip
- Git (optional)

### Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the Kaggle dataset at:

```text
data/creditcard.csv
```

## 9. Running FraudShield

### Initialise the database

```bash
python -c "from fraudshield.database import init_db; init_db(); print('Database ready')"
```

### Run the dashboard

```bash
streamlit run app.py
```

### Run automated tests

```bash
python -m pytest -q
```

### Re-run the research experiments

```bash
python run_experiments.py --data data/creditcard.csv --seeds 5
```

This regenerates the experiment results and model artifacts. It is not required for normal dashboard operation when the saved artifacts are already available.

### Run cost sensitivity analysis

```bash
python run_cost_sensitivity.py
```

## 10. Database

FraudShield uses SQLite for persistent transaction storage.

The database file is:

```text
data/fraudshield.db
```

It stores transaction records, model decisions, review status, reviewer decisions, notes, and timestamps.

SQLite is used because the application is designed as a local, single-user prototype.

## 11. Evaluation

The system evaluates fraud detection and triage using metrics such as:

- Precision-Recall AUC
- Brier score
- Expected calibration error
- Fraud blocked rate
- Silent miss rate
- Fraud deferred rate
- Deferral rate
- Modelled total cost

The expected-cost results depend on the selected cost assumptions and reviewer accuracy. They should not be interpreted as guaranteed real-world savings.

## 12. Limitations

- The dataset contains anonymised PCA features, limiting direct business interpretation.
- Fraud cases are highly imbalanced.
- Calibration and tuning splits contain relatively few fraud examples.
- Cost values and reviewer accuracy are hypothetical assumptions.
- The system is an academic prototype and has not been validated for deployment in a live banking environment.

## 13. Future Enhancements

- Multi-user authentication and role-based access.
- Integration with real-time transaction streams.
- More extensive reviewer feedback and audit trails.
- Evaluation on additional fraud datasets.
- Improved monitoring for data drift and model degradation.
- Deployment on a cloud-based infrastructure.

---

**FraudShield**

Academic project | Computer Science and Engineering, Data Science
