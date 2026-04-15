Telecom Customer Churn Prediction — Profit-Based Model Evaluation

Can a model with the highest AUC actually generate less profit than one ranked lower?
Yes — and this project proves it empirically.

The Problem
Telecom companies lose significant revenue to customer churn every year. The standard approach is to build ML models, rank them by accuracy or AUC, and deploy the winner. But this ignores a critical asymmetry:
OutcomeCostMissing a churner (False Negative)−$777 (full Customer Lifetime Value lost)Unnecessary retention offer (False Positive)−$13 (20% discount on monthly charges)Cost ratio60:1
A model optimized for AUC treats these errors equally. A profit-optimized model does not.
Key Finding
ModelAUC RankProfit RankRank Changed?Logistic Regression1st (84.8%)2nd ($403,882)⚠ YesGradient Boosting2nd (84.6%)1st ($406,369)⚠ YesRandom Forest3rd (82.6%)3rd ($401,098)No
The model ranked #1 by AUC is not the most profitable. This confirms the research gap identified by Imani et al. (2025) — a systematic review of 100+ churn studies that found profit-based metrics are rarely adopted.
Methodology
Dataset

Source: IBM Telco Customer Churn (Kaggle)
Size: 7,043 customers × 21 features
Churn rate: 26.5% (class imbalance present)

Pipeline
Data Understanding → Cleaning → EDA → Preprocessing → Model Training → Traditional Evaluation → Profit Evaluation → Conclusion
Models Trained

Logistic Regression (linear baseline)
Random Forest (ensemble)
Gradient Boosting (sequential boosting)

All trained on 70/30 stratified split with 5-fold cross-validation.
Evaluation Frameworks

Traditional: Accuracy, Precision, Recall, F1, ROC-AUC
Profit-based: CLV-derived cost matrix + Expected Maximum Profit (EMP) from Verbraken et al. (2013)

Profit Framework
CLV = Avg Monthly Charge × 12 months = $64.76 × 12 = $777
Retention Cost = 20% × Monthly Charge = $13
Optimal Threshold ≠ 0.5 → found via profit curve sweep (0.01–0.99)
Key Insights from EDA

Contract type is the strongest churn predictor — month-to-month customers churn at 42.7% vs 2.8% for two-year contracts (15× difference)
Churn is front-loaded — customers in their first year have ~48% churn rate; those past 4 years drop below 10%
Higher-paying customers churn more — churners pay ~$74/month avg vs $55 for retained customers
Optimal classification threshold is well below 0.5 for all models due to the 60:1 cost asymmetry

Project Structure
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_traditional_evaluation.ipynb
│   ├── 07_profit_evaluation.ipynb
│   └── 08_conclusion.ipynb
├── data/
│   └── telco_churn.csv
├── requirements.txt
└── README.md
Tech Stack
Language: Python 3.x
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
Models: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
Evaluation: Custom EMP implementation based on Verbraken et al. (2013)
How to Run
bashgit clone https://github.com/harsh241005/churn-prediction-profit-analysis.git
cd churn-prediction-profit-analysis
pip install -r requirements.txt
Open notebooks in order (01 → 08) using Jupyter Notebook or VS Code.
References

Imani, M., Jafari, M., Bharti, P., & Chidambaranathan, S. (2025). Customer churn prediction: A systematic review. Machine Learning and Knowledge Extraction, 7, 105.
Verbraken, T., Verbeke, W., & Baesens, B. (2013). A novel profit maximizing metric for measuring classification performance of customer churn prediction models. IEEE TKDE, 25(5), 961–973.

Author
Harsh Palyekar — B.Sc. Data Science, Goa University
LinkedIn · GitHub
