# Delhivery Last-Mile Delay Intelligence Pipeline

## 1. Project Overview

This project develops a machine learning pipeline to predict last-mile shipment delays and help logistics operations teams identify shipments and hubs that may require attention.

The pipeline covers data cleaning, exploratory data analysis, regression, classification, hub segmentation, model packaging, and shipment-level operational recommendations.

## 2. Project Objectives

- Clean and prepare shipment data for analysis.
- Identify factors associated with delivery delays.
- Predict delivery time using regression models.
- Classify shipments as delayed or not delayed.
- Segment hubs based on their operational delay patterns.
- Generate actionable recommendations for shipments at risk.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Machine Learning
- Data Analysis

## 4. Project Structure

```text
Delhivery-Last-Mile-Delay-Intelligence-Pipeline/
│
├── data/
│   ├── raw_shipments.csv
│   ├── hubs.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
│
├── models/
│   └── delay_model.pkl
│
├── outputs/
│   ├── part_a_checks.csv
│   ├── part_a_delay_by_group.csv
│   ├── part_a_correlations.csv
│   ├── part_b_metrics.csv
│   ├── part_c_models.csv
│   ├── part_c_final_test.csv
│   ├── hub_segments.csv
│   ├── segment_profile.csv
│   └── new_shipment_actions.csv
│
├── src/
│   └── shipment_prep.py
│
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── requirements.txt
└── README.md
```

## 5. How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/paidimukkalasunny43-blip/Delhivery-Last-Mile-Delay-Intelligence-Pipeline.git
```

### Step 2: Open the project directory

```bash
cd Delhivery-Last-Mile-Delay-Intelligence-Pipeline
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the pipeline

Execute the scripts in the following order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

The scripts generate analysis outputs, trained model files, hub segments, and shipment-level recommendations.

## 6. Data Findings

The initial shipment dataset contained 1,224 rows.

After removing 24 duplicate records, 1,200 unique shipments remained.

Important findings:

- Overall shipment delay rate: **23.17%**.
- Express service delay rate: **48.97%**.
- Standard service delay rate: **20.56%**.
- Economy service delay rate: **8.57%**.
- Hub Tier 3 delay rate: **44.00%**.
- Shipments travelling more than 600 km had an **85.23%** delay rate.

Distance, hub tier, number of stops, and hub capacity were important factors associated with shipment delays.

## 7. Model Results – Regression

Multiple regression models were evaluated to predict delivery hours.

| Model | Validation RMSE |
|---|---:|
| Linear Regression | 4.8912 |
| Decision Tree | 6.2796 |
| Random Forest | 5.4680 |
| Gradient Boosting | 4.9165 |

Linear Regression was selected based on validation performance.

**Held-out test results:**

- MAE: 3.5671 hours
- RMSE: 4.5235 hours
- R² Score: 0.9493

These results indicate that the selected model performed well on the held-out test dataset.

## 8. Model Results – Classification

Classification models were evaluated to identify delayed shipments.

The final configuration used:

- Feature selection: 7 features
- Random Forest: 100 estimators
- Classification threshold: 0.50

**Held-out test results:**

- Accuracy: 86.67%
- Precision: 68.89%
- Recall: 75.61%
- F1 Score: 72.09%

The model identifies shipments at risk of delay and can support proactive operational decisions.

## 9. Hub Segmentation

Hub-level operational profiles were analyzed using K-Means clustering.

Three operational segments were identified:

| Segment | Hubs | Delay Rate |
|---|---:|---:|
| Standard | 7 | 2.57% |
| Watch | 8 | 15.50% |
| Escalate | 9 | 46.00% |

These segments help prioritize hubs based on their observed delay patterns.

## 10. Operations Recommendation

The analysis supports the following operational recommendations:

- **Standard:** Continue normal operations and routine monitoring.
- **Watch:** Monitor shipment delays, capacity, and delivery performance more frequently.
- **Escalate:** Prioritize operational review, investigate capacity constraints, and consider additional resources.

The shipment prediction pipeline also generates actions for shipments identified as high risk.

In the evaluated output, 8 of 48 new shipments were flagged for attention:
- 2 proactive customer calls.
- 6 escalations to hub managers.

## 11. Production Plan

A possible production implementation would include:

1. Integrating shipment and hub data from operational systems.
2. Running the prediction pipeline on incoming shipments.
3. Generating delay-risk scores and recommended actions.
4. Providing a dashboard for operations teams.
5. Monitoring prediction performance and model drift.
6. Retraining models using updated shipment data.

This is a proposed future deployment plan, not a claim that the pipeline is already connected to Delhivery's production systems.

## 12. Limitations

- Model performance is based on the available project dataset.
- The dataset contains 1,200 unique shipments after duplicate removal.
- Performance on new real-world operational data may differ.
- Classification predictions can include false positives and false negatives.
- External validation and continuous production monitoring are still required.
- The proposed production plan requires integration with operational systems.

## 13. Conclusion

The Delhivery Last-Mile Delay Intelligence Pipeline demonstrates how data analysis and machine learning can support last-mile logistics operations.

It combines delivery-time prediction, delay classification, hub segmentation, and shipment-level recommendations into a structured workflow.

The project provides a foundation for identifying delay risks and supporting data-driven operational decisions.

## 14. Author

**Project:** Delhivery Last-Mile Delay Intelligence Pipeline

**GitHub Repository:**  
https://github.com/paidimukkalasunny43-blip/Delhivery-Last-Mile-Delay-Intelligence-Pipeline
