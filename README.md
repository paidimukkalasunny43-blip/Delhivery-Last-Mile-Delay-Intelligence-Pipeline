# Delhivery – Last Mile Delay Intelligence Pipeline

## Project Overview

The **Delhivery Last Mile Delay Intelligence Pipeline** is a Python-based data analytics and machine learning project designed to analyze shipment data, study delivery delays, and generate actionable insights.

The project includes data preprocessing, exploratory data analysis, regression, classification, model evaluation, and hub segmentation.

## Objectives

- Analyze shipment data and identify delivery delay patterns.
- Perform data cleaning and exploratory data analysis.
- Build machine learning models for delay prediction.
- Evaluate model performance using appropriate metrics.
- Segment delivery hubs based on their operational characteristics.
- Generate insights and recommendations for shipments.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Data Visualization

## Project Structure

```text
Delhivery-Last-Mile-Delay-Intelligence-Pipeline/
│
├── data/
│   ├── hubs.csv
│   ├── modelling_table.csv
│   ├── new_shipments.csv
│   └── raw_shipments.csv
│
├── models/
│   ├── delay_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   └── impute_values.csv
│
├── outputs/
│   ├── hub_profile.csv
│   ├── hub_segments.csv
│   ├── part_a_checks.csv
│   ├── part_b_metrics.csv
│   └── other analysis outputs
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
└── requirements.txt
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/paidimukkalasunny43-blip/Delhivery-Last-Mile-Delay-Intelligence-Pipeline.git
```

### 2. Navigate to the project folder

```bash
cd Delhivery-Last-Mile-Delay-Intelligence-Pipeline
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Project Modules

**Part A – Data Cleaning and EDA**
- Data preprocessing
- Data quality checks
- Correlation analysis
- Delay analysis

**Part B – Regression**
- Regression modelling
- Model evaluation
- Hyperparameter analysis

**Part C – Classification**
- Classification modelling
- Feature selection
- Threshold tuning
- Model evaluation

**Part D – Hub Segmentation**
- Hub analysis
- Hub profiling
- Segmentation
- Operational insights

## Model and Prediction

The trained model and supporting files are stored in the `models/` directory.

The `predict_delay.py` script is provided for shipment delay prediction.

## Results

The analysis and model outputs are available in the `outputs/` directory.

These include model metrics, feature scores, hub profiles, segmentation results, and shipment action recommendations.

## Author

**Sunny Paidimukkala**

GitHub: https://github.com/paidimukkalasunny43-blip

## License

This project is available for educational and portfolio purposes. Add a formal license if you intend to grant specific reuse permissions.
