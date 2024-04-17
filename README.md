# Churn Prediction for Telecom Customers

This repository contains a machine learning project focused on predicting churn among telecom customers. The goal is to identify customers who are likely to churn and understand the key factors contributing to customer churn. Based on the predictive model's insights, we provide actionable recommendations for the telecom company to retain customers more effectively.

## Project Structure

- `data/`: Folder containing the dataset used in the analysis.
- `src/`: Source code with utility functions and model scripts used across the project.
- `README.md`: Documentation and overview of the project.

## Installation

To set up your environment to run this code, you'll need Python and several libraries used in data manipulation, machine learning, and visualization.

### Prerequisites

- Python 3.8 or later
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mustafakaswani10/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv env
   # On Windows
   env\Scripts\activate
   # On Unix or MacOS
   source env/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

To run the churn prediction models and generate predictions, follow these steps:

1. **Ensure the data is in the `data/` directory**.
2. **Run the script**:
   ```bash
   python telcochurn.py
   ```

This will execute the data preprocessing, model training, and evaluation steps, and it will save the trained models to the `models/` directory.

## Model Insights and Recommendations

### Key Findings

- More frequent customer service calls are associated with higher churn rates, indicating potential issues with service quality.
- Higher monthly charges, especially without corresponding data plans, lead to increased churn.
- Customers with data plans tend to show lower churn rates, suggesting that offering appropriate data plans can improve customer retention.

### Recommendations

- **Improve Customer Service**: Enhance the quality and responsiveness of customer support to address issues proactively.
- **Optimize Pricing**: Review and adjust the pricing plans to ensure competitiveness and fairness.
- **Promote Data Plans**: Introduce attractive data plans for high data users and promote them actively to the relevant customer segments.
- **Engage Customers**: Increase customer engagement through regular communication and personalized offers based on the customer's usage patterns.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Data provided by Kaggle
