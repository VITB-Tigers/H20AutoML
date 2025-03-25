# H2OAutoML
A comprehensive framework for training AutoML models using H2O's AutoML library. This project includes data ingestion, transformation, and model training functionalities, all accessible through an interactive Streamlit web application.

## 📑 Table of Contents

- [Why AutoML?](#-why-automl)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)

## 🤔 Why AutoML?

AutoML is a very valuable tool for several reasons:

- **Efficiency**: Quickly upload and check datasets to figure out which algorithms would work best with the highest accuracy.
- **Accessibility**: Simplifies the machine learning process, making it accessible to non-experts.
- **Automation**: Automates repetitive tasks such as feature engineering, model selection, and hyperparameter tuning.
- **Performance**: Often achieves competitive performance with less manual intervention.
- **Scalability**: Can handle large datasets and complex models efficiently.
- **Consistency**: Reduces human error and ensures consistent results across different datasets and projects.
- **Time-saving**: Speeds up the model development process, allowing data scientists to focus on more strategic tasks.

By leveraging AutoML, you can streamline your machine learning workflow and achieve high-quality results with minimal effort.

## ✨ Features

- 📊 Data ingestion and preprocessing
- 🔄 Data transformation (scaling, encoding, imputation, etc.)
- 🤖 Automated machine learning model training using H2O AutoML
- 📈 Model evaluation and leaderboard generation
- 💾 Integration with Redis for data storage and retrieval

## 🔧 Prerequisites

- Python 3.11.15
- H2O 3.46.0.6
- Streamlit 1.40.2
- Redis 5.2.1
- Pandas 2.2.3

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/VITB-Tigers/H2OAutoML.git
cd H2OAutoML
```

2. Create and activate a virtual environment (recommended). If using Conda:
```bash
conda create -n env_name python==3.11.15 -y
conda activate env_name
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Configure Redis connection, upload your dataset, perform data transformations, and train AutoML models!

## 📁 Project Structure

```
DIY-Python-H2OAutoML/
    ├── data/
    │   ├── dataset1.csv (Datasets 1-4 are smaller datasets for better accuracy but higher accuracy)
    │   ├── dataset2.csv
    │   ├── dataset3.csv
    │   ├── dataset4.csv
    │   └── Mock_Data.csv (A more realistic, bigger dataset with lower accuracy due to being a mock dataset.)
    ├── source/
    │   ├── app.py
    │   ├── automl.py
    │   ├── db_utils.py
    │   └── transform.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

## 📚 Documentation

For detailed information about the project, please refer to:
- [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) - H2O AutoML documentation
- [Streamlit](https://docs.streamlit.io/) - Streamlit documentation
- [Redis](https://redis.io/documentation) - Redis documentation

