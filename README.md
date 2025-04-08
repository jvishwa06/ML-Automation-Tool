# ML-Automation

An automated machine learning tool with a user-friendly Gradio interface that streamlines the entire ML workflow.

## Features

- **Automated Data Preprocessing**: Handles missing values, encoding categorical variables, and scaling numeric features
- **Interactive Exploratory Data Analysis**: Generate various plots and visualizations with a few clicks
- **Automated Model Training**: Trains multiple ML models appropriate for your task (classification or regression)
- **Model Evaluation**: Compares model performance with relevant metrics
- **Feature Importance Analysis**: Visualizes feature importance for better interpretability
- **Model Export**: Download trained models for deployment

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages (install using `pip install -r requirements.txt`):
  - gradio
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - joblib

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ML-Automation.git
cd ML-Automation
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and go to the URL displayed in the terminal (typically http://127.0.0.1:7860)

3. Upload your CSV data file, select the target column and task type (classification or regression)

4. Click "Process Data and Train Models" to start the automated ML workflow

5. Explore the different tabs to analyze data, visualize results, and download trained models

## How It Works

The tool consists of three main modules:

1. **Preprocessing Module**: Handles data cleaning, feature engineering, and preparation for model training
2. **Modeling Module**: Trains various ML models and evaluates their performance
3. **Visualization Module**: Generates informative plots for data exploration and model analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
