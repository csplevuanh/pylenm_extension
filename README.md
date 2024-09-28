# PyLEnM Extension for Predicting Contaminant Attenuation Time Frames

### Authors: Vu Anh Le, Haruko Murakami Wainwright, Hansell Gonzalez-Raymat, Carol Eddy-Dilek

## Overview

This project extends the capabilities of the PyLEnM package to enhance predictions of contaminant attenuation timeframes using multiple machine learning algorithms. It focuses on developing models to predict when contaminants like Sr-90 and I-129 will reach regulatory safety standards at groundwater sites, such as the Savannah River Site (SRS) F-Area.

The following machine learning methods are integrated:
- **Linear Regression** for estimating the time to Maximum Contaminant Level (MCL).
- **Random Forest Regression** to identify key environmental factors affecting contaminant decline.
- **Bidirectional Long Short-Term Memory (LSTM)** for near-term forecasting of contaminant concentration trends.

The results are visualized through various plots, maps, and reports, contributing to long-term monitoring and site closure strategies.

## Code Description

### 1. `Bidirectional_LSTM.ipynb`
This notebook contains the implementation of a **Bidirectional LSTM** model for time series forecasting of contaminant concentrations. The LSTM architecture is designed to capture both past and future dependencies within the sequence data, making it effective for predicting future contaminant trends. The code includes:
- Data preprocessing to generate input sequences.
- Building and training a Bidirectional LSTM model using Keras.
- Predictions for contaminant concentrations and generating visualizations to show performance.

**Key Features:**
- Predicts future contaminant levels based on past data.
- Includes mechanisms for avoiding overfitting through dropout and early stopping.
- Model evaluation using MSE and R² metrics.

### 2. `Random_Forest_Regression.ipynb`
This notebook introduces a **Random Forest Regression** model to analyze the factors affecting the heterogeneity in the time required for contaminants to reach regulatory safety standards. The model identifies critical variables, such as well depth, aquifer characteristics, and geographical factors.

**Key Features:**
- Feature importance analysis to determine the most influential variables.
- Training a Random Forest model using the scikit-learn library.
- Evaluating model performance using Mean Squared Error (MSE) and R² scores.
- Feature importance plots to highlight the factors impacting contaminant attenuation.

### 3. `Linear_Regression_+_Mapping.ipynb`
This notebook applies **Linear Regression** to estimate the time-to-MCL for each analyte across different wells. It also includes visualization functionalities for mapping the concentration trends and spatial distribution of wells using geospatial libraries like Folium.

**Key Features:**
- Linear regression for predicting time-to-MCL.
- Confidence intervals for predictions using statistical methods.
- Geospatial visualization of well locations and contaminant trends using Folium maps.
- Outlier removal for ensuring accurate trends.

## Manuscript Summary

The accompanying manuscript, titled _"Extension of PyLEnM: Machine Learning Algorithms to Assess Site Closure Time Frames for Soil and Groundwater Contamination"_, discusses the motivation behind this research, outlines the methodologies, and presents the results. It includes:
- An introduction to Monitored Natural Attenuation (MNA) and its importance for cost-effective remediation.
- Detailed methodology for the integration of linear regression, random forest, and LSTM models into PyLEnM.
- Case study using data from the Savannah River Site (SRS) F-Area.
- Discussion of results, including time-to-MCL predictions and feature importance analysis.

## Installation and Usage

### Requirements:
- Python 3.8+
- Jupyter Notebook
- Required Python libraries: 
  - pandas
  - numpy
  - scikit-learn
  - keras
  - matplotlib
  - folium
  - scipy

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/contaminant-attuation-prediction.git
   cd contaminant-attuation-prediction
   ```

2. Install the required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Notebooks:

#### 1. **Bidirectional LSTM Model**:
- Open the `Bidirectional_LSTM.ipynb` file in Jupyter Notebook.
- Run the notebook to train the LSTM model on your dataset and generate predictions for contaminant concentrations.

#### 2. **Random Forest Regression**:
- Open the `Random_Forest_Regression.ipynb` file in Jupyter Notebook.
- Run the notebook to train a Random Forest model and identify important factors affecting contaminant attenuation.

#### 3. **Linear Regression and Mapping**:
- Open the `Linear_Regression_+_Mapping.ipynb` file in Jupyter Notebook.
- Run the notebook to predict the time-to-MCL using linear regression and visualize the results on a map.

### Example Usage:
For detailed instructions on how to use each model, refer to the individual notebooks. Each notebook is designed to be self-contained and includes comments to guide you through the code.

### Citation
If you use this work, please cite:

**Vu Anh Le, Haruko Murakami Wainwright, Hansell Gonzalez-Raymat, Carol Eddy-Dilek**, _Extension of PyLEnM: Machine Learning Algorithms to Assess Site Closure Time Frames for Soil and Groundwater Contamination_.

### Acknowledgments
This project is supported by the Massachusetts Institute of Technology, the Savannah River National Laboratory, and the MIT Summer Research Program. Special thanks to Professor Haruko Murakami Wainwright for her guidance and support throughout the project.

### Instructions:
1. Copy and paste the above content into a new `README.md` file in your GitHub repository.
2. Adjust the repository link (currently `https://github.com/your-username/contaminant-attuation-prediction.git`) and author details as necessary.
3. Push the changes to your GitHub repository.

Once uploaded, this markdown file will be rendered properly on GitHub, providing a detailed overview and instructions for your project.
