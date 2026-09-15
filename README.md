This repository contains a data cleaning and exploratory data analysis (EDA) pipeline for a used car dataset. 

The script processes raw data, identifies missing values, and strictly handles price and mileage outliers using the Interquartile Range (IQR) method to prevent skewed visualizations. It generates data distribution charts and outputs a clean, machine-learning-ready CSV file.

To use this pipeline, install the packages by running `pip install -r requirements.txt` and then execute the main python script.