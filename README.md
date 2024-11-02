# Telco Churn Prediction and Client Segmentation

This project demonstrates a complete Data Science workflow for analyzing and predicting customer churn, as well as segmenting clients in the telecommunications sector. It includes a Jupyter Notebook and a Streamlit application, designed to give insights into churn patterns and customer segmentation using interpretable machine learning techniques. The project follows the CRISP-DM methodology, showcasing each step in a standard Data Science process.

## Project Structure

- **Telco-Analysis.ipynb**: Jupyter Notebook containing data exploration, preprocessing, churn prediction model development, and customer segmentation analysis. See full notebook [here](Telco-Analysis.ipynb)
- **telco-streamlit/**: Folder containing a Streamlit app that provides an interactive dashboard for exploring churn insights, including reasons for churn visualized with SHAP values. You can find the streamlit app in [this folder](telco-streamlit)

## CRISP-DM Phases

1. **Business Understanding**  
   The goal is to understand drivers of customer churn and identify distinct customer segments.

2. **Data Collection**  
   The dataset was collected from [Kaggle public datasets](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

3. **Data Understanding**  
   The dataset was explored to analyze patterns and relationships between features related to customer behavior, services, and demographics.

4. **Data Preparation**  
   Steps include data cleaning, handling missing values, feature engineering, and encoding categorical variables.

5. **Modelisation**  
   - **Churn Prediction**: Built predictive models (e.g., Logistic Regression, Decision Trees, Random Forest) to classify customers likely to churn.
   - **Client Segmentation**: Performed clustering techniques to segment customers based on behavioral and demographic factors.

6. **Evaluation**  
   - Evaluated model performance using metrics such as accuracy, precision, recall, F1-score and ROC AUC score for churn prediction.
   - Clustering quality was assessed to ensure meaningful segmentation.

7. **Deployment**  
   A Streamlit dashboard app was developed for user-friendly interaction and visualization of model results.

## Streamlit App Features

- **Dashboard for Churn Analysis**  
   Users can explore the impact of various factors on churn likelihood through interactive visualizations, supported by SHAP (SHapley Additive exPlanations) values to explain model predictions at both global and individual levels.

## Acknowledgments

This project was developed to demonstrate a structured Data Science approach, utilizing interpretable machine learning methods for practical insights.
