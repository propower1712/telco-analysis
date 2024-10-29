import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model from a file
with open('models/telco_churn_rf.pkl', 'rb') as f:
    model = pickle.load(f)

features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
       'Partner_Yes', 'Dependents_Yes', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No internet service',
       'TechSupport_Yes', 'Contract_One year', 'Contract_Two year',
       'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

# Function to make predictions
def predict_churn(customer_data):
    prediction = model.predict(customer_data)
    proba = model.predict_proba(customer_data)[:, 1]
    return prediction, proba

def shap_plot(customer_data):
    explainer = shap.TreeExplainer(model)

    # Compute Shapley values for the specific customer
    shap_values = explainer.shap_values(customer_data)
    print(customer_data.iloc[0].values)
    print(features)
    shap_df = pd.DataFrame({"Feature" : features, "Values" : customer_data.iloc[0].values, "Shapley_Value" : shap_values[1].tolist()[0]})
    shap_df["Feature"] = shap_df["Feature"] + "=" + shap_df["Values"].astype(str)
    del shap_df["Values"]
    shap_df = shap_df.sort_values(by = "Shapley_Value", ascending=False).head(5)
    shap_df = shap_df.loc[shap_df["Shapley_Value"] > 0]
    # Plot the bar plot
    plt.figure(figsize=(5, 3))
    sns.barplot(x='Shapley_Value', y='Feature', data=shap_df)
    plt.xlabel('Shapley Value')
    plt.ylabel('Feature')
    plt.title('Contribution of Each Feature in Churn Score')
    return plt

# Main function to run the Streamlit app
def main():
    st.title('Telco Churn Prediction')

    # Sidebar with user input options
    st.sidebar.header('Customer Input Features')

    # Collect user inputs
    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    tenure = st.sidebar.number_input('Tenure (months)', min_value=0, max_value=100, value=1)
    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=50.0, step=0.01)
    total_charges = st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=2000.0, step=1.0)
    senior_citizen = st.sidebar.checkbox('Senior Citizen')
    partner = st.sidebar.checkbox('Partner')
    dependents = st.sidebar.checkbox('Dependents')
    paperless_billing = st.sidebar.checkbox('Paperless Billing')
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.sidebar.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.sidebar.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)'])


    # Create a DataFrame with the user input data
    user_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'SeniorCitizen' : [senior_citizen],
        'Partner_Yes' : [partner],
        'Dependents_Yes' : [dependents],
        'InternetService_Fiber optic' : [True if internet_service == "Fiber optic" else False],
        'InternetService_No' : [True if internet_service == "No" else False],
        'OnlineSecurity_No internet service' : [True if online_security == "No internet service" else False],
        'OnlineSecurity_Yes' : [True if online_security == "Yes" else False],
        'OnlineBackup_No internet service' : [True if online_backup == "No internet service" else False],
        'OnlineBackup_Yes' : [True if online_backup == "Yes" else False],
        'DeviceProtection_No internet service' : [True if device_protection == "No internet service" else False],
        'DeviceProtection_Yes' : [True if device_protection == "Yes" else False],
        'TechSupport_No internet service' : [True if tech_support == "No internet service" else False],
        'TechSupport_Yes' : [True if tech_support == "Yes" else False],
        'Contract_One year' : [True if contract == "One year" else False],
        'Contract_Two year' : [True if contract == "Two year" else False],
        'PaperlessBilling_Yes' : [True if paperless_billing == "Yes" else False],
        'PaymentMethod_Credit card (automatic)' : [True if payment_method == "Credit card (automatic)" else False],
        'PaymentMethod_Electronic check' : [True if payment_method == "Electronic check" else False],
        'PaymentMethod_Mailed check' : [True if payment_method == "Mailed check" else False],
    })

    user_data = user_data[features]

    # Make prediction
    if st.sidebar.button('Predict'):
        prediction, churn_proba = predict_churn(user_data)
        if prediction[0] == 1:
            st.error('Customer is likely to churn.')
        else:
            st.success('Customer is not likely to churn.')
        st.info('probability of churn {}'.format(churn_proba))


        st.subheader('SHAP Summary Plot')
        st.pyplot(shap_plot(user_data))

if __name__ == '__main__':
    main()