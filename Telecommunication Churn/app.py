import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Preprocess input data
def preprocess_input(account_length, voice_plan, voice_messages, intl_plan, intl_mins, intl_calls, 
                     day_mins, day_calls, eve_mins, eve_calls, 
                     night_mins, night_calls,customer_calls):

    # Prepare the input data as a DataFrame
    data = pd.DataFrame({
        'account.length': [account_length],
        'voice.plan': [1 if voice_plan == 'yes' else 0],
        'voice.messages': [voice_messages],
        'intl.plan': [1 if intl_plan == 'yes' else 0],
        'intl.mins': [intl_mins],
        'intl.calls': [intl_calls],
        'day.mins': [day_mins],
        'day.calls': [day_calls],
        'eve.mins': [eve_mins],
        'eve.calls': [eve_calls],
        'night.mins': [night_mins],
        'night.calls': [night_calls],
        'customer.calls': [customer_calls],
    })
    
    # Select only scale the 11 features we need
    scaled_data = scaler.transform(data[['account.length','voice.plan','voice.messages','intl.plan', 'intl.mins','intl.calls','day.mins','day.calls','eve.mins','eve.calls','night.mins','night.calls',
'customer.calls']])
    
    # Return the scaled data
    return scaled_data

# Streamlit app
def main():
    st.title("Telecom Churn Prediction")
    st.write("Enter customer details to predict the likelihood of churn.")

    # Input fields for a single customer
    account_length = st.slider("Account Length", 1, 243, 100)
    voice_plan = st.selectbox("Voice Plan", ["yes", "no"])
    voice_messages = st.slider("Number of Voice Messages", 0, 52, 8)
    intl_plan = st.selectbox("International Plan", ["yes", "no"])
    intl_mins = st.slider("International Minutes", 0.0, 20.0, 10.3, step=0.1)
    intl_calls = st.slider("Number of International Calls", 0, 20, 4)
   
    day_mins = st.slider("Day Minutes", 0.0, 351.5, 180.1, step=0.1)
    day_calls = st.slider("Number of Day Calls", 0, 163, 100)
   
    eve_mins = st.slider("Evening Minutes", 0.0, 361.8, 200.6, step=0.1)
    eve_calls = st.slider("Number of Evening Calls", 0, 170, 100)

    night_mins = st.slider("Night Minutes", 0.0, 395.0, 200.6, step=0.1)
    night_calls = st.slider("Number of Night Calls", 0, 175, 100)

    customer_calls = st.slider("Customer Service Calls", 0, 5, 1,)

    if st.button("Predict"):
        # Preprocess and scale the input
        input_data = preprocess_input(account_length, voice_plan, voice_messages, intl_plan, 
                                      intl_mins, intl_calls, day_mins, day_calls, eve_mins, eve_calls, night_mins, 
                                      night_calls, customer_calls)
        
        print(input_data)

        # Make predictions
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        # st.success(prediction)

        if prediction == 1:
            st.success(f"The customer is predicted to **churn** with a probability of {prediction_proba:.2%}.")
        else:
            st.success(f"The customer is predicted **not to churn** with a probability of {(1 - prediction_proba):.2%}.")

if __name__ == "__main__":
    main()
