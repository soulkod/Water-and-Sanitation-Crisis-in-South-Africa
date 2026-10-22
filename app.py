import streamlit as st
import joblib
import pandas as pd

# Load the saved pipeline securely
@st.cache_resource
def load_pipeline():
    return joblib.load('locked_pipeline.pkl')

pipeline = load_pipeline()

st.title("SA Municipal Water & Sanitation Prioritization Tool")
st.write("### Decision-Support System for Proactive Municipal Infrastructure Audits")

# User Input Form (Question 7.1 & 7.2)
with st.form("prediction_form"):
    st.subheader("Enter Household Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prov = st.selectbox("Province", [
            "1. Western Cape", "2. Eastern Cape", "3. Northern Cape", 
            "4. Free State", "5. KwaZulu-Natal", "6. North West", 
            "7. Gauteng", "8. Mpumalanga", "9. Limpopo"
        ])
        geotype = st.selectbox("Geography Type", ["1. Urban", "2. Traditional", "3. Farms"])
        hsg_maind = st.selectbox("Main Dwelling Type", [
            "1. Formal dwelling", "2. Informal dwelling", "3. Traditional dwelling"
        ])
        
    with col2:
        head_sex = st.selectbox("Head of Household Sex", ["1. Male", "2. Female"])
        hholdsz_clean = st.number_input("Household Size", min_value=1, max_value=30, value=4, step=1)
        totmhinc_clean = st.number_input("Total Monthly Household Income (ZAR)", min_value=0, value=3500, step=500)
    
    submitted = st.form_submit_button("Assess Household Vulnerability")

# Prediction Execution and Validation (Question 7.2 & 7.3)
if submitted:
    try:
        # Wrap input into DataFrame matching training feature columns
        input_data = pd.DataFrame({
            'prov': [prov],
            'geotype': [geotype],
            'hsg_maind': [hsg_maind],
            'hholdsz_clean': [float(hholdsz_clean)],
            'totmhinc_clean': [float(totmhinc_clean)],
            'head_sex': [head_sex]
        })
        
        # Run through saved pipeline
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        confidence = probabilities[prediction] * 100
        
        st.divider()
        st.subheader("Assessment Results")
        
        if prediction == 1:
            st.error(f"🔴 **High Vulnerability / Service Deficit Detected** (Confidence: {confidence:.1f}%)\n\n"
                     f"**Plain-Language Explanation:** This household exhibits risk indicators consistent with non-piped water sources or unsafe sanitation facilities[cite: 15]. \n\n"
                     f"👉 **Actionable Recommendation:** Flag this household/ward for priority municipal infrastructure inspection and mobile water tanker dispatch[cite: 15].")
        else:
            st.success(f"🟢 **Adequate Access Expected** (Confidence: {confidence:.1f}%)\n\n"
                       f"**Plain-Language Explanation:** This household's profile indicates standard access to piped water and flush/VIP sanitation facilities[cite: 15]. \n\n"
                       f"👉 **Actionable Recommendation:** Low priority for immediate emergency intervention.")
                       
    except Exception as e:
        st.error(f"⚠️ **Error in processing input:** {e}. Please ensure all fields are valid.")