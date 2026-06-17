import streamlit as st
import pandas as pd
import pickle

# Load saved objects
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
full_feature_columns = pickle.load(open("columns.pkl", "rb"))

st.title("🚀 SpaceX Falcon 9 Landing Predictor")
st.markdown("Predict whether the Falcon 9 first stage will successfully land.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    flight_number = st.number_input("Flight Number", value=100)
    payload_mass = st.number_input("Payload Mass (kg)", value=5500.0)
    orbit = st.selectbox("Orbit", ['GTO', 'LEO', 'ISS', 'PO', 'SSO', 'VLEO', 'MEO', 'HEO', 'SO', 'GEO', 'ES-L1'])
    launch_site = st.selectbox("Launch Site", ['CCSFS SLC 40', 'KSC LC 39A', 'VAFB SLC 4E'])
    flights = st.number_input("Flights", value=1)
    grid_fins = st.checkbox("Grid Fins", value=True)

with col2:
    reused = st.checkbox("Reused", value=False)
    legs = st.checkbox("Legs", value=True)
    landing_pad = st.selectbox("Landing Pad", ['LZ-1', 'LZ-2', 'OCISLY', 'JRTI', 'None'])
    block = st.number_input("Block", value=5.0)
    reused_count = st.number_input("Reused Count", value=0.0)
    serial = st.text_input("Serial", value="B1049")

# Predict
if st.button("Predict"):
    input_dict = {
        'FlightNumber': flight_number,
        'PayloadMass': payload_mass,
        'Orbit': orbit,
        'LaunchSite': launch_site,
        'Flights': flights,
        'GridFins': grid_fins,
        'Reused': reused,
        'Legs': legs,
        'LandingPad': landing_pad,
        'Block': block,
        'ReusedCount': reused_count,
        'Serial': serial
    }

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=full_feature_columns, fill_value=0).fillna(0)

    # Skip imputer — go straight to scaler
    df_scaled = scaler.transform(df.values)

    prob = model.predict_proba(df_scaled)[0][1]
    cost = prob * 62e6 + (1 - prob) * 165e6
    risk = "Low" if prob >= 0.85 else "Medium" if prob >= 0.65 else "High"

    st.divider()
    col3, col4, col5 = st.columns(3)
    col3.metric("Landing Probability", f"{prob:.2%}")
    col4.metric("Expected Cost", f"${cost:,.0f}")
    col5.metric("Risk Level", risk)

    if risk == "Low":
        st.success("✅ High confidence of successful landing. SpaceX is a great choice!")
    elif risk == "Medium":
        st.warning("⚠️ Moderate confidence. Consider negotiating price or getting insurance.")
    else:
        st.error("❌ Low confidence. Consider a competitor or purchase additional insurance.")