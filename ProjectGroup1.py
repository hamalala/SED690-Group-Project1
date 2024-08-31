
import streamlit as st

# Set title
st.title("Discovering Frequent Patterns and Association Rules")

st.markdown('[Can edit this sheet for you transaction](https://docs.google.com/spreadsheets/d/1-h4q2swBlPGO76pJkExdtIoVoMyoePjkovyk71c-PzA)')
# Define the radio button options
options = ['Apriori Algorithm', 'FP-Growth Algorithm']
# Create a radio button widget
algorithm = st.radio("Choose an Algorithm:", options)

# Creating a range slider
min_support = st.slider(
    "Select minimum Support:",
    min_value=2,  # Minimum value of the slider
    max_value=20,  # Maximum value of the slider
    value=3,  # Default value of the slider
    step=1  # Step size between values
)

# Creating a range slider
min_confidence = st.slider(
    "Select minimum Confidence:",
    min_value=0.0,  # Minimum value of the slider
    max_value=1.0,  # Maximum value of the slider
    value=0.5,  # Default value of the slider
    step=0.05  # Step size between values
)

# Creating a range slider
min_lift = st.slider(
    "Select minimum Lift:",
    min_value=0.0,  # Minimum value of the slider
    max_value=10.0,  # Maximum value of the slider
    value=1.0,  # Default value of the slider
    step=0.1  # Step size between values
)

# Display prediction result
st.write(f"## Prediction Result: {algorithm}")
st.write(f"## Prediction min_support: {min_support}")
st.write(f"## Prediction min_confidence: {min_confidence}")
st.write(f"## Prediction min_lift: {min_lift}")
