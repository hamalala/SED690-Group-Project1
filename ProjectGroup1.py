
import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpmax
from mlxtend.frequent_patterns import association_rules

def predict(rules, input_items):
    results = rules[rules['antecedents'].apply(lambda x: x.issubset(input_items))]
    return results[['antecedents', 'consequents', 'confidence', 'lift']]
    
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

# Create a button
if st.button("Run Algorithm"):
    # Actions to perform when button is clicked
    url = 'https://docs.google.com/spreadsheets/d/1-h4q2swBlPGO76pJkExdtIoVoMyoePjkovyk71c-PzA/export?format=csv'
    df = pd.read_csv(url, header=None)
    transactions = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    if algorithm == 'Apriori Algorithm':
        frequent_itemsets = apriori(df_encoded, min_support=min_support/len(transactions), use_colnames=True)
        st.write("Frequent Itemsets using Apriori:")
        st.dataframe(frequent_itemsets)

    else:
        frequent_itemsets = fpmax(df_encoded, min_support=min_support/len(transactions), use_colnames=True)
        st.write("Frequent Itemsets using FP-Growth:")
        st.dataframe(frequent_itemsets)
        
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules[rules['confidence'] >= min_confidence]
    rules = rules[rules['lift'] > min_lift]

    st.write("### Association Rules")
    st.dataframe(rules)  # or use st.table(rules) for a static table

    # Text input widget for entering a comma-separated string
    input_string = st.text_input("Enter items (comma-separated) for prediction:", "Monitor")
# Check if rules exist in session state
if 'rules' in st.session_state:
    # Text input widget for entering a comma-separated string
    input_string = st.text_input("Enter items (comma-separated) for prediction:", "Monitor")
    # Convert the input string to a list of strings
    input_items = frozenset([item.strip() for item in input_string.split(',')])
    prediction = predict(st.session_state['rules'], input_items)
    
    if not prediction.empty:
        st.write("### Prediction Results")
        st.dataframe(prediction)
    else:
        st.write("### No strong association rules found for the given items.")
