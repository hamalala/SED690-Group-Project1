import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def predict(rules, input_items):
    results = rules[rules['antecedents'].apply(lambda x: x.issubset(input_items))]
    return results[['antecedents', 'consequents', 'confidence', 'lift']]

# Set title
st.title("Discovering Frequent Patterns and Association Rules")
st.write("Group: 704-710-712-714")
st.markdown('[Can edit this sheet for your transaction](https://docs.google.com/spreadsheets/d/1-h4q2swBlPGO76pJkExdtIoVoMyoePjkovyk71c-PzA)')

# Define the radio button options
options = ['Apriori Algorithm', 'FP-Growth Algorithm']
algorithm = st.radio("Choose an Algorithm:", options)

# Creating sliders
min_support = st.slider("Select minimum Support:", min_value=2, max_value=20, value=3, step=1)
min_confidence = st.slider("Select minimum Confidence:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
min_lift = st.slider("Select minimum Lift:", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Create a button
if st.button("Run Algorithm"):
    url = 'https://docs.google.com/spreadsheets/d/1-h4q2swBlPGO76pJkExdtIoVoMyoePjkovyk71c-PzA/export?format=csv'
    df = pd.read_csv(url, header=None)
    transactions = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
    
    # Check if there are any transactions to process
    if not transactions:
        st.write("No transactions found in the data.")
    else:
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        if algorithm == 'Apriori Algorithm':
            frequent_itemsets = apriori(df_encoded, min_support=min_support/len(transactions), use_colnames=True)
            st.session_state['algorithm_title'] = "Frequent Itemsets using Apriori:"
        else:
            frequent_itemsets = fpgrowth(df_encoded, min_support=min_support/len(transactions), use_colnames=True)
            st.session_state['algorithm_title'] = "Frequent Itemsets using FP-Growth:"

        st.session_state['algorithm_df'] = frequent_itemsets
        
        if frequent_itemsets.empty:
            st.write("No frequent itemsets found with the given minimum support.")
        else:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules = rules[rules['confidence'] >= min_confidence]
            rules = rules[rules['lift'] > min_lift]

            

            if rules.empty:
                st.write("No strong association rules found with the given minimum confidence and lift.")
            else:
		st.session_state['rules'] = rules
                st.write(st.session_state['algorithm_title'])
                st.dataframe(st.session_state['algorithm_df'])
                st.write("### Association Rules")
                st.dataframe(st.session_state['rules'])

# Predict section
if 'rules' in st.session_state and not st.session_state['rules'].empty:
    input_string = st.text_input("Enter items (comma-separated) for prediction:", "Monitor")
    if st.button("Predict"):
        input_items = frozenset([item.strip() for item in input_string.split(',')])
        prediction = predict(st.session_state['rules'], input_items)
        
        if not prediction.empty:
            st.write("### Prediction Results")
            st.dataframe(prediction)
        else:
            st.write("### No strong association rules found for the given items.")
