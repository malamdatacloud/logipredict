import shap
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from streamlit_shap import st_shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
shap.initjs()

# Path to the saved model directory
model_path = 'C:/Users/User/Desktop/Projects/Fritz/PredictionApp/air/model_a2.h5'

# Load the model
model_a2 = tf.keras.models.load_model(model_path)

# Read Air No Features, better Accuracy, aka a2
a2 = pd.read_excel("C:/Users/User/Desktop/Projects/Fritz/PredictionApp/air/air_normal_no_4_features.xlsx")
a2.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)

df_a2 = pd.read_excel("C:/Users/User/Desktop/Projects/Fritz/PredictionApp/air/air_normal_no_4_features.xlsx")
df_a2.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)

int64_columns_a2 = a2.select_dtypes(include=['int'])
a2[int64_columns_a2.columns] = a2[int64_columns_a2.columns].astype('float32', copy=False)

X_a2, y_a2 = a2.drop(columns=['Shipping Company'], axis=1), a2['Shipping Company']

lookup_loading_country_a2 = tf.keras.layers.StringLookup()
lookup_loading_country_a2.adapt(X_a2['Loading Country'])

lookup_loading_port_a2 = tf.keras.layers.StringLookup()
lookup_loading_port_a2.adapt(X_a2['Loading Port'])

lookup_destination_port_a2 = tf.keras.layers.StringLookup()
lookup_destination_port_a2.adapt(X_a2['Destination Port'])

lookup_destination_country_a2 = tf.keras.layers.StringLookup()
lookup_destination_country_a2.adapt(X_a2['Destination Country'])

lookup_shipping_company_a2 = tf.keras.layers.StringLookup()
lookup_shipping_company_a2.adapt(y_a2)

def map_features_a2(row):
    # Apply StringLookup layers to the respective features within the dataset mapping function
    return {
        'input_loading_country': lookup_loading_country_a2(row['Loading Country']),
        'input_loading_port': lookup_loading_port_a2(row['Loading Port']),
        'input_destination_country': lookup_destination_country_a2(row['Destination Country']),
        'input_destination_port': lookup_destination_port_a2(row['Destination Port']),
        'input_numerical': [row['Hazard'], #1
                            row['Legs'],#2
                            row['Billable Weight'],#3
                            row['Gross Weight'], #4
                            row['Volume'], #5
                            row['Pack Qty'], #6
                            row['ATA_year'], #7
                            row['ATA_month'], #8
                            row['ATA_day'], #9
                            row['ATA_weekday'], #10
                            row['ATD_year'], #11
                            row['ATD_month'], #12
                            row['ATD_day'], #13
                            row['ATD_weekday'], #14
                            row['OnTimeArrival'], #15
                            row['DelayedDeparture'], #16
                            row['Time elapsed ATD-ETD'], #17
                            row['DeliveryDelay'], #18
                            row['EarlyDelivery'], #19
                            row['Average_Delay'], #20
                            row['On_Time_Percentage_per_Ship_Comp'], #21
                           ]}

def process_dataframe_a2(features_df, target_df):

    target_indices = lookup_shipping_company_a2(y_a2)
    dataset = tf.data.Dataset.from_tensor_slices((features_df.to_dict('list'), target_indices))
    dataset = dataset.map(lambda x, y: (map_features_a2(x), y))
    return dataset

# Apply conversion to dataset
full_dataset_a2 = process_dataframe_a2(X_a2, y_a2)
# Shuffle and batch the dataset
full_dataset_a2 = full_dataset_a2.shuffle(buffer_size=len(X_a2)).batch(32)

# Calculate the number of batches to split into training and validation
train_size_a2 = int(0.8 * len(X_a2))
val_size_a2 = len(X_a2) - train_size_a2

train_dataset_a2 = full_dataset_a2.take(train_size_a2 // 32)  # Use train_size divided by batch size
val_dataset_a2 = full_dataset_a2.skip(train_size_a2 // 32)

# Define the model with functional API to handle multiple inputs
input_loading_country_a2 = tf.keras.Input(shape=(1,), name='input_loading_country', dtype=tf.float32)
input_loading_port_a2 = tf.keras.Input(shape=(1,), name='input_loading_port', dtype=tf.float32)
input_destination_country_a2 = tf.keras.Input(shape=(1,), name='input_destination_country', dtype=tf.float32)
input_destination_port_a2 = tf.keras.Input(shape=(1,), name='input_destination_port', dtype=tf.float32)
input_numerical_a2 = tf.keras.Input(shape=(21,), name='input_numerical')

# Check for best output_dim based on input_dim:
vocabulary_size_loading_country_a2 = lookup_loading_country_a2.vocabulary_size()
vocabulary_size_loading_port_a2 = lookup_loading_port_a2.vocabulary_size()
vocabulary_size_destination_country_a2 = lookup_destination_country_a2.vocabulary_size()
vocabulary_size_destination_port_a2 = lookup_destination_port_a2.vocabulary_size()
vocabulary_size_shipping_a2 = lookup_shipping_company_a2.vocabulary_size()

loading_country_dim_a2 = int(math.sqrt(vocabulary_size_loading_country_a2))
loading_port_dim_a2 = int(math.sqrt(vocabulary_size_loading_port_a2))

destination_country_dim_a2 = int(math.sqrt(vocabulary_size_destination_country_a2))
destination_port_dim_a2 = int(math.sqrt(vocabulary_size_destination_port_a2))

shipping_dim_a2 = int(math.sqrt(vocabulary_size_shipping_a2))

# Embeddings for categorical inputs
loading_embedding_country_a2 = tf.keras.layers.Embedding(
    input_dim = lookup_loading_country_a2.vocabulary_size(),
    output_dim = loading_country_dim_a2)(input_loading_country_a2)

loading_embedding_port_a2 = tf.keras.layers.Embedding(
    input_dim = lookup_loading_port_a2.vocabulary_size(),
    output_dim = loading_port_dim_a2)(input_loading_port_a2)

destination_embedding_country_a2 = tf.keras.layers.Embedding(
    input_dim = lookup_destination_country_a2.vocabulary_size(),
    output_dim = destination_country_dim_a2)(input_destination_country_a2)

destination_embedding_port_a2 = tf.keras.layers.Embedding(
    input_dim = lookup_destination_port_a2.vocabulary_size(),
    output_dim = destination_port_dim_a2)(input_destination_port_a2)

# Flatten embeddings and concatenate with numerical inputs
loading_country_flat_a2 = tf.keras.layers.Flatten()(loading_embedding_country_a2)
loading_port_flat_a2 = tf.keras.layers.Flatten()(loading_embedding_port_a2)

destination_country_flat_a2 = tf.keras.layers.Flatten()(destination_embedding_country_a2)
destination_port_flat_a2 = tf.keras.layers.Flatten()(destination_embedding_port_a2)

concatenated_a2 = tf.keras.layers.Concatenate()([loading_country_flat_a2,
                                                    loading_port_flat_a2,
                                                    destination_country_flat_a2,
                                                    destination_port_flat_a2,
                                                    input_numerical_a2
                                                    ])

############ predict_top_companies_o2 ###########################


def predict_top_companies_a2(loading_port,loading_country,
                          destination_port,destination_country,legs):


    loading_port_encoded = lookup_loading_port_a2(tf.constant([loading_port]))
    loading_country_encoded = lookup_loading_country_a2(tf.constant([loading_country]))

    destination_port_encoded = lookup_destination_port_a2(tf.constant([destination_port]))
    destination_country_encoded = lookup_destination_country_a2(tf.constant([destination_country]))

    numerical_features = np.zeros((1,21))  #
    numerical_features[0, 20] = legs  #

    predictions_a2 = model_a2.predict([
        loading_port_encoded,
        loading_country_encoded,
        destination_port_encoded,
        destination_country_encoded,
        numerical_features
        ])

    # Find the indices of the top 3 predictions
    top_indices = np.argsort(predictions_a2[0])[-3:][::-1]
    top_confidences = [predictions_a2[0][i] for i in top_indices]

    # Adjust confidences based on model's overall accuracy
    model_accuracy =  0.8832
    adjusted_confidences = [conf * model_accuracy for conf in top_confidences]

    # Get the shipping companies names
    top_companies = [lookup_shipping_company_a2.get_vocabulary()[i] for i in top_indices]

    # Create a results table
    results_table = pd.DataFrame({
        'Predicted Shipping Company': top_companies,
        'Confidence': [f"{conf * 100:.2f}%" for conf in adjusted_confidences]
    })
    return results_table

############ process_and_visualize_o2 ###########################

for col in ['Loading Port', 'Loading Country', 'Destination Port', 'Destination Country']:
    le = LabelEncoder()
    df_a2[col] = le.fit_transform(df_a2[col])

le_company = LabelEncoder()
df_a2['Encoded Shipping Company'] = le_company.fit_transform(df_a2['Shipping Company'])

# Store the mapping from encoded labels back to original strings
encoded_to_company_a2 = dict(zip(df_a2['Encoded Shipping Company'], df_a2['Shipping Company']))

# Assuming encoded_to_company is {encoded_value: 'company_name'}
company_to_encoded_a2 = {v: k for k, v in encoded_to_company_a2.items()}

#@st.cache_data
def process_and_visualize_a2(company_name, df, company_to_encoded_a2):
    if company_name in company_to_encoded_a2:
        encoded_label = company_to_encoded_a2[company_name]
        
        # Create a binary target column
        y = (df_a2['Encoded Shipping Company'] == encoded_label).astype(int)
        X = df.drop(['Shipping Company', 'Encoded Shipping Company'], axis=1)
    
        # Train a RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

         # Sample X for SHAP values calculation
        X_sample = X.sample(n=1000, random_state=42)
    
        # SHAP values calculation
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sample)
    
        # Visualize the SHAP values for the positive class
        return st_shap(shap.summary_plot(shap_values[1], X_sample, show=False))
    

def integrated_prediction_and_visualization_a2(loading_port, loading_country, destination_port, destination_country, legs):
    # Get predictions
    result_table = predict_top_companies_a2(loading_port, loading_country, destination_port, destination_country, legs)
    
    df, company_to_encoded = df_a2, company_to_encoded_a2
    
    shap_figures = []
    max_items = 4
    for company_name in result_table['Predicted Shipping Company']:
        if len(shap_figures) < max_items:
            fig = process_and_visualize_a2(company_name, df, company_to_encoded)
            shap_figures.append(fig)
        else:
            break
          
    return result_table, shap_figures








































