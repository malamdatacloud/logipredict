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
model_path = 'C:/Users/User/Desktop/Projects/Fritz/PredictionApp/ocean/model_o2.h5'

# Load the model
model_o2 = tf.keras.models.load_model(model_path)

# Read Air No Features, better Accuracy, aka a2
o2 = pd.read_excel("C:/Users/User/Desktop/Projects/Fritz/PredictionApp/ocean/ocean_normal_no_4_features.xlsx")
o2.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)

df_o2 = pd.read_excel("C:/Users/User/Desktop/Projects/Fritz/PredictionApp/ocean/ocean_normal_no_4_features.xlsx")
df_o2.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)

int64_columns_o2 = o2.select_dtypes(include=['int'])
o2[int64_columns_o2.columns] = o2[int64_columns_o2.columns].astype('float32', copy=False)

X_o2, y_o2 = o2.drop(columns=['Shipping Company'], axis=1), o2['Shipping Company']

lookup_loading_country_o2 = tf.keras.layers.StringLookup()
lookup_loading_country_o2.adapt(X_o2['Loading Country'])

lookup_loading_port_o2 = tf.keras.layers.StringLookup()
lookup_loading_port_o2.adapt(X_o2['Loading Port'])

lookup_destination_port_o2 = tf.keras.layers.StringLookup()
lookup_destination_port_o2.adapt(X_o2['Destination Port'])

lookup_destination_country_o2 = tf.keras.layers.StringLookup()
lookup_destination_country_o2.adapt(X_o2['Destination Country'])

lookup_shipping_company_o2 = tf.keras.layers.StringLookup()
lookup_shipping_company_o2.adapt(y_o2)

def map_features_o2(row):
    # Apply StringLookup layers to the respective features within the dataset mapping function
    return {
        'input_loading_country': lookup_loading_country_o2(row['Loading Country']),
        'input_loading_port': lookup_loading_port_o2(row['Loading Port']),
        'input_destination_country': lookup_destination_country_o2(row['Destination Country']),
        'input_destination_port': lookup_destination_port_o2(row['Destination Port']),
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
                            row['ETAAccuracy'], #21
                            row['On_Time_Percentage_per_Ship_Comp'], #22
                            row['Amount Containers 20'], #23
                            row['Amount Containers 40'], #24
                           ]}

def process_dataframe_o2(features_df, target_df):

    target_indices = lookup_shipping_company_o2(y_o2)
    dataset = tf.data.Dataset.from_tensor_slices((features_df.to_dict('list'), target_indices))
    dataset = dataset.map(lambda x, y: (map_features_o2(x), y))
    return dataset

# Apply conversion to dataset
full_dataset_o2 = process_dataframe_o2(X_o2, y_o2)
# Shuffle and batch the dataset
full_dataset_o2 = full_dataset_o2.shuffle(buffer_size=len(X_o2)).batch(32)

# Calculate the number of batches to split into training and validation
train_size_o2 = int(0.8 * len(X_o2))
val_size_o2 = len(X_o2) - train_size_o2

train_dataset_o2 = full_dataset_o2.take(train_size_o2 // 32)  # Use train_size divided by batch size
val_dataset_o2 = full_dataset_o2.skip(train_size_o2 // 32)

# Define the model with functional API to handle multiple inputs
input_loading_country_o2 = tf.keras.Input(shape=(1,), name='input_loading_country', dtype=tf.float32)
input_loading_port_o2 = tf.keras.Input(shape=(1,), name='input_loading_port', dtype=tf.float32)
input_destination_country_o2 = tf.keras.Input(shape=(1,), name='input_destination_country', dtype=tf.float32)
input_destination_port_o2 = tf.keras.Input(shape=(1,), name='input_destination_port', dtype=tf.float32)
input_numerical_o2 = tf.keras.Input(shape=(24,), name='input_numerical')


# Check for best output_dim based on input_dim:
vocabulary_size_loading_country_o2 = lookup_loading_country_o2.vocabulary_size()
vocabulary_size_loading_port_o2 = lookup_loading_port_o2.vocabulary_size()
vocabulary_size_destination_country_o2 = lookup_destination_country_o2.vocabulary_size()
vocabulary_size_destination_port_o2 = lookup_destination_port_o2.vocabulary_size()
vocabulary_size_shipping_o2 = lookup_shipping_company_o2.vocabulary_size()

loading_country_dim_o2 = int(math.sqrt(vocabulary_size_loading_country_o2))
loading_port_dim_o2 = int(math.sqrt(vocabulary_size_loading_port_o2))

destination_country_dim_o2 = int(math.sqrt(vocabulary_size_destination_country_o2))
destination_port_dim_o2 = int(math.sqrt(vocabulary_size_destination_port_o2))

shipping_dim_o2 = int(math.sqrt(vocabulary_size_shipping_o2))

# Embeddings for categorical inputs
loading_embedding_country_o2 = tf.keras.layers.Embedding(
    input_dim = lookup_loading_country_o2.vocabulary_size(),
    output_dim = loading_country_dim_o2)(input_loading_country_o2)

loading_embedding_port_o2 = tf.keras.layers.Embedding(
    input_dim = lookup_loading_port_o2.vocabulary_size(),
    output_dim = loading_port_dim_o2)(input_loading_port_o2)

destination_embedding_country_o2 = tf.keras.layers.Embedding(
    input_dim = lookup_destination_country_o2.vocabulary_size(),
    output_dim = destination_country_dim_o2)(input_destination_country_o2)

destination_embedding_port_o2 = tf.keras.layers.Embedding(
    input_dim = lookup_destination_port_o2.vocabulary_size(),
    output_dim = destination_port_dim_o2)(input_destination_port_o2)

# Flatten embeddings and concatenate with numerical inputs
loading_country_flat_o2 = tf.keras.layers.Flatten()(loading_embedding_country_o2)
loading_port_flat_o2 = tf.keras.layers.Flatten()(loading_embedding_port_o2)

destination_country_flat_o2 = tf.keras.layers.Flatten()(destination_embedding_country_o2)
destination_port_flat_o2 = tf.keras.layers.Flatten()(destination_embedding_port_o2)

concatenated_o2 = tf.keras.layers.Concatenate()([loading_country_flat_o2,
                                                    loading_port_flat_o2,
                                                    destination_country_flat_o2,
                                                    destination_port_flat_o2,
                                                    input_numerical_o2
                                                    ])

def predict_top_companies_o2(loading_port,loading_country,
                          destination_port,destination_country,legs):
    
    loading_port_encoded = lookup_loading_port_o2(tf.constant([loading_port]))
    loading_country_encoded = lookup_loading_country_o2(tf.constant([loading_country]))

    destination_port_encoded = lookup_destination_port_o2(tf.constant([destination_port]))
    destination_country_encoded = lookup_destination_country_o2(tf.constant([destination_country]))

    numerical_features = np.zeros((1,24))  #
    numerical_features[0, 23] = legs  #

    predictions_o2 = model_o2.predict([
        loading_port_encoded,
        loading_country_encoded,
        destination_port_encoded,
        destination_country_encoded,
        numerical_features
        ])

    # Find the indices of the top 3 predictions
    top_indices = np.argsort(predictions_o2[0])[-3:][::-1]
    top_confidences = [predictions_o2[0][i] for i in top_indices]

    # Adjust confidences based on model's overall accuracy
    model_accuracy =  0.8360
    adjusted_confidences = [conf * model_accuracy for conf in top_confidences]

    # Get the shipping companies names
    top_companies = [lookup_shipping_company_o2.get_vocabulary()[i] for i in top_indices]

    # Create a results table
    results_table = pd.DataFrame({
        'Predicted Shipping Company': top_companies,
        'Confidence': [f"{conf * 100:.2f}%" for conf in adjusted_confidences],
        
    })

    return results_table



for col in ['Loading Port', 'Loading Country', 'Destination Port', 'Destination Country']:
    le = LabelEncoder()
    df_o2[col] = le.fit_transform(df_o2[col])

le_company = LabelEncoder()
df_o2['Encoded Shipping Company'] = le_company.fit_transform(df_o2['Shipping Company'])
encoded_to_company_o2 = dict(zip(df_o2['Encoded Shipping Company'], df_o2['Shipping Company']))
company_to_encoded_o2 = {v: k for k, v in encoded_to_company_o2.items()}



#@st.cache_data
def process_and_visualize_o2(company_name, df, company_to_encoded_o2):

    #encoded_to_company_o2 = dict(zip(df_o2['Encoded Shipping Company'], df_o2['Shipping Company']))
    #company_to_encoded_o2 = {v: k for k, v in encoded_to_company_o2.items()}
    if company_name in company_to_encoded_o2:
        encoded_label = company_to_encoded_o2[company_name]
        
        # Create a binary target column
        y = (df['Encoded Shipping Company'] == encoded_label).astype(int)
        X = df.drop(['Shipping Company', 'Encoded Shipping Company'], axis=1)
    
        # Train a RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        # Sample X for SHAP values calculation
        X_sample = X.sample(n=1000, random_state=42)
    
        # SHAP values calculation
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sample)
    
        # Generate the SHAP summary plot
        return st_shap(shap.summary_plot(shap_values[1], X_sample, show=False))





def integrated_prediction_and_visualization_o2(loading_port, loading_country, destination_port, destination_country, legs):
    # Get predictions
    result_table = predict_top_companies_o2(loading_port, loading_country, destination_port, destination_country, legs)
    df, company_to_encoded = df_o2, company_to_encoded_o2
    
    shap_figures = []
    for company_name in result_table['Predicted Shipping Company']:
        fig = process_and_visualize_o2(company_name, df, company_to_encoded)
        shap_figures.append(fig)
          
    return result_table, shap_figures






































































