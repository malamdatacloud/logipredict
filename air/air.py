import shap
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
import category_encoders as ce
from streamlit_shap import st_shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

shap.initjs()

# Path to the saved model directory
model_path = 'C:/Users/User/Desktop/Projects/Fritz/PredictionApp/air/model_air.h5'

# Load the model
model_air = tf.keras.models.load_model(model_path)

air = pd.read_csv("C:/Users/User/Desktop/Projects/Fritz/PredictionApp/air/data_air.csv", 
                  low_memory=False)

int64_columns_df = air.select_dtypes(include=['int', 'float'])
air[int64_columns_df.columns] = air[int64_columns_df.columns].astype('float32', copy=False)

air['Routes'] = air['Routes'].fillna('Missing')
air['Routes'] = air['Routes'].astype(str)
air.drop(air[air['Routes'] == "Missing"].index, inplace=True)

X = air.drop(columns=['Shipping Company'], axis=1)
y = air['Shipping Company']

# Initialize and adapt StringLookUp layers for categorical columns
lookup_loading_country = tf.keras.layers.StringLookup()
lookup_loading_country.adapt(X['Loading Country'])

lookup_loading_port = tf.keras.layers.StringLookup()
lookup_loading_port.adapt(X['Loading Port'])

lookup_destination_country = tf.keras.layers.StringLookup()
lookup_destination_country.adapt(X['Destination Country'])

lookup_destination_port = tf.keras.layers.StringLookup()
lookup_destination_port.adapt(X['Destination Port'])

lookup_routes = tf.keras.layers.StringLookup()
lookup_routes.adapt(X['Routes'])

lookup_shipping_company = tf.keras.layers.StringLookup()
lookup_shipping_company.adapt(y)

# Create the TensorFlow dataset
def map_features(row):
    # Apply StringLookup layers to the respective features within the dataset mapping function
    return {
        'input_loading_country': lookup_loading_country(row['Loading Country']),
        'input_loading_port': lookup_loading_port(row['Loading Port']),
        'input_destination_country': lookup_destination_country(row['Destination Country']),
        'input_destination_port': lookup_destination_port(row['Destination Port']),
        'input_routes': lookup_routes(row['Routes']),
        'input_numerical': [row['Hazard'],
                            row['Amount Containers 20'],
                            row['Amount Containers 40'],
                            row['Billable Weight'],
                            row['Gross Weight'],
                            row['Goods Value Shipment'],
                            row['Delay'],
                            row['LateEarly'],
                            row['ATA_month'],
                            row['ATA_day'],
                            row['ATA_weekday'],
                            row['ATD_month'],
                            row['ATD_day'],
                            row['ATD_weekday'],
                            row['DeliveryDelay'],
                            row['EarlyDelivery'],
                            row['ETAAccuracy'],
                            row['Average_Delay'],
                            row['On_Time_Percentage'],
                            row['Average_Delay_Per_Route'],
                            row['Num of Legs'],
                            row['OnTimeArrival'],
                            row['Time elapsed ATA-ATD'],
                            row['Time elapsed ATD-ETD'],
                            row['Time elapsed ETA-ETD'],
                           ]}

def process_dataframe(features_df, target_df):
    target_indices = lookup_shipping_company(y)
    dataset = tf.data.Dataset.from_tensor_slices((features_df.to_dict('list'), target_indices))
    dataset = dataset.map(lambda x, y: (map_features(x), y))
    return dataset

# Apply conversion to dataset
full_dataset = process_dataframe(X, y)
# Shuffle and batch the dataset
full_dataset = full_dataset.shuffle(buffer_size=len(X)).batch(32)

# Calculate the number of batches to split into training and validation
train_size = int(0.8 * len(X))
val_size = len(X) - train_size

train_dataset = full_dataset.take(train_size // 32)  # Use train_size divided by batch size
val_dataset = full_dataset.skip(train_size // 32)

# Define the model with functional API to handle multiple inputs
input_loading_country = tf.keras.Input(shape=(1,), name='input_loading_country', dtype=tf.float32)
input_loading_port = tf.keras.Input(shape=(1,), name='input_loading_port', dtype=tf.float32)
input_destination_country = tf.keras.Input(shape=(1,), name='input_destination_country', dtype=tf.float32)
input_destination_port = tf.keras.Input(shape=(1,), name='input_destination_port', dtype=tf.float32)
input_routes = tf.keras.Input(shape=(1,), name='input_routes', dtype=tf.float32)
input_numerical = tf.keras.Input(shape=(25,), name='input_numerical')

vocabulary_size_loading_country = lookup_loading_country.vocabulary_size()
vocabulary_size_loading_port = lookup_loading_port.vocabulary_size()
vocabulary_size_destination_country = lookup_destination_country.vocabulary_size()
vocabulary_size_destination_port = lookup_destination_port.vocabulary_size()
vocabulary_size_routes = lookup_routes.vocabulary_size()
vocabulary_size_shipping = lookup_shipping_company.vocabulary_size()

loading_country_dim = int(math.sqrt(vocabulary_size_loading_country))
loading_port_dim = int(math.sqrt(vocabulary_size_loading_port))

destination_country_dim = int(math.sqrt(vocabulary_size_destination_country))
destination_port_dim = int(math.sqrt(vocabulary_size_destination_port))

routes_dim = int(math.sqrt(vocabulary_size_routes))

# Embeddings for categorical inputs
loading_embedding_country = tf.keras.layers.Embedding(
    input_dim = lookup_loading_country.vocabulary_size(),
    output_dim = loading_country_dim)(input_loading_country)

loading_embedding_port = tf.keras.layers.Embedding(
    input_dim = lookup_loading_port.vocabulary_size(),
    output_dim = loading_port_dim)(input_loading_port)

destination_embedding_country = tf.keras.layers.Embedding(
    input_dim = lookup_destination_country.vocabulary_size(),
    output_dim = destination_country_dim)(input_destination_country)

destination_embedding_port = tf.keras.layers.Embedding(
    input_dim = lookup_destination_port.vocabulary_size(),
    output_dim = destination_port_dim)(input_destination_port)

routes_embedding = tf.keras.layers.Embedding(
    input_dim = lookup_routes.vocabulary_size(),
    output_dim = routes_dim)(input_routes)

# Flatten embeddings and concatenate with numerical inputs
loading_country_flat = tf.keras.layers.Flatten()(loading_embedding_country)
loading_port_flat = tf.keras.layers.Flatten()(loading_embedding_port)

destination_country_flat = tf.keras.layers.Flatten()(destination_embedding_country)
destination_port_flat = tf.keras.layers.Flatten()(destination_embedding_port)

routes_flat = tf.keras.layers.Flatten()(routes_embedding)

concatenated = tf.keras.layers.Concatenate()([loading_country_flat,
                                              loading_port_flat,
                                              destination_country_flat,
                                              destination_port_flat,
                                              routes_flat,
                                              input_numerical
                                              ])

##############
##############

def predict_top_companies(loading_port,
                          loading_country,
                          destination_port,
                          destination_country,
                          route,
                          leg):
  
  # Convert input to TensorFlow tensors
  loading_port_encoded = lookup_loading_port(tf.constant([loading_port]))
  loading_country_encoded = lookup_loading_country(tf.constant([loading_country]))
  destination_port_encoded = lookup_destination_port(tf.constant([destination_port]))
  destination_country_encoded = lookup_destination_country(tf.constant([destination_country]))
  routes_encoded = lookup_routes(tf.constant([route]))

  numerical_input = np.zeros((1, 25))
  numerical_input[0, 24] = leg

  predictions = model_air.predict([loading_country_encoded,
                                   loading_port_encoded,
                                   destination_country_encoded,
                                   destination_port_encoded,
                                   routes_encoded,
                                   numerical_input
                                    ])

  # Find indices for top 3 predictions
  top_indices = np.argsort(predictions[0])[-3:][::-1]
  top_companies = [lookup_shipping_company.get_vocabulary()[i] for i in top_indices]
  confidence_scores = [predictions[0][i] * 100 for i in top_indices]

  # Return as a DataFrame
  result_df = pd.DataFrame({
        'Predicted Shipping Companies': top_companies,
        'Confidence Score': [f"{score:.2f}%" for score in confidence_scores]
    })
  result_df.index = result_df.index + 1
  
  return result_df

#####################
#####################

# Copy of DF:
df = air.copy()
for col in ['Loading Port', 
            'Loading Country', 
            'Destination Port', 
            'Destination Country',
            'Routes']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
le_company = LabelEncoder()
df['Encoded Shipping Company'] = le_company.fit_transform(df['Shipping Company'])

# Store the mapping from encoded labels back to original strings
encoded_to_company = dict(zip(df['Encoded Shipping Company'], df['Shipping Company']))

# Assuming encoded_to_company is {encoded_value: 'company_name'}
company_to_encoded = {v: k for k, v in encoded_to_company.items()}

#@st.cache_data

def process_and_visualize(company_name):
  if company_name in company_to_encoded:
      encoded_label = company_to_encoded[company_name]
      
      # Create a binary target column
      y = (df['Encoded Shipping Company'] == encoded_label).astype(int)
      X = df.drop(['Shipping Company', 'Encoded Shipping Company'], axis=1)

      X_train, X_test, y_train, y_test = train_test_split(X, 
                                                          y, 
                                                          test_size=0.2, 
                                                          random_state=42
                                                          )
  
      # Train a RandomForestClassifier
      clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
      clf.fit(X_train, y_train)

        # Sample X for SHAP values calculation
      X_test_sample = X_test.sample(n=1000, random_state=42)
  
      # SHAP values calculation
      explainer = shap.TreeExplainer(clf)
      shap_values = explainer.shap_values(X_test_sample)
  
      # Visualize the SHAP values for the positive class
      st_shap(shap.summary_plot(shap_values, X_test_sample))


