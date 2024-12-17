import numpy as np
import streamlit as st
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import euclidean_distances
scaler = StandardScaler()
warnings.filterwarnings('ignore')

# --------- Load Data ------------
data = pd.read_csv('output_with_tags.csv')
df = pd.read_csv('output_with_tags.csv')

# --------- Data Cleaning and Preprocessing ------------
def combine_cat(row, weights=[1, 0, 1]):
    combined_features = ' '.join([row['tags']] * weights[0])
    combined_features += ' ' + ' '.join([f"job_type_{row['job_type']}"] * weights[1])
    combined_features += ' ' + ' '.join([f"experience_level_{row['experience_level']}"] * weights[2])
    return combined_features

def process_data(df, weights=[1, 0, 1], alpha=0.9):
    scaler = MinMaxScaler()
    df['hourly_rate'] = np.where(df['job_type'] == 'Hourly', df['lower_range'] + df['higher_range'] / 2, np.nan)
    df.loc[df['budget'].notna(), 'budget'] = scaler.fit_transform(df.loc[df['budget'].notna(), ['budget']])
    df.loc[df['hourly_rate'].notna(), 'hourly_rate'] = scaler.fit_transform(df.loc[df['hourly_rate'].notna(), ['hourly_rate']])
    # Fill NaN values with empty strings and preprocess
    df['tags'] = df['tags'].fillna('')  # Handle missing values in tags
    df['combined_features'] = df.apply(combine_cat, axis=1)
    return df

def get_similarity(df, weights=[1, 0, 1], alpha=0.9):

    # Create TF-IDF matrix
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['combined_features'])

    # Numerical features
    numerical_features = ['hourly_rate', 'budget']  # Replace with desired features
    num_features = df[numerical_features].fillna(0).values

    # Compute cosine similarity for text features
    text_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Compute Euclidean distances for numerical features
    numerical_distances = euclidean_distances(num_features)

    # Convert Euclidean distances to similarities
    # You can use an exponential decay function or normalize:
    numerical_similarity = 1 / (1 + numerical_distances)  # Similarity = 1 / (1 + distance)

    # Combine similarities with weights
    alpha = 0.9  # Weight for text similarity
    beta = 1 - alpha  # Weight for numerical similarity
    combined_similarity = alpha * text_similarity + beta * numerical_similarity
    return combined_similarity

# Recommendation function
def get_recommendations(data, index, similarity_matrix):
    if index < 0 or index >= len(data):
        return f"Index {index} is out of range."
    sim_scores = list(enumerate(similarity_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Exclude the input itself
    job_indices = [i[0] for i in sim_scores]
    return data.iloc[job_indices]

# --------- process Data ------------

processed_data = process_data(df)
similarity_matrix = get_similarity(processed_data)

