import os
import pickle
import faiss
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def load_recommendation_system():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RECOMMENDATIONS_DIR = os.path.join(BASE_DIR, '../recommendations')
    DATA_DIR = '/home/sagemaker-user/Data/'
    
    # Load the SVD++ model
    model_path = os.path.join(RECOMMENDATIONS_DIR, 'SVD++_best_model.pkl')
    with open(model_path, 'rb') as f:
        recommendation_model = pickle.load(f)
    
    # Load user and item factors
    user_factors_path = os.path.join(RECOMMENDATIONS_DIR, 'user_factors.npy')
    item_factors_path = os.path.join(RECOMMENDATIONS_DIR, 'item_factors.npy')
    
    user_factors = np.load(user_factors_path)
    item_factors = np.load(item_factors_path)
    
    # Load user and item ID mappings
    user_id_map_path = os.path.join(RECOMMENDATIONS_DIR, 'user_id_map.pkl')
    item_id_map_path = os.path.join(RECOMMENDATIONS_DIR, 'item_id_map.pkl')
    
    with open(user_id_map_path, 'rb') as f:
        user_id_map = pickle.load(f)
    
    with open(item_id_map_path, 'rb') as f:
        item_id_map = pickle.load(f)
    
    # Load FAISS index
    faiss_index_path = os.path.join(RECOMMENDATIONS_DIR, 'item_index.faiss')
    index = faiss.read_index(faiss_index_path)
    
    # Load pre-generated recommendations
    recommendations_path = os.path.join(RECOMMENDATIONS_DIR, 'recommendations.h5')
    loaded_recommendations = {}
    with h5py.File(recommendations_path, 'r') as hf:
        for user_id in hf.keys():
            recommended_items = hf[user_id][:]
            # Decode bytes to strings if necessary
            recommended_items = [item.decode('utf-8') if isinstance(item, bytes) else item for item in recommended_items]
            loaded_recommendations[user_id] = recommended_items
    
    # Load product details
    filtered_data_path = os.path.join(DATA_DIR, 'filtered_data_unique_asin.pkl')
    if os.path.exists(filtered_data_path):
        filtered_data = pd.read_pickle(filtered_data_path)
    else:
        print("Product details file does not exist. Please ensure 'filtered_data_unique_asin.pkl' is in the DATA_DIR.")
        filtered_data = pd.DataFrame()
    
    # Load TF-IDF Vectorizer and matrix
    tfidf_vectorizer_path = os.path.join(RECOMMENDATIONS_DIR, 'tfidf_vectorizer.pkl')
    tfidf_matrix_path = os.path.join(RECOMMENDATIONS_DIR, 'tfidf_matrix.npz')
    
    with open(tfidf_vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
    
    return (recommendation_model, user_factors, item_factors,
            user_id_map, item_id_map, index, loaded_recommendations,
            filtered_data, tfidf_vectorizer, tfidf_matrix)

def recommend(user_id, user_factors, item_factors, user_id_map, item_id_map, index, loaded_recommendations, top_k=10):
    """
    Generate recommendations for a given user using collaborative filtering.
    
    Parameters:
    - user_id (str): The ID of the user.
    - user_factors (np.ndarray): User factor matrix.
    - item_factors (np.ndarray): Item factor matrix.
    - user_id_map (dict): Mapping from user IDs to indices.
    - item_id_map (dict): Mapping from item indices to IDs.
    - index (faiss.Index): FAISS index for efficient similarity search.
    - loaded_recommendations (dict): Pre-generated recommendations.
    - top_k (int): Number of top recommendations to return.
    
    Returns:
    - recommendations (list): List of recommended item ASINs.
    """
    # Check if the user is in the pre-generated recommendations
    if user_id in loaded_recommendations:
        recommendations = loaded_recommendations[user_id][:top_k]
    else:
        # If user is not in the pre-generated list, attempt to generate recommendations using user factors
        if user_id in user_id_map:
            user_idx = user_id_map[user_id]
            user_vector = user_factors[user_idx].reshape(1, -1).astype('float32')
            
            # Perform nearest neighbor search using FAISS
            _, item_indices = index.search(user_vector, top_k)
            item_indices = item_indices[0]
            reverse_item_id_map = {v: k for k, v in item_id_map.items()}
            recommendations = [reverse_item_id_map.get(idx) for idx in item_indices if idx in reverse_item_id_map]
        else:
            # If user is unknown, return an empty list
            recommendations = []
    
    return recommendations

def content_based_recommendation(user_keywords, tfidf_vectorizer, tfidf_matrix, filtered_data, top_k=5):
    """
    Generate content-based recommendations based on user-provided keywords.
    
    Parameters:
    - user_keywords (str): Keywords entered by the user.
    - tfidf_vectorizer (TfidfVectorizer): Pre-trained TF-IDF vectorizer.
    - tfidf_matrix (sparse matrix): Pre-computed TF-IDF matrix for all products.
    - filtered_data (pd.DataFrame): DataFrame containing product details.
    - top_k (int): Number of top recommendations to return.
    
    Returns:
    - recommended_asins (list): List of recommended product ASINs.
    """
    # Convert user input keywords to TF-IDF vector
    user_tfidf = tfidf_vectorizer.transform([user_keywords])
    # Compute cosine similarity with all products
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    # Get indices of top similar products
    top_indices = cosine_similarities.argsort()[-top_k:][::-1]
    # Retrieve corresponding ASINs
    recommended_asins = filtered_data.iloc[top_indices]['parent_asin'].tolist()
    return recommended_asins

def load_hot_products():
    """
    Load the top 5 hot products from a CSV file.
    
    Returns:
    - top_5_df (pd.DataFrame): DataFrame containing top 5 hot products.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RECOMMENDATIONS_DIR = os.path.join(BASE_DIR, '../recommendations')
    hot_products_path = os.path.join(RECOMMENDATIONS_DIR, 'top_5.csv')

    if not os.path.exists(hot_products_path):
        print("Hot products data file does not exist. Please ensure 'top_5.csv' is in the recommendations directory.")
        return pd.DataFrame()

    top_5_df = pd.read_csv(hot_products_path)
    return top_5_df

def get_product_details(filtered_data, asin):
    """
    Retrieve detailed information of a product based on its ASIN.
    
    Parameters:
    - filtered_data (pd.DataFrame): DataFrame containing product details.
    - asin (str): The ASIN of the product.
    
    Returns:
    - details (dict or None): Dictionary containing product details or None if not found.
    """
    product = filtered_data[filtered_data['parent_asin'] == asin]
    if not product.empty:
        product = product.iloc[0]
        details = {
            'ASIN': product['parent_asin'],
            'Description': ' '.join(product['description']) if isinstance(product['description'], list) else str(product['description']),
            'Details': ', '.join([f"{k}: {v}" for k, v in product['details'].items()]) if isinstance(product['details'], dict) else str(product['details']),
            'Categories': ' > '.join(product['categories']) if isinstance(product['categories'], list) else str(product['categories']),
            'Average Rating': product['average_rating'],
            'Number of Ratings': product['rating_number'],
            'Popularity Score': product['popularity_score']
        }
        return details
    else:
        return None
