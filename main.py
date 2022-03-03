import numpy as np
import pandas as pd
import os
import pickle
from google.cloud import storage
from scipy.sparse import csr_matrix

knn = None
movies = None
movie_matrix = None
sprse_mat = None
def download_model_file():

    from google.cloud import storage

    # Model Bucket details
    BUCKET_NAME        = "movie_model_2022"
    PROJECT_ID         = "unique-bebop-342715"
    GCS_MODEL_FILE     = "knnpickle_file.pkl"
    GCS_MOVIES_FILE     = "movie.csv"
    GCS_RATINGS_FILE     = "movie_matrix.csv"
    GCS_SPARS_FILE     = "spars"

    # Initialise a client
    client   = storage.Client(PROJECT_ID)
    
    # Create a bucket object for our bucket
    bucket   = client.get_bucket(BUCKET_NAME)
    
    # Create a blob object from the filepath
    blob     = bucket.blob(GCS_MODEL_FILE)
    blob2     = bucket.blob(GCS_MOVIES_FILE)
    blob3     = bucket.blob(GCS_RATINGS_FILE)
    blob4     = bucket.blob(GCS_SPARS_FILE)

    folder = '/tmp/'
    if not os.path.exists(folder):
      os.makedirs(folder)
    # Download the file to a destination
    blob.download_to_filename(folder + "local_model.pkl")
    blob2.download_to_filename(folder + "movie.csv")
    blob3.download_to_filename(folder + "movie_matrix.csv")
    blob4.download_to_filename(folder + "spars")

def get_movie_recommendation(request):

    
    # Use the global model variable 
    global knn
    global movies
    global movie_matrix
    global sprse_mat
    if not knn:
      download_model_file()
      knn = pickle.load(open("/tmp/local_model.pkl", 'rb'))
    
    if not movies:
      movies = pd.read_csv("/tmp/movie.csv", encoding_errors='ignore',  on_bad_lines='skip')

    if not movie_matrix:
      movie_matrix = pd.read_csv("/tmp/movie_matrix.csv", encoding_errors='ignore', on_bad_lines='skip')
      #sprse_mat = csr_matrix(movie_matrix)
      #movie_matrix.reset_index(inplace=True)
    if not sprse_mat:
      sprse_mat = pickle.load(open("/tmp/spars", 'rb'))
    

    request_json = request.get_json()
    if request.args and 'movie' in request.args:
      movie_name =  request.args.get('movie')
    elif request_json and 'movie' in request_json:
      movie_name =  request_json['movie']
    else:
      return "NO"

    # params = request.get_json()
    # if (params is not None) and ('movie' in params):
    #   movie_name = params['movie']
    # if params is None:
    #   return "NO"
    

    num_recommendations=10
    all_movies=movies[movies['title'].str.contains(movie_name.title())]
    if len(all_movies):
        idx=all_movies.iloc[0]['movieId']
        idx=movie_matrix[movie_matrix['movieId'] == idx].index[0]
        dist,idc = knn.kneighbors(sprse_mat[idx],n_neighbors=num_recommendations+1)
        mve_idc = sorted(list(zip(idc.squeeze().tolist(),dist.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for ele in mve_idc:
            idx = movie_matrix.iloc[ele[0]]['movieId']
            idx = movies[movies['movieId'] == idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0]})
        df = pd.DataFrame(recommend_frame)
        data = df["Title"].to_json(index=False, orient='split')
        return data
    else:
        return {"data": ["The Movie is not in the dataset"]}