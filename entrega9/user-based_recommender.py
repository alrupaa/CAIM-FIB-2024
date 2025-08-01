import pandas as pd
import numpy as np

import similarity as sim
import naive_recommender as nav
import utils as ut
from scipy.spatial.distance import cosine
import time


n = 20
k = 5

def generate_m(movies_idx, users, ratings):
    # Complete the datastructure for rating matrix 
    m = pd.DataFrame(index=users, columns=movies_idx)

    for _, row in ratings.iterrows():
        m.at[row['userId'], row['movieId']] = row['rating']
    
    m.fillna(value=np.nan, inplace=True)  # Pongo como nan aquellas celdas que no tienen calificaciones

    return m


def user_based_recommender(target_user_idx, matrix):
    target_user = matrix.iloc[target_user_idx]
    
    # Compute the similarity between the target user and each other user in the matrix. 
    similarities = []
    for idx, other_user in matrix.iterrows():
        if idx != target_user_idx:
            sim_value = sim.compute_similarity(target_user, other_user)
            similarities.append((idx, sim_value))
    
    similaritiesDF = pd.DataFrame(similarities, columns=["userId","similarity"])

    # Selección de los k vecinos más similares
    similaritiesDF.sort_values(by='similarity', ascending=False, inplace=True)
    top_n_users = similaritiesDF.head(n)
    
    # Determine the unseen movies by the target user. Those films are identified 
    unseen_movies = target_user[target_user.isna()].index.tolist()
     
    # Generate recommendations for unrated movies based on user similarity and ratings.
    user_actual_mean = target_user.mean()
    recommendations = []

    for movie in unseen_movies:
        numerador = 0
        denominador = 0
        for idx, row in top_n_users.iterrows():
            user_idx, sim_value = row["userId"], row["similarity"]
            if not pd.isna(matrix.loc[user_idx, movie]):  # Si el usuario ha visto la película
                other_user_mean = matrix.loc[user_idx].mean() if user_idx in matrix.index else 0  # Validación para evitar errores
                numerador += sim_value * (matrix.loc[user_idx, movie] - other_user_mean)  # Desviación de puntuación con multiplicador según similitud
                denominador += abs(sim_value)

        if denominador > 0:
            pred = user_actual_mean + numerador / denominador
            recommendations.append((movie, pred))
        else:
            recommendations.append((movie, 0.0))

    recommendations.sort(key=lambda x: x[1], reverse=True)  # Ordenar por interés en orden decreciente

    return recommendations



# Calcula un vector de proporciones de cada género dentro de un conjunto de películas
def genre_frequencies(genres_matrix, movie_ids):
    genre_counts = genres_matrix.loc[movie_ids].sum(axis=0)  # Sumamos por columna
    return genre_counts / genre_counts.sum()  # Normalizmos para obtener proporciones

# Compara lo buenas que son las recomendaciones de ambos sistemas de recomendación para un usuario
def compare(user_id, rec1, rec2, k):
    genres_matrix = ut.matrix_genres(dataset["movies.csv"])
    
    top_k_rec1 = [movie for movie, _ in rec1[:k]]
    top_k_rec2 = [movie for movie, _ in rec2[:k]]
    user_validation_movies = ratings_val[ratings_val["userId"] == user_id]["movieId"]

    freq_rec1 = genre_frequencies(genres_matrix, top_k_rec1)
    freq_rec2 = genre_frequencies(genres_matrix, top_k_rec2)
    freq_user = genre_frequencies(genres_matrix, user_validation_movies)

    """
    print(freq_rec1)
    print(freq_rec2)
    print(freq_user)
    """
    
    # Para obtener la similitud calculamos el coseno del ángulo de los vectores.
    similarity_rec1 = 1 - cosine(freq_rec1, freq_user)
    similarity_rec2 = 1 - cosine(freq_rec2, freq_user)
    
    return similarity_rec1, similarity_rec2


if __name__ == "__main__":
    
    # Load the dataset
    path_to_ml_latest_small = 'D:\\Datos\\CARRERA\\5o semestre\\CAIM\\entrega9\\ml-latest-small'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)
    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]


    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)

    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    m = generate_m(movies_idx, users_idy, ratings_train)

    # user-to-user similarity
    target_users_idx = [123, 467, 607]
    results = {}
     
    for user_id in target_users_idx:
        first_time = time.time()
        rec1 = user_based_recommender(user_id, m)
        second_time = time.time()
        rec2 = nav.naive_recommender(ratings, movies)
        third_time = time.time()

        print(f"User-To-User lasted: {second_time-first_time}")
        print(f"Naive lasted: {third_time-second_time}")

        """
        print("Top 5 recomendaciones del User-Based Recommender:")
        for movie, score in rec1[:5]:
            print(f"Movie: {movie}, Score: {score}")

        print("\nTop 5 recomendaciones del Naive Recommender:")
        for movie, score in rec2[:5]:
            print(f"Movie: {movie}, Score: {score}")
        """

        similarity_rec1, similarity_rec2 = compare(user_id, rec1, rec2, k)
        results[user_id] = (similarity_rec1, similarity_rec2)

    for user_id, (sim_rec1, sim_rec2) in results.items():
        print(f"User {user_id}: Rec1 Similarity: {sim_rec1}, Rec2 Similarity: {sim_rec2}")



    """ 
    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))
    """
    

