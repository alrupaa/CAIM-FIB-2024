import pandas as pd
import utils as ut

def naive_recommender(ratings: object, movies: object, k: int = 10) -> list: 
    # Provide the code for the naive recommender here. This function should return 
    # the list of the top most viewed films according to the ranking (sorted in descending order).
    # Consider using the utility functions from the pandas library.
    most_seen_movies= []

    average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index() # Calculo la media de puntuación de cada película
    movie_ratings = average_ratings.merge(movies, on='movieId')                 # hago un merge para saber el nombre de las películas
    top_movies = movie_ratings.sort_values(by='rating', ascending=False)        # Ordeno las películas según su puntuación decrecientemente
    most_seen_movies = top_movies[['movieId', 'rating']]

    return list(most_seen_movies.itertuples(index=False, name=None))


if __name__ == "__main__":
    path_to_ml_latest_small = 'D:\\Datos\\CARRERA\\5o semestre\\CAIM\\entrega9\\ml-latest-small'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
    top_movies = naive_recommender(ratings, movies)

    print("Top 10 recommended movies:")
    for idx, (movie,rating) in enumerate(top_movies, 1):
        print(f"{idx}. {movie} ({rating})")