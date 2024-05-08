from keras.models import load_model
# import tensorflow as tf
# from tf.keras.models import load_model
from datetime import datetime
from tabulate import tabulate
import pandas as pd
import numpy as np
import joblib
import json
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def nice_console_print(message):
    line_width = 60
    padding = (line_width - len(message)) // 2
    print("\n")
    print("=" * line_width)
    print(" " * padding + message)
    print("=" * line_width)


def get_valid_user_id():
    user_id_in = input("\nPlease enter user ID: ")

    while True:
        if user_id_in.lower() == 'e' or user_id_in.lower() == 'ex' or user_id_in.lower() == 'exi' or user_id_in.lower() == 'exit':
            print("Exiting...")
            quit()

        if user_id_in.isdigit() and int(user_id_in) > 0:
            user_id_in = int(user_id_in)
            break
        else:
            user_id_in = input("Please enter a valid user ID (or type 'e' to exit): ")

    return user_id_in


def prepare_movies(user_id, recsys_config):
    try:
        # Load ratings from .csv file
        ratings = pd.read_csv(recsys_config['ratings_only_userid_movieid'])

        # Filter ratings for the given user_id
        user_df = ratings[ratings['userId'] == user_id]

        # Extract only unique movie_ids - i.e. movies which the user had rated
        rated_movie_ids = user_df['movieId'].unique()

        # Load movies from .csv file
        movies = pd.read_csv(recsys_config['movies_context'])

        # Get movies that were not rated by the user
        non_rated_movies = movies[~movies['movieId'].isin(rated_movie_ids)]

        del ratings, movies, rated_movie_ids, user_df
        return non_rated_movies

    except OSError as er1:
        print('\nERROR loading movies_with_context in main.py')
        print(er1)
        quit()


# Helper function for creating holiday information to time context
def is_date_in_interval(date, start_date, end_date, holiday_name):
    if holiday_name == 'new_years':
        if date.month == 12:
            if date.day >= start_date.day:
                return True

        elif date.month == 1:
            if date.day <= end_date.day:
                return True
        return False
    else:
        if start_date.month <= date.month <= end_date.month:
            if start_date.day <= date.day <= end_date.day:
                return True
        return False


# Helper function for creating holiday information to time context
def find_holiday(date, holiday_dates):
    for holiday_name, interval in holiday_dates.items():
        start_date = datetime.strptime(interval['start'], '%m-%d')
        end_date = datetime.strptime(interval['end'], '%m-%d')
        if is_date_in_interval(date, start_date, end_date, holiday_name):
            return holiday_name
    return 'no_holiday'


def add_time_context(movies_data, recsys_config):
    try:
        time_context = []

        ts = time.time()
        date_now = datetime.fromtimestamp(ts)

        """ Create a week day value from timestamp
        0: Monday
        1: Tuesday
        2: Wednesday
        3: Thursday
        4: Friday
        5: Saturday
        6: Sunday
        """
        day = datetime.fromtimestamp(ts).isoweekday()
        time_context.append(day)

        """ Create a isWeekday value from day column
        0: false / weekend
        1: true  / weekday
        """
        if day == 6 or day == 7:
            time_context.append(0)
        else:
            time_context.append(1)

        """ Create a season value
        1: Spring	
        2: Summer
        3: Fall
        4: Winter
        """
        month = date_now.month
        if 3 <= month <= 5:
            time_context.append(1)
        elif 6 <= month <= 8:
            time_context.append(2)
        elif 9 <= month <= 11:
            time_context.append(3)
        else:
            time_context.append(4)

        """ Create a partOfDay value
        1 - Morning
        2 - Afternoon
        3 - Evening
        4 - Night
        """
        hour = date_now.hour
        if 5 <= hour < 12:
            time_context.append(1)
        elif 12 <= hour < 17:
            time_context.append(2)
        elif 17 <= hour < 21:
            time_context.append(3)
        else:
            time_context.append(4)

        # Load .json file fwith holiday dates
        with open(recsys_config['holidays'], 'r') as json_file:
            holidays = json.load(json_file)

        """ Create a holiday value
        More info in data/holidays.json
        """
        time_context.append(find_holiday(date_now, holidays))

        # Add current time context to each row of unrated movie with context
        non_rated_movies_copy = movies_data.copy()
        non_rated_movies_copy.loc[:, 'day'] = time_context[0]
        non_rated_movies_copy.loc[:, 'isWeekday'] = time_context[1]
        non_rated_movies_copy.loc[:, 'season'] = time_context[2]
        non_rated_movies_copy.loc[:, 'partOfDay'] = time_context[3]
        non_rated_movies_copy.loc[:, 'holiday'] = time_context[4]

        del movies_data
        return non_rated_movies_copy

    except Exception as er2:
        print('\nERROR while creating time context in main.py')
        print(er2)
        quit()


def add_uder_id_and_order_columns(user_id, tc_movies):

    tc_movies['userId'] = user_id

    new_order = ['userId', 'movieId', 'day', 'isWeekday', 'season', 'partOfDay', 'holiday', 'movieYear', 'titleType',
                 'isAdult', 'runtimeMinutes', 'directors', 'actor', 'genreAction', 'genreAdult', 'genreAdventure',
                 'genreAnimation', 'genreBiography', 'genreChildren', 'genreComedy', 'genreCrime', 'genreDocumentary',
                 'genreDrama', 'genreFamily', 'genreFantasy', 'genreFilm-noir', 'genreHistory', 'genreHorror',
                 'genreImax', 'genreMusic', 'genreMusical', 'genreMystery', 'genreNews', 'genreReality-tv',
                 'genreRomance', 'genreSci-fi', 'genreShort', 'genreSport', 'genreThriller', 'genreWar', 'genreWestern']

    tc_movies = tc_movies[new_order]
    return tc_movies


def transform_data(data, recsys_config):
    try:

        # Load label encoders
        actor_label_encoder = joblib.load(recsys_config['actor_label_encoder'])
        directors_label_encoder = joblib.load(recsys_config['directors_label_encoder'])
        holiday_label_encoder = joblib.load(recsys_config['holiday_label_encoder'])
        titleType_label_encoder = joblib.load(recsys_config['titleType_label_encoder'])

        # Load scaler
        scaler = joblib.load(recsys_config['scaler'])

        # Label encode data
        data['actor'] = actor_label_encoder.transform(data['actor'])
        data['directors'] = directors_label_encoder.transform(data['directors'])
        data['holiday'] = holiday_label_encoder.transform(data['holiday'])
        data['titleType'] = titleType_label_encoder.transform(data['titleType'])

        # Scale data
        new_data = scaler.transform(data)

        del data
        return new_data

    except Exception as er3:
        print('\nERROR loading scaler and label encoders')
        print(er3)
        quit()


def predict_ratings(data_to_predict_on, recsys_config):

    try:
        # nn_model = load_model(recsys_config['model'], compile=True)
        # model_path = "model/arch8_25m_added_imdb_context_max_abs_scaler_checkpoint.h5"
        # model_path = "model/arch8_25m_added_imdb_context_max_abs_scaler_run2_trained.keras"
        # nn_model = tf.keras.models.load_model(model_path)

        nn_model = load_model(recsys_config['model'], compile=True)
        # nn_model = load_model(model_path, compile=True)

        predictions = nn_model.predict(data_to_predict_on, verbose=0)

        return predictions
    except Exception as er4:
        print('\nERROR loading model')
        print(er4)
        quit()


def load_movies_with_info(recsys_config):

    return pd.read_csv(recsys_config['movie_titles'])


def recommend_movies(predictions, top_k, not_rated_movies, all_movies_uncut, count):

    top_k_movies = []
    recommendations = []

    # set interval of which top_k predictions to show
    total_elements = predictions.flatten().shape[0]
    start_index = total_elements - (top_k * (count + 1))
    end_index = total_elements - (top_k * count)

    top_k_movie_indices = np.argsort(predictions.flatten())[start_index:end_index][::-1]

    for movie_index in top_k_movie_indices:
        top_k_movies.append({
            'movieId': not_rated_movies.iloc[movie_index]['movieId'],
            'movieRating': predictions.flatten()[movie_index]
        })

    for index, movie in enumerate(top_k_movies):
        movie_info = all_movies_uncut[all_movies_uncut['movieId'] == movie['movieId']]
        one_recommended_movie = [f"{index + 1 + (top_k * count)}.", str(movie_info['title'].iloc[0]), str(movie['movieRating'])]
        recommendations.append(one_recommended_movie)

    header = ["#", "Movie title", "Predicted rating"]

    return recommendations, header


def recommned_more_titles(question):
    while True:
        user_input = input(question + " (Y/n) ").strip().lower()
        if user_input in ["yes", "y", "ye"]:
            return True
        else:
            return False


if __name__ == '__main__':

    try:

        nice_console_print("Movie Recommender System")

        valid_user_id = get_valid_user_id()
        k = 10

        with open('recsys_congif.json') as config_file:
            config = json.load(config_file)

        print('\nRecommending movies...\n')

        movies_not_rated_by_user = prepare_movies(valid_user_id, config)
        # print('1')
        movies_with_time_context = add_time_context(movies_not_rated_by_user, config)
        # print('2')
        recsys_data = add_uder_id_and_order_columns(valid_user_id, movies_with_time_context)
        # print('3')
        transformed_data = transform_data(recsys_data, config)
        # print('4')
        predicted_ratins = predict_ratings(transformed_data, config)
        # print('5')
        all_movies = load_movies_with_info(config)
        # print('6')]
        
        recommend_more = True
        counter = -1    # it will start at zero
        while recommend_more:
            counter = counter + 1
            recommended_movies, headers = recommend_movies(predicted_ratins, k, movies_not_rated_by_user, all_movies, counter)

            print("Recommended movies:")
            print(tabulate(recommended_movies, headers=headers, tablefmt="rounded_outline"))

            recommend_more = recommned_more_titles("Recommend another movies?")

    except OSError as er_main:
        print('\nERROR loading config file in main.py')
        print(er_main)
        quit()
