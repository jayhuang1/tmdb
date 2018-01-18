# Author:   Jay Huang
# E-mail: askjayhuang at gmail dot com
# GitHub: https://github.com/jayh1285
# Created:  2018-01-07T22:06:54.489Z

"""A module for ."""

################################################################################
# Imports
################################################################################

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from itertools import chain
from time import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression

################################################################################
# Global Constants
################################################################################

MOVIES_PATH = 'data/tmdb_5000_movies.csv'
CREDITS_PATH = 'data/tmdb_5000_credits.csv'

################################################################################
# Functions
################################################################################


def max_display_df():
    """Maximize DataFrame display options."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 175)


def prep_csv_df():
    """Prepare DataFrame from csv files."""
    df_movies = pd.read_csv(MOVIES_PATH)
    df_credits = pd.read_csv(CREDITS_PATH)
    df_merged = pd.merge(df_movies, df_credits, left_on='id', right_on='movie_id')

    df_wrangled = wrangle_df(df_merged)

    return df_wrangled


def wrangle_df(df):
    """Wrangle movies DataFrame."""
    # Drop columns
    df.drop(['title_x', 'title_y', 'id', 'movie_id', 'homepage', 'original_language', 'overview',
             'production_countries', 'spoken_languages', 'status', 'tagline', 'release_date'], axis=1, inplace=True)

    # Drop row with missing revenue data
    df = df[df.revenue != 0]

    # Convert strings into list of wanted values
    df.cast = [eval(x) for x in df.cast]
    for idx, movie in df.cast.iteritems():
        cast = []
        for character in movie:
            cast.append(character['name'])
        df.cast.set_value(idx, cast)

    df.crew = [eval(x) for x in df.crew]
    directors_all = []
    writers_all = []
    for idx, movie in df.crew.iteritems():
        directors = []
        writers = []
        for crew in movie:
            if crew['job'] == 'Director':
                directors.append(crew['name'])
            if crew['department'] == 'Writing':
                writers.append(crew['name'])
        directors_all.append(list(set(directors)))
        writers_all.append(list(set(writers)))
    df = df.assign(directors=directors_all, writers=writers_all)
    # df['directors'] = directors_all
    # df['writers'] = writers_all
    df.drop('crew', axis=1, inplace=True)

    df.genres = [eval(x) for x in df.genres]
    for idx, movie in df.genres.iteritems():
        genres = []
        for genre in movie:
            genres.append(genre['name'])
        df.genres.set_value(idx, genres)

    df.keywords = [eval(x) for x in df.keywords]
    for idx, movie in df.keywords.iteritems():
        keywords = []
        for keyword in movie:
            keywords.append(keyword['name'])
        df.keywords.set_value(idx, keywords)

    df.production_companies = [eval(x) for x in df.production_companies]
    for idx, movie in df.production_companies.iteritems():
        companies = []
        for company in movie:
            companies.append(company['name'])
        df.production_companies.set_value(idx, companies)

    # Re-arrange index and columns
    df = df.reindex(columns=['original_title', 'runtime', 'genres', 'keywords', 'production_companies',
                             'cast', 'directors', 'writers', 'vote_average', 'vote_count', 'popularity', 'budget', 'revenue'])
    df.set_index('original_title', drop=True, inplace=True)

    df.drop(['keywords', 'production_companies', 'cast', 'directors', 'writers'], axis=1, inplace=True)

    return df


def explore_data(df):
    """Explore frequency values of data."""
    cast = list(chain.from_iterable(df.cast))
    directors = list(chain.from_iterable(df.directors))
    writers = list(chain.from_iterable(df.writers))
    genres = list(chain.from_iterable(df.genres))
    keywords = list(chain.from_iterable(df.keywords))
    production_companies = list(chain.from_iterable(df.production_companies))

    print('Cast:', len(set(cast)))
    print('Directors:', len(set(directors)))
    print('Writers:', len(set(writers)))
    print('Genres:', len(set(genres)))
    print('Keywords:', len(set(keywords)))
    print('Production Companies:', len(set(production_companies)))

    sns.set()
    ax = sns.countplot(y=genres)
    plt.ylabel('Genre')
    plt.title('Genre Frequency')
    plt.show()

    formatter = FuncFormatter(millions_format)
    ax = sns.distplot(df.budget, kde=False)
    ax.xaxis.set_major_formatter(formatter)
    plt.title('Budget Distribution')
    plt.show()

    ax = sns.distplot(df.revenue, kde=False)
    ax.xaxis.set_major_formatter(formatter)
    plt.title('Revenue Distribution')
    plt.show()


def millions_format(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)


def multi_label_binarize(data):
    mlb = MultiLabelBinarizer()

    for feature in categorical_features:
        data[feature] = mlb.fit_transform(data[feature]).tolist()
        features = list(mlb.classes_)
        data[features] = pd.DataFrame(data[feature].values.tolist(), index=data.index)
        data.drop(feature, axis=1, inplace=True)

    return data

################################################################################
# Execution
################################################################################


if __name__ == '__main__':
    max_display_df()

    # Prepare data into DataFrame
    df = prep_csv_df()
    # explore_data(df)

    # Set X and y
    X = df.drop('revenue', axis=1)
    y = df.revenue

    # Find categorical features in DataFrame
    categorical_features = [column for column in X.columns if X[column].dtype == 'object']

    # Binarize categorical features
    X = multi_label_binarize(X)

    # Fit data into linear regression model
    regr = LinearRegression()
    regr.fit(X, y)

    # Format coefficients into human readable data
    fl = ["${:,.2f}".format(x) for x in list(regr.coef_)]

    for feature, coef in zip(X.columns, fl):
        print(feature, coef)

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train = multi_label_binarize(X_train)
    X_test = multi_label_binarize(X_test)

    regr = LinearRegression()

    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
