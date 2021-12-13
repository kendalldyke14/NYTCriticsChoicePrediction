'''
This script grabs movie metadata for a current year from New York Times Movie Review API and OMDB API.

DEPENDENCIES: config.py file with New York Times API key and OMDB API key named nyt_key and omdb_key, respectively

OUTPUT: data/NYTData{year}.csv, data/OMDBData{year}.csv

'''

import time

import pandas as pd
import requests
import json

from NYTCriticsChoicePrediction import config


def get_nyt_movies(key, year):
    nyt_data = []  # all movie data from NYT API return
    titles = []  # list of only titles, used as query for OMDB data
    has_more = True  # check if another request is required to get all results
    offset = 0
    while has_more:
        # make API request
        params = {"offset": offset, "publication-date": f"{year}-01-01:{year}-12-31",
                  "api-key": key}
        r = requests.get('https://api.nytimes.com/svc/movies/v2/reviews/search.json', params=params)

        # load data as Python dictionary
        data = json.loads(r.text)

        # offset query by an additional 20 and check if an additional iteration is needed
        offset += 20
        has_more = data['has_more']

        # add full data and title names to respective lists
        nyt_data.extend(data['results'])
        titles.extend([x['display_title'] for x in data['results']])

        # ensure the request per minute limit is not reached
        time.sleep(6)

    # convert to dataframe and write to CSV
    nyt_df = pd.DataFrame(nyt_data)
    nyt_df.to_csv(f"data/NYTData{year}.csv")

    return nyt_data, titles


def get_details_from_omdb(key, year, titles):
    omdb_data = []
    for title in titles:
        # request data for given title
        omdb_params = {"apikey": key, "t": title, "y": year, "plot": "full"}
        omdb_baseurl = "http://www.omdbapi.com/"
        r = requests.get(omdb_baseurl, params=omdb_params)
        data = json.loads(r.text)

        # if the movie was not found, it may be in the OMDB system under a different year
        # try taking out the year parameter
        if json.loads(r.text)["Response"] == "False":
            omdb_params = {"apikey": key, "t": title, "plot": "full"}
            r = requests.get(omdb_baseurl, omdb_params)
            data = json.loads(r.text)

        # if the OMDB finds a matching movie, add the dictionary to a list
        if data["Response"] == "True":
            omdb_data.append(data)

    # convert to dataframe and write to CSV
    omdb_df = pd.DataFrame(omdb_data)
    omdb_df.to_csv(f"data/OMDBData{year}.csv")

    return omdb_data


if __name__ == '__main__':
    nyt_key = config.nyt_key
    omdb_key = config.omdb_key
    year = 2020  # Change year here!

    nyt_results, movie_titles = get_nyt_movies(nyt_key, year)
    omdb_results = get_details_from_omdb(omdb_key, year, movie_titles)
