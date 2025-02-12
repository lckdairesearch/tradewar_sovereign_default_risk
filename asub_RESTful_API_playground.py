"""
https://restcountries.com/#api-endpoints-using-this-project
"""
# %% Setup
import os
import json
import re
import requests


import numpy as np
import pandas as pd

# %% CONSTANTS
RESTful_API_BASE_URL = "https://restcountries.com/v3.1/all"

# %% RESTful API
def get_restful_country_info():
    """
    Fetch country information from the RESTful API for the specified country.

    Returns:
        pd.DataFrame: Country information as a pandas DataFrame.
    """

    request_url = f"{RESTful_API_BASE_URL}?fields=name,cca2,cca3,ccn3,region,subregion"
    response = requests.get(request_url)
    response.raise_for_status()

    country_info = (
        pd.DataFrame(response.json())
        .pipe(lambda x: x['name'].apply(pd.Series).rename(columns=lambda y: y+"_name").join(x))
        .drop(columns=['nativeName_name', 'name']))
    
    # Data harmonization
    country_info = (country_info
        .rename(columns=lambda x: re.sub(r"^[^a-zA-Z0-9]+", "", x))
        .rename(columns=lambda x: re.sub(r"[^a-zA-Z0-9]+", "_", x))
    )

    return country_info

# %% Run
# see whether file country_info.csv exist with os
if not os.path.exists("../data/processed/country_info.csv"):
    country_info = get_restful_country_info()
    country_info.to_csv("../data/processed/country_info.csv", index=False, encoding="utf-8")
