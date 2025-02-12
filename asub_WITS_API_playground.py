# %% Setup
import os
import re
import requests
import inspect
from itertools import chain
from typing import Literal

import numpy as np
import pandas as pd
import xmltodict
import inflect
import inflection


p = inflect.engine()

# %% CONSTANTS
WITS_META_BASE_URL = "https://wits.worldbank.org/API/V1/WITS"
WITS_DATA_BASE_URL = "https://wits.worldbank.org/API/V1/SDMX/V21"

# %% Country Info
if os.path.exists("../data/processed/country_info.csv"):
    country_info = pd.read_csv("../data/processed/country_info.csv")
else:
    # execute RESTful_API_playground.py 
    os.system("python RESTful_API_playground.py")
    country_info = pd.read_csv("../data/processed/country_info.csv")

# %% Functions
def convert_country_code(country, from_, to_):
    try:
        country_info = globals()["country_info"]
    except KeyError:
        country_info = pd.read_csv("country_info.csv")
        globals()["country_info"] = country_info


def validate_paramval_isin_meta(datasource, param, check_param, param_info_df_name, meta_fetcher, validation_field):
    param = "ALL" if param.upper() == "ALL" else param
    if (
        param != "ALL" and 
        not (check_param == "country" and param == "000")
        ):
        try:
            info = globals()[param_info_df_name]
        except KeyError:
            info = meta_fetcher(**{"datasource": datasource, check_param: "ALL"})
            globals()[param_info_df_name] = info
        assert all(
            item in getattr(info, validation_field).tolist() for item in param.split(";")
        ), f"Invalid {check_param} code, must be a valid {check_param} code separated by ';'"

    

# %% WITS UNCTAD TRAINS
def get_wits_TRAINS_meta(country=None, nomenclature=None, product=None, datasource="TRAINS"):

    country = str(country)
    country = country.upper()
    if country is not None:
        validate_paramval_isin_meta(
            datasource="TRAINS", param=country, check_param="country", 
            param_info_df_name="unctadtrains_country_info", 
            meta_fetcher=get_wits_TRAINS_meta, validation_field="ccn3")
    if nomenclature is not None:
        nomenclature = nomenclature.upper()
        validate_paramval_isin_meta(
            datasource="TRAINS", param=nomenclature, check_param="nomenclature",
            param_info_df_name="unctadtrains_nomenclature_info",
            meta_fetcher=get_wits_TRAINS_meta, validation_field="nomenclaturecode" )
    if product is not None:
        product = product
        validate_paramval_isin_meta(
            datasource="TRAINS", param=product, check_param="product",
            param_info_df_name="unctadtrains_product_info",
            meta_fetcher=get_wits_TRAINS_meta, validation_field="productcode")
    assert sum(param is not None for param in [country, nomenclature, product]) == 1, (
        "Only one of the parameters country, nomenclature, or product can be not None")

    mata_params = {"country": country, "nomenclature": nomenclature, "product": product}
    meta_param = next(name for name, value in mata_params.items() if value is not None)

    request_url = (
        f"{WITS_META_BASE_URL}/datasource/trn"
        f"{f'/country/{country}' if country is not None else ''}"
        f"{f'/nomenclature/{nomenclature}' if nomenclature is not None else ''}"
        f"{f'/product/{product}' if product is not None else ''}"
    )
    print(request_url)
    response = requests.get(request_url)
    response.raise_for_status()
    response_dict = xmltodict.parse(response.content)
    response_dict = response_dict["wits:datasource"][f"wits:{p.plural(meta_param)}"][f"wits:{meta_param}"]
    response_dict = [response_dict] if isinstance(response_dict, dict) else response_dict
    meta_info = pd.DataFrame(response_dict)

    # Data harmonization
    meta_info = (meta_info
        .rename(columns=lambda x: re.sub(r"^[^a-zA-Z0-9]+", "", x))
        .rename(columns=lambda x: re.sub(r"[^a-zA-Z0-9]+", "_", x))
        .rename(columns={"countrycode": "ccn3"})
        .rename(columns=lambda x: "cca3" if re.match(r".*iso_?3.*", x, re.I) else x)
        .rename(columns=lambda x: "cca2" if re.match(r".*iso_?2.*", x, re.I) else x)
    )
    return meta_info

def get_wits_TRAINS_data(reporter, partner, year, product, datatype = "reported"):
    
    reporter = str(reporter)
    validate_paramval_isin_meta(
        datasource="TRAINS", param=reporter, check_param="country",
        param_info_df_name="unctadtrains_country_info",
        meta_fetcher=get_wits_TRAINS_meta, validation_field="ccn3")
    partner = str(partner)
    validate_paramval_isin_meta(
        datasource="TRAINS", param=partner, check_param="country",
        param_info_df_name="unctadtrains_country_info",
        meta_fetcher=get_wits_TRAINS_meta, validation_field="ccn3")
    year = str(year)
    year = "ALL" if year.upper() == "ALL" else year
    if year != "ALL":
        year_list = year.split(";")
        year_list = [
            [y]
            if "-" not in y
            else list(map(str, range(int(y.split("-")[0]), int(y.split("-")[1]) + 1)))
            for y in year_list
        ]
        year_list = list(chain(*year_list))
        assert all(2000 <= int(y) <= 2025 for y in year_list), "Year must be between 2000 and 2025"
        year = ";".join(list(set(year_list)))
    validate_paramval_isin_meta(
        datasource="TRAINS", param=product, check_param="product",
        param_info_df_name="unctadtrains_product_info",
        meta_fetcher=get_wits_TRAINS_meta, validation_field="productcode")
    assert datatype in ["reported", "aveestimated"], (
        "datatype must be either 'reported' or 'aveestimated'")
    
    request_url = (
        f"{WITS_DATA_BASE_URL}/datasource/TRN"
        f"/reporter/{reporter}"
        f"/partner/{partner}"
        f"/product/{product}"
        f"/year/{year}"
        f"/datatype/{datatype}"
    )
    print(request_url)
    response = requests.get(request_url)
    response.raise_for_status()
    response_dict = xmltodict.parse(response.content)
    return response_dict


# %% WITS TRADE Stats
def get_wits_trade_stats_meta(datasource: Literal["trade", "tariff", "development"], country=None, product=None, indicator=None):
    
    assert datasource in ["trade", "tariff", "development"], (
        "datasource must be either 'trade', 'tariff', or 'development'")
    if country is not None:
        country = country.upper()
        validate_paramval_isin_meta(
            datasource=datasource, param=country, check_param="country",
            param_info_df_name=f"{datasource}_country_info",
            meta_fetcher=get_wits_trade_stats_meta, validation_field="cca3"        )
    if product is not None:
        validate_paramval_isin_meta(
            datasource=datasource, param=product, check_param="product",
            param_info_df_name=f"{datasource}_product_info",
            meta_fetcher=get_wits_trade_stats_meta, validation_field="productcode"
        )
    if indicator is not None:
        indicator = indicator.upper()
        validate_paramval_isin_meta(
            datasource=datasource, param=indicator, check_param="indicator",
            param_info_df_name=f"{datasource}_indicator_info",
            meta_fetcher=get_wits_trade_stats_meta, validation_field="indicatorcode"
        )
    assert sum(param is not None for param in [country, product, indicator]) == 1, (
    "Only one of the parameters country, product, or indicator can be not None")

    mata_params = {"country": country, "product": product, "indicator": indicator}
    meta_param = next(name for name, value in mata_params.items() if value is not None)

    request_url = (
        f"{WITS_META_BASE_URL}/datasource/tradestats-{datasource}"
        f"{f'/country/{country}' if country is not None else ''}"
        f"{f'/product/{product}' if product is not None else ''}"
        f"{f'/indicator/{indicator}' if indicator is not None else ''}"
    )
    print(request_url)
    response = requests.get(request_url)
    response.raise_for_status()
    response_dict = xmltodict.parse(response.content)
    response_dict = response_dict["wits:datasource"][f"wits:{p.plural(meta_param)}"][f"wits:{meta_param}"]
    response_dict = [response_dict] if isinstance(response_dict, dict) else response_dict
    meta_info = pd.DataFrame(response_dict)

    # Data harmonization
    meta_info = (meta_info
        .rename(columns=lambda x: re.sub(r"^[^a-zA-Z0-9]+", "", x))
        .rename(columns=lambda x: re.sub(r"[^a-zA-Z0-9]+", "_", x))
        .rename(columns={"countrycode": "ccn3"})
        .rename(columns=lambda x: "cca3" if re.match(r".*iso_?3.*", x, re.I) else x)
        .rename(columns=lambda x: "cca2" if re.match(r".*iso_?2.*", x, re.I) else x)
    )

    return meta_info

def get_wits_trade_stats_data(datasource, reporter, year, indicator, partner = None, product = None):
    assert datasource in ["trade", "tariff", "development"], (
        "datasource must be either 'trade', 'tariff', or 'development'")
    reporter = reporter.upper()
    validate_paramval_isin_meta(
        datasource=datasource, param=reporter, check_param="country",
        param_info_df_name=f"{datasource}_country_info",
        meta_fetcher=get_wits_trade_stats_meta, validation_field="cca3")
    if partner is not None:
        partner = partner.upper()
        validate_paramval_isin_meta(
            datasource=datasource, param=partner, check_param="country",
            param_info_df_name=f"{datasource}_country_info",
            meta_fetcher=get_wits_trade_stats_meta, validation_field="cca3")
    year = str(year)
    year = "ALL" if year.upper() == "ALL" else year
    if year != "ALL":
        year_list = year.split(";")
        year_list = [
            [y]
            if "-" not in y
            else list(map(str, range(int(y.split("-")[0]), int(y.split("-")[1]) + 1)))
            for y in year_list
        ]
        year_list = list(chain(*year_list))
        assert all(2000 <= int(y) <= 2025 for y in year_list), "Year must be between 2000 and 2025"
        year = ";".join(list(set(year_list)))
    indicator = indicator.upper()
    validate_paramval_isin_meta(
        datasource=datasource, param=indicator, check_param="indicator",
        param_info_df_name=f"{datasource}_indicator_info",
        meta_fetcher=get_wits_trade_stats_meta, validation_field="indicatorcode")
    if product is not None:
        validate_paramval_isin_meta(
            datasource=datasource, param=product, check_param="product",
            param_info_df_name=f"{datasource}_product_info",
            meta_fetcher=get_wits_trade_stats_meta, validation_field="productcode")
    
    request_url = (
        f"{WITS_DATA_BASE_URL}/datasource/tradestats-{datasource}"
        f"/reporter/{reporter}"
        f"/year/{year}"
        f"{f'/partner/{partner}' if partner is not None else ''}"
        f"{f'/product/{product}' if product is not None else ''}"
        f"/indicator/{indicator}")
    print(request_url)
    response = requests.get(request_url)
    response.raise_for_status()
    response_dict = xmltodict.parse(response.content)
    response_dict = response_dict['message:StructureSpecificData']['message:DataSet']['Series']
    response_dict = [response_dict] if isinstance(response_dict, dict) else response_dict

    wits_data = (
        pd.DataFrame(response_dict)
        .assign(Obs = lambda x: x.Obs.apply(lambda y: [y] if isinstance(y, dict) else y))
        .explode("Obs")
        .reset_index(drop=True)
        .pipe(lambda x: x.join(x['Obs'].apply(pd.Series)))
        .drop(columns=['Obs']))
    
    # Data harmonization
    wits_data = (wits_data
        .rename(columns=lambda x: re.sub(r"^[^a-zA-Z0-9]+", "", x))
        .rename(columns=lambda x: re.sub(r"[^a-zA-Z0-9]+", "_", x))
        .rename(columns=lambda x: x.lower())
        .rename(columns={"countrycode": "ccn3"})
        .rename(columns=lambda x: "cca3" if re.match(r".*iso_?3.*", x, re.I) else x)
        .rename(columns=lambda x: "cca2" if re.match(r".*iso_?2.*", x, re.I) else x)
        .rename(columns={"reporter": "reporter_cca3", "partner": "partner_cca3", "time_period": "year"}))

    return wits_data



# %% Test
unctadtrains_country_info = get_wits_TRAINS_meta(country="ALL")
response_dict = get_wits_TRAINS_data("000", "156", "2022", "847149")
(pd.DataFrame(response_dict['message:StructureSpecificData']['message:DataSet']['Series'])
    .rename(columns=lambda x: re.sub(r"^[^a-zA-Z0-9]+", "", x))
    .rename(columns=lambda x: re.sub(r"[^a-zA-Z0-9]+", "_", x))
    .rename(columns=lambda x: inflection.underscore(x))
    .merge(unctadtrains_country_info[['ccn3', 'wits_name']], left_on='reporter', right_on='ccn3', how='left')
    .rename(columns={'wits_name': 'reporter_name'})
)