#%% Setup
import os
import shutil
import glob
import re
import json
import numpy as np
import pandas as pd
import pyreadr
import pycountry

RAW_DATA_DIR = "../data/raw"
LARGE_RAW_DATA_DIR = "/Volumes/SSKat/Blog/1/data/raw"
PROCESSED_DATA_DIR = "../data/processed"
# %% Country Info with RESTful API
os.system("python 1_RESTful_API_playground.py")

# %% List of APAC Countries
if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, "apac_country.json")):
    shutil.copy(os.path.join(RAW_DATA_DIR, "apac_country.json"), PROCESSED_DATA_DIR)


# %% Economic Complexity Index
economic_complexity_index = (pd.read_stata(
        os.path.join(RAW_DATA_DIR, "economic-complexity-index.dta")
        )
    [['country_id', 'year', 'hs_eci', 'hs_eci_rank']]
    .query("hs_eci != 0")
    .assign(country_id=lambda x: x['country_id'].apply(lambda y: str(y).zfill(3)))
    .rename(columns={"country_id": "ccn3"})
    .reset_index(drop=True)
)

economic_complexity_index.to_csv(os.path.join(PROCESSED_DATA_DIR, "economic_complexity_index.csv"), index=False)

del economic_complexity_index

# %% Country Similarity
from asub_contry_conversion import country_similarity_country_to_cca3
country_similarity = (
    pd.read_csv(os.path.join(RAW_DATA_DIR, "country-similarity-distance-matrix.csv"))
    .rename(columns={"Unnamed: 0": "country"})
    .melt(id_vars="country", var_name="country2", value_name="similarity")
    .assign(similarity = lambda x: 100 - x['similarity'])
    .assign(country_cca3 = lambda x: x['country'].replace(country_similarity_country_to_cca3),
            country2_cca3 = lambda x: x['country2'].replace(country_similarity_country_to_cca3))
    .query('similarity != 100')
    [['country_cca3', 'country2_cca3', 'similarity']]
)

country_similarity.to_csv(os.path.join(PROCESSED_DATA_DIR, "country_similarity.csv"), index=False)


# %% Sovereign Defacult Risk
from asub_contry_conversion import sovereigndefault_country_to_cca3
sovereign_default_risk = (
    pd.read_excel(os.path.join(RAW_DATA_DIR, "sovereign_default_risk.xlsx"), sheet_name="Sheet1")
    .rename(columns=lambda x: re.sub(r"\s+", "_", x).lower())
    .rename(columns=lambda x: re.sub(f"'", "", x))
    .assign(cca3=lambda x: x['country'].replace(sovereigndefault_country_to_cca3))
    .assign(year = lambda x: x['year'].astype('int') - 1)
    .astype({'year': 'str'})
    .dropna(subset=['cca3'])
    [['cca3', 'year', 'moodys_rating', 'default_spread', 'equity_risk_premium', 'country_risk_premium']]
)
sovereign_default_risk.to_csv(os.path.join(PROCESSED_DATA_DIR, "sovereign_default_risk.csv"), index=False)

# %% WhoGov
who_gov = (pyreadr.read_r(os.path.join(RAW_DATA_DIR, "WhoGov_crosssectional_V3.0.rds"))[None]
   .rename(columns={"country_isocode": "cca3"})
)

who_gov.to_csv(os.path.join(PROCESSED_DATA_DIR, "who_gov.csv"), index=False)
del who_gov

#%% IMF Data
from asub_contry_conversion import imf_country_to_cca3
(pd.read_csv(os.path.join(RAW_DATA_DIR, "imf_gdp.csv"))
    .rename({'GDP, current prices (Billions of U.S. dollars)': 'country'}, axis=1)
    .assign(cca3=lambda x: x['country'].replace(imf_country_to_cca3))
    .melt(id_vars=['cca3', 'country'], var_name='year', value_name='gdp_billion_usd')
    .assign(gdp_billion_usd=lambda x: np.where(x['gdp_billion_usd'] == 'no data', np.nan, x['gdp_billion_usd']))
    .assign(gdp_billion_usd=lambda x: x['gdp_billion_usd'].astype('float'))
    .to_csv(os.path.join(PROCESSED_DATA_DIR, "gdp.csv"), index=False)
)
(pd.read_csv(os.path.join(RAW_DATA_DIR, "imf_gdp_per_capita.csv"))
    .rename({'GDP per capita, current prices\n (U.S. dollars per capita)': 'country'}, axis=1)
    .assign(cca3=lambda x: x['country'].replace(imf_country_to_cca3))
    .melt(id_vars=['cca3', 'country'], var_name='year', value_name='gdp_per_capita_usd')
    .assign(gdp_per_capita_usd=lambda x: np.where(x['gdp_per_capita_usd'] == 'no data', np.nan, x['gdp_per_capita_usd']))
    .assign(gdp_per_capita_usd=lambda x: x['gdp_per_capita_usd'].astype('float'))
    .to_csv(os.path.join(PROCESSED_DATA_DIR, "gdp_per_capita.csv"), index=False)
)
(pd.read_csv(os.path.join(RAW_DATA_DIR, "imf_debt.csv"))
    .rename({'General government gross debt (Percent of GDP)': 'country'}, axis=1)
    .assign(cca3=lambda x: x['country'].replace(imf_country_to_cca3))
    .melt(id_vars=['cca3', 'country'], var_name='year', value_name='govt_debt_percent_gdp')
    .assign(govt_debt_percent_gdp=lambda x: np.where(x['govt_debt_percent_gdp'] == 'no data', np.nan, x['govt_debt_percent_gdp']))
    .assign(govt_debt_percent_gdp=lambda x: x['govt_debt_percent_gdp'].astype('float'))
    .to_csv(os.path.join(PROCESSED_DATA_DIR, "govt_debt.csv"), index=False)
)
 




# %% Tariff Data
from asub_process_tariff import get_processed_tariff_data


# %% Trade
# import a_UNComtrade_API_playground
from asub_process_trade import get_processed_trade_data



# %%  Digitally Delivered Services
country_info = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "country_info.csv"), dtype='str')
country_cca2_to_cca3 = country_info.set_index("cca2").cca3.to_dict()

dds_im = pd.read_csv(os.path.join(RAW_DATA_DIR, "DDS-Imports.csv"))
dds_ex = pd.read_csv(os.path.join(RAW_DATA_DIR, "DDS-Exports.csv"))
dds = (pd.concat([dds_im, dds_ex], ignore_index=True)
 .rename(columns=lambda x: x.lower())
 .rename(columns={"reporter_code": "cca2"})
 .assign(cca3=lambda x: x['cca2'].replace(country_cca2_to_cca3))
 .astype({"value": float})
)
dds.to_csv(os.path.join(PROCESSED_DATA_DIR, "digitally_delivered_services.csv"), index=False)
