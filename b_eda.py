#%% Setup
import os
import json
import numpy as np
import pandas as pd
import itertools
from patsy import dmatrices
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from asub_process_trade import get_processed_trade_data
from asub_process_tariff import get_processed_tariff_data

PROCESSED_DATA_DIR = "../data/processed"
VALIDATED_SKIP = ["PRK", "MHL", "NRU", "PLW", "TKM", "TUV", "SLB"]
RELEVANT_DDS = ['Telecommunications services', 'Computer services','Information services']


years = range(2008, 2022+1)

country_info = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "country_info.csv"), dtype='str')
country_cca3_to_ccn3 = dict(zip(country_info.cca3, country_info.ccn3))
country_ccn3_to_cca3 = dict(zip(country_info.ccn3, country_info.cca3))

with open("../data/processed/apac_country.json", "r") as f:
    apac_countries = json.load(f)
apac_country = (
    pd.DataFrame(apac_countries)
    .assign(ccn3=lambda x: x['ccn3'].apply(lambda y: str(y).zfill(3))))
apac_country = (apac_country
    .assign(cca3 = lambda x: x.ccn3.replace(country_ccn3_to_cca3))
    .query('cca3 not in @VALIDATED_SKIP')
)
apac_country_ccn3s =  apac_country.ccn3.unique()
del apac_countries

sovereign_default_risk = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "sovereign_default_risk.csv"))

dds = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "digitally_delivered_services.csv"))

#%% Top level aggregation - yearly export value and tariff exposure
def get_agg_trade_tariff_stat_by_reporter_year(reporter, year):
    trade_data = get_processed_trade_data(reporter=reporter, year=year)
    importers = trade_data.importer_cca3.unique()
    importers = [x for x in importers if x != 'W00']
    tariff_data = (get_processed_tariff_data(reporter=importers, partner=reporter, year=year)
        .assign(
            exporter_cca3=lambda x: x.exporter_ccn3.replace(country_ccn3_to_cca3),
            importer_cca3=lambda x: x.importer_ccn3.replace(country_ccn3_to_cca3))
        .drop(columns=['exporter_ccn3', 'importer_ccn3', 'hs_nomenclature']))
    
    data = (trade_data
        .merge(
            tariff_data,
            on=['year', 'importer_cca3', 'exporter_cca3', 'hs6'],
            how = 'left')
        )
    agg_stats = (data
        .assign(tariff_amount=lambda x: x.trade_value * x.min_ad_valorem)
        .query('is_world == False')
        .groupby('importer_cca3')
        .agg({'trade_value': 'sum', 'tariff_amount': 'sum'})
        .assign(tariff_exposure=lambda x: x.tariff_amount / x.trade_value)
        .agg({'trade_value': 'sum', 'tariff_amount': 'sum', 'tariff_exposure': 'mean'})
        .to_frame().T
        .assign(exporter_cca3=reporter, year=year)
     )
    return agg_stats


#%% EDAs

agg_coarse_trade_tariff_stats = pd.read_csv(f"{PROCESSED_DATA_DIR}/agg_coarse_ trade_tariff_stats.csv")

#%% EDA Reg
reg_data = (agg_coarse_trade_tariff_stats
    .astype({'year': 'int'})
    .merge(sovereign_default_risk,
    left_on= ['exporter_cca3', 'year'],
    right_on= ['cca3', 'year'],
    how='left')
    .drop(columns=['cca3'])
    .dropna(subset=['default_spread'])
    .merge(
        apac_country[['cca3', 'is_emerging_market']],
        left_on='exporter_cca3',
        right_on='cca3',
    )
    .drop(columns=['cca3'])
    .assign(
        not_emerging_market=lambda x: ~x.is_emerging_market
    )
    .sort_values(['exporter_cca3', 'year'])

    .merge(
        (dds
            .query('flow == "X"')
            .query('indicator_name in @RELEVANT_DDS')
            .groupby(['cca3', 'year'])
            .agg({'value': 'sum'})
            .reset_index()
            .rename(columns={'value': 'dds_export'})
            .assign(dds_export=lambda x: x.dds_export * 1e6)
            .sort_values(['cca3', 'year'])
            .astype({'year': 'int'})
            [['cca3', 'year', 'dds_export']]
        ),
        left_on=['exporter_cca3', 'year'],
        right_on=['cca3', 'year'],
    )
    .drop(columns=['cca3'])
    .assign(dds_trade_ratio = lambda x: x.dds_export / x.trade_value)
    .assign(
        default_spread = lambda x: x.default_spread * 1e4,
        equity_risk_premium = lambda x: x.equity_risk_premium * 1e2,
        country_risk_premium = lambda x: x.country_risk_premium * 1e2,
        dds_trade_ratio = lambda x: x.dds_trade_ratio * 1e2,
        tariff_exposure = lambda x: x.tariff_exposure * 1e2
    )
    .astype({'year': 'int'})
    .sort_values(['exporter_cca3', 'year'])
    .assign(
        default_spread_lag1 = lambda x: x.groupby('exporter_cca3')['default_spread'].shift(1),
        equity_risk_premium_lag1 = lambda x: x.groupby('exporter_cca3')['equity_risk_premium'].shift(1),
        country_risk_premium_lag1 = lambda x: x.groupby('exporter_cca3')['country_risk_premium'].shift(1),
        default_spread_lag2 = lambda x: x.groupby('exporter_cca3')['default_spread'].shift(2),
        equity_risk_premium_lag2 = lambda x: x.groupby('exporter_cca3')['equity_risk_premium'].shift(2),
        country_risk_premium_lag2 = lambda x: x.groupby('exporter_cca3')['country_risk_premium'].shift(2),
    )
    [['exporter_cca3', 'year', 'is_emerging_market', 'not_emerging_market',
    'trade_value', 'tariff_amount', 'tariff_exposure', 
    'default_spread', 'equity_risk_premium', 'country_risk_premium',
    'dds_export', 'dds_trade_ratio',
    'default_spread_lag1', 'equity_risk_premium_lag1', 'country_risk_premium_lag1',
    'default_spread_lag2', 'equity_risk_premium_lag2', 'country_risk_premium_lag2'
    ]]
)

# %% Full sample
model = smf.ols("default_spread ~ tariff_exposure + not_emerging_market + C(year)", data=reg_data).fit()
model.summary()


# %% Emerging Market
model = smf.ols("country_risk_premium ~ tariff_exposure*dds_trade_ratio + C(year) ", data=reg_data.query("is_emerging_market == True")).fit()
model.summary()
# anova_results = anova_lm(model, typ=2)
# print(anova_results)

# %%Non Emerging Market
model = smf.ols("country_risk_premium ~ tariff_exposure*dds_trade_ratio + C(year) ", data=reg_data.query("is_emerging_market == False")).fit()
model.summary()
# anova_results = anova_lm(model, typ=2)  # Type II ANOVA
# print(anova_results)

# %%VIF to prove multicollinearity
y, X = dmatrices('country_risk_premium ~ country_risk_premium_lag1 + tariff_exposure * dds_trade_ratio + C(exporter_cca3)', data=reg_data, return_type='dataframe')
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
# %%
