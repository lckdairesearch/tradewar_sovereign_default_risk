#%% Setup
import os
import json
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import folium
from patsy import dmatrices
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from folium.features import DivIcon
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
from asub_process_trade import get_processed_trade_data
from asub_process_tariff import get_processed_tariff_data

# Constants
PROCESSED_DATA_DIR = "../data/processed"
OUTPUT_DIR = "../output"


#%% Load Data
agg_partner_hs4_trade_tariff = pd.read_csv(
    f"{PROCESSED_DATA_DIR}/agg_partner_hs4_trade_tariff.csv", 
    dtype={
        'exporter_cca3': 'str', 'importer_cca3': 'str', 
        'year': 'int', 'hs4': 'str', 'trade_value': 'float', 
        'customs_duty': 'float', 'wavg_ad_valorem': 'float'})




# %% Agg to HS2
agg_partner_hs2_trade_tariff =(agg_partner_hs4_trade_tariff
    .assign(hs2 = lambda x: x.hs4.str[:2])
    .groupby(['exporter_cca3', 'year', 'importer_cca3', 'hs2'])
    .agg({'trade_value': sum, 'customs_duty': sum})
    .reset_index()
    .assign(wavg_ad_valorem=lambda x: x.customs_duty / x.trade_value)
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=['wavg_ad_valorem'])
    .assign(wavg_ad_valorem =lambda x: x.wavg_ad_valorem * 1e2)
    .assign(log_trade_value =lambda x: np.log(x.trade_value))
)

# %% Reg Impact of Tariff on Trade
hs2s = agg_partner_hs2_trade_tariff.hs2.unique().tolist()

hs2_davg_ad_valorem_trade_impact = []
for hs2 in hs2s:
    model = smf.ols(
        "log_trade_value ~ 0 + wavg_ad_valorem + C(year)", 
        data=agg_partner_hs2_trade_tariff.query('hs2 == @hs2')).fit()
    hs2_davg_ad_valorem_trade_impact.append(model.params['wavg_ad_valorem'])

hs2_davg_ad_valorem_trade_impact = pd.DataFrame({
    'hs2': hs2s,
    'davg_ad_valorem_trade_impact': hs2_davg_ad_valorem_trade_impact
})

#%% Hypothetical
hyp_usplus10_agg_partner_hs2_trade_tariff = (agg_partner_hs2_trade_tariff
    .query('year == 2022')
    .assign(year=2025)
    .merge(hs2_davg_ad_valorem_trade_impact, on='hs2')
    # Tariff increase by 10% for export to USA
    .assign(wavg_ad_valorem=lambda x: np.where(x.importer_cca3 == 'USA', x.wavg_ad_valorem + 10, x.wavg_ad_valorem))
    # Calculate new trade value
    .assign(hyp_trade_impact=
            lambda x:
                np.where(
                    x.importer_cca3 == 'USA',
                    x.davg_ad_valorem_trade_impact * 10 + 1,
                    1
                ))
    .assign(hyp_trade_impact=lambda x: np.where(x.hyp_trade_impact < 0, 0, x.hyp_trade_impact))
    .assign(trade_value=lambda x: x.trade_value * x.hyp_trade_impact)
    .assign(wavg_ad_valorem=lambda x: x.wavg_ad_valorem * 1e-2)
    .assign(customs_duty=lambda x: x.trade_value * x.wavg_ad_valorem)
)
hyp_usplus10_agg_partner_hs2_trade_tariff.to_csv(f"{PROCESSED_DATA_DIR}/hyp_usplus10_agg_partner_hs2_trade_tariff.csv", index=False)
#%% Agg to Top Level
hyp_usplus10_agg_toplv = (hyp_usplus10_agg_partner_hs2_trade_tariff
    .groupby(['exporter_cca3', 'year'])
    .agg({'trade_value': sum, 'customs_duty': sum})
    .reset_index()
    .assign(wavg_ad_valorem=lambda x: x.customs_duty / x.trade_value)
)

hyp_usplus10_agg_toplv.to_csv(f"{PROCESSED_DATA_DIR}/hyp_usplus10_agg_toplv.csv", index=False)







# %%
