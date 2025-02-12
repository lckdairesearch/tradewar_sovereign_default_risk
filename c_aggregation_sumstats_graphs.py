#%% Setup
import os
import json
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import folium
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
VALIDATED_SKIP = ["PRK", "MHL", "NRU", "PLW", "TKM", "TUV", "SLB"]
RELEVANT_DDS = ['Telecommunications services', 'Computer services','Information services']
YEARS = range(2008, 2022+1)

# %% Load Data
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
del apac_countries
apac_country_ccn3s =  apac_country.ccn3.unique()
apac_country_ccn3s = [c for c in apac_country_ccn3s if country_ccn3_to_cca3[c] not in VALIDATED_SKIP]
apac_country_cca3s = [country_ccn3_to_cca3[c] for c in apac_country_ccn3s]

with open(f"{PROCESSED_DATA_DIR}/apac_capital_coordinate.json", "r") as f:
    apac_capital_coordinates = json.load(f)
apac_capital_coordinate = (
    pd.DataFrame(apac_capital_coordinates)
    .T
    .reset_index()
    .rename(columns={"index": "cca3"})
)
del apac_capital_coordinates

sovereign_default_risk = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "sovereign_default_risk.csv"))

dds = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "digitally_delivered_services.csv"))
#%% Load and Aggregate Trade and Tariff Data
def get_trade_tariff_data_by_hs4(reporter, year):
    global country_cca3_to_ccn3
    global country_ccn3_to_cca3

    def agg_sum_with_nan(series):
        return np.nan if series.isna().any() else series.sum()

    trade_data = get_processed_trade_data(reporter=reporter, year=year)
    importers = trade_data.importer_cca3.unique()
    importers = [x for x in importers if x != 'W00']
    tariff_data = (get_processed_tariff_data(reporter=importers, partner=reporter, year=year)
        .assign(
            exporter_cca3=lambda x: x.exporter_ccn3.replace(country_ccn3_to_cca3),
            importer_cca3=lambda x: x.importer_ccn3.replace(country_ccn3_to_cca3))
        .drop(columns=['exporter_ccn3', 'importer_ccn3', 'hs_nomenclature']))



    trade_tariff_data = (trade_data
        .merge(
            tariff_data,
            on=['year', 'importer_cca3', 'exporter_cca3', 'hs6'],
            how = 'left')
        .assign(customs_duty=lambda x: x.trade_value * x.min_ad_valorem)
        .assign(hs4 = lambda x: x.hs6.str[:4])
        .groupby(['year', 'exporter_cca3', 'importer_cca3', 'hs4'])
        .agg({'trade_value': agg_sum_with_nan, 'customs_duty': agg_sum_with_nan})
        .reset_index()
        .assign(wavg_ad_valorem=lambda x: x.customs_duty / x.trade_value)
        .replace([np.inf, -np.inf], np.nan)
        [['exporter_cca3', 'importer_cca3', 'year', 'hs4', 'trade_value', 'customs_duty', 'wavg_ad_valorem']]
    )
    return trade_tariff_data

# # Commented out to prevent accidental execution
# pd.DataFrame(columns=['exporter_cca3', 'importer_cca3', 'year', 'hs4', 'trade_value', 'customs_duty', 'wavg_ad_valorem']).to_csv("../data/processed/agg_partner_hs4_trade_tariff.csv", index=False)
# for reporter, year in itertools.product(apac_country_cca3s, YEARS):
#     data = get_trade_tariff_data_by_hs4(reporter, year)
#     data.to_csv(f"{PROCESSED_DATA_DIR}/agg_partner_hs4_trade_tariff.csv", mode='a', header=False, index=False)

agg_partner_hs4_trade_tariff = pd.read_csv(
    f"{PROCESSED_DATA_DIR}/agg_partner_hs4_trade_tariff.csv", 
    dtype={
        'exporter_cca3': 'str', 'importer_cca3': 'str', 
        'year': 'int', 'hs4': 'str', 'trade_value': 'float', 
        'customs_duty': 'float', 'wavg_ad_valorem': 'float'})

# %% Aggregation to partner HS2 Level
agg_partner_hs2_trade_tariff =(agg_partner_hs4_trade_tariff
    .assign(hs2 = lambda x: x.hs4.str[:2])
    .groupby(['exporter_cca3', 'year', 'importer_cca3', 'hs2'])
    .agg({'trade_value': sum, 'customs_duty': sum})
    .reset_index()
    .assign(wavg_ad_valorem=lambda x: x.customs_duty / x.trade_value)
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=['wavg_ad_valorem'])
    .assign(wavg_ad_valorem =lambda x: x.wavg_ad_valorem)
)
agg_partner_hs2_trade_tariff.to_csv(f"{PROCESSED_DATA_DIR}/agg_partner_hs2_trade_tariff.csv", index=False)


# %% Aggregation at Top Level by Year Exporter
agg_toplv_trade_tariff = (agg_partner_hs4_trade_tariff
    .assign(is_importer_world=lambda x: x.importer_cca3 == 'W00')
    .query('is_importer_world == False')
    .groupby(['year', 'exporter_cca3'])
    .agg({'trade_value': 'sum', 'customs_duty': 'sum'})
    .reset_index()
    .assign(wavg_ad_valorem=lambda x: x.customs_duty / x.trade_value)
)

data_agg_toplv = (agg_toplv_trade_tariff
    .merge(
        apac_country[['cca3', 'is_emerging_market']],
        left_on='exporter_cca3',
        right_on='cca3',)
    .drop(columns=['cca3'])
    .merge(sovereign_default_risk,
        left_on= ['exporter_cca3', 'year'],
        right_on= ['cca3', 'year'],
        how='left')
    .drop(columns=['cca3'])
    .merge((dds
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
        right_on=['cca3', 'year'],)
    .drop(columns=['cca3'])
    .assign(
        wavg_ad_valorem=lambda x: np.where(
            x.wavg_ad_valorem > x.wavg_ad_valorem.mean() + 1.96 * x.wavg_ad_valorem.std(),
            np.nan,x.wavg_ad_valorem))
    .assign(dds_ex_to_trade=lambda x: x.dds_export / x.trade_value)
    .assign(market_type = lambda x: np.where(x.is_emerging_market, 'Other Emerging', 'Developed'))
    .assign(market_type = lambda x: np.where(x.exporter_cca3 == "CHN", "China", x.market_type))
    .assign(market_type = lambda x: np.where(x.exporter_cca3 == "IND", "India", x.market_type))
)

data_agg_toplv.to_csv(f"{PROCESSED_DATA_DIR}/data_agg_toplv.csv", index=False)



# %% Aggregation at Top Level by Year MarketType
data_agg_toplv = pd.read_csv(f"{PROCESSED_DATA_DIR}/data_agg_toplv.csv")

data_agg_toplv_bymarkettype = (data_agg_toplv
    .assign(market_type = lambda x: np.where(x.is_emerging_market, 'Other Emerging', 'Developed'))
    .assign(market_type = lambda x: np.where(x.exporter_cca3 == "CHN", "China", x.market_type))
    .assign(market_type = lambda x: np.where(x.exporter_cca3 == "IND", "India", x.market_type))
    .groupby(['year', 'market_type'])
    .agg({'trade_value': 'sum', 'customs_duty': 'sum', 'country_risk_premium': 'mean', 'dds_export': 'sum'})
    .reset_index()
    .assign(wavg_ad_valorem=lambda x: x.customs_duty / x.trade_value)
    .assign(dds_ex_to_trade=lambda x: x.dds_export / x.trade_value)
)
# %% Hypothetical increase tariff by 10% in US imports
from f_hypothetical import hyp_usplus10_agg_toplv
hyp_usplus10_agg_toplv_bymarkettype = (
    hyp_usplus10_agg_toplv
    .assign(year=lambda x: x.year.astype('int'))
    .merge(
        apac_country[['cca3', 'is_emerging_market']],
        left_on='exporter_cca3',
        right_on='cca3',)
    .drop(columns=['cca3'])
    .assign(market_type =lambda x: np.where(x.is_emerging_market, 'Other Emerging', 'Developed'))
    .assign(market_type = lambda x: np.where(x.exporter_cca3 == "CHN", "China", x.market_type))
    .assign(market_type = lambda x: np.where(x.exporter_cca3 == "IND", "India", x.market_type))
    .groupby(['year', 'market_type'])
    .agg({'trade_value': 'sum', 'customs_duty': 'sum'})
    .reset_index()
    .astype({'year': 'int'})
    .assign(wavg_ad_valorem=lambda x: x.customs_duty / x.trade_value)
    .assign(is_hypothetical=True)
)

#%% View difference in weighted ad valorem

(
    pd.concat([
        data_agg_toplv_bymarkettype
            .query('year == 2022'),
        hyp_usplus10_agg_toplv_bymarkettype])
    .assign(prev_wavg_ad_valorem=lambda x: x.groupby('market_type').wavg_ad_valorem.shift(1))
    .assign(wavg_ad_valorem_diff=lambda x: x.wavg_ad_valorem - x.prev_wavg_ad_valorem)
    .dropna(subset=['prev_wavg_ad_valorem'])
    [['year', 'market_type', 'wavg_ad_valorem', 'prev_wavg_ad_valorem', 'wavg_ad_valorem_diff']]
    
)

# %% Plot Weighted Ad Valorem Across Years
plt.figure(figsize=(10, 6))
sns.lineplot(data=data_agg_toplv_bymarkettype,
             x='year', y='wavg_ad_valorem', hue='market_type',
             palette={
                 'Developed': 'blue', 
                 'Other Emerging': 'green',
                 'China': 'red', 
                 'India':'orange'}, 
            linewidth=2.5, marker='o')

sns.scatterplot(data=hyp_usplus10_agg_toplv_bymarkettype,
                x='year', y='wavg_ad_valorem', hue='market_type',
                palette={
                    'Developed': 'blue', 
                    'Other Emerging': 'green',
                    'China': 'red', 
                    'India':'orange'}, 
                marker='x', s=100)

sns.lineplot(data=pd.concat([data_agg_toplv_bymarkettype, hyp_usplus10_agg_toplv_bymarkettype]),
             x='year', y='wavg_ad_valorem', hue='market_type',
             palette={
                 'Developed': 'blue', 
                 'Other Emerging': 'green',
                 'China': 'red', 
                 'India':'orange'}, 
            linewidth=1, marker='', alpha = 0.3)

plt.axvspan(2024, 2026, color='grey', alpha=0.2)

ax = plt.gca()
ymin, ymax = ax.get_ylim()
y_center = (ymin + ymax) / 2
ax.text(2025, y_center, 'If US\n+10% tariff\non everything',
        fontsize=12, color='black', ha='center', va='bottom')

plt.xlabel('Year', fontsize=18)
plt.ylabel('Weighted Avg Ad Valorem (%)', fontsize=18)
desired_order = ['Developed', 'Other Emerging', 'China', 'India']
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [handles[labels.index(label)] for label in desired_order if label in labels]
ordered_labels = [label for label in desired_order if label in labels]
plt.legend(ordered_handles, ordered_labels, title='Market Type',
           title_fontsize=18, fontsize=16, loc='upper left')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.figtext(0.5, 0.01,
            'Note: This hypothetical calculation assumes export values remain constant from the last available data year.',
            wrap=False, horizontalalignment='center', fontsize=11)
plt.grid(True)
plt.title("Customs Burden as Weighted Average Ad Valorem Across Years", fontsize=20)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.tight_layout()
plt.show()


#%% Plot of Trade in AI-related Services Across Years
plt.figure(figsize=(10, 6))
sns.lineplot(data=data_agg_toplv_bymarkettype,
             x='year', y='dds_export', hue='market_type',
             palette={'Developed': 'blue', 'Other Emerging': 'green', 'China': 'red', 'India':'orange'}, linewidth=2.5, marker='o')
sns.lineplot(data=data_agg_toplv.query('exporter_cca3 == "CHN"'),x='year', y='dds_export', 
             color='red', linewidth=2.5, marker='o', alpha = 0.25)
plt.xlabel('Year', fontsize=18)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylabel('Trade Value of AI-related Services (billion USD)', fontsize=16)
desired_order = ['Developed', 'Other Emerging', 'China', 'India']
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [handles[labels.index(label)] for label in desired_order if label in labels]
ordered_labels = [label for label in desired_order if label in labels]
plt.legend(ordered_handles, ordered_labels, title='Market Type',
           title_fontsize=18, fontsize=16, loc='upper left')
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y/1e9:.0f}'))
plt.title("Trade in AI-related Services Across Years", fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot AI-related Service Exports and Country Risk Premium in 2022
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))# %% Geo Plot of AI-related Service Exports and Country Risk Premium
data_agg_toplv_ = (data_agg_toplv
    .merge(
        apac_capital_coordinate,
        left_on='exporter_cca3',
        right_on='cca3')
    .drop(columns=['cca3'])
    .dropna(subset=['lat', 'lon'])
)

data_agg_toplv_gdf_ = gpd.GeoDataFrame(
    data_agg_toplv_.query('year == 2022'),
    geometry=gpd.points_from_xy(data_agg_toplv_.query('year == 2022').lon, data_agg_toplv_.query('year == 2022').lat),
    crs="EPSG:4326"
)
extent = (5000000, 20000000, -6500000, 7500000)
data_agg_toplv_gdf_ = data_agg_toplv_gdf_.to_crs(epsg=3857)
data_agg_toplv_gdf_ = data_agg_toplv_gdf_[
    lambda x: 
        (x.geometry.x >= extent[0]) & (x.geometry.x <= extent[1]) &
        (x.geometry.y >= extent[2]) & (x.geometry.y <= extent[3])
]
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(
    data_agg_toplv_gdf_.geometry.x, data_agg_toplv_gdf_.geometry.y,
    s=(data_agg_toplv_gdf_['dds_export']/5e8)**2,
    color='turquoise',
    alpha=0.4,
    label='AI-related Service Exports'
)

ax.scatter(
    data_agg_toplv_gdf_.geometry.x, data_agg_toplv_gdf_.geometry.y,
    s=(data_agg_toplv_gdf_['country_risk_premium'] * 4e2) ** 2,
    color='red',
    alpha=0.4,
    label='Country Risk Premium'
)

for _, row in data_agg_toplv_gdf_.iterrows():
    ax.text(row.geometry.x, row.geometry.y, row.exporter_cca3, fontsize=10, ha='right', color="black")



ax.axis(extent)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager)


ax.set_axis_off()

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='AI-related Service Exports',
           markerfacecolor='turquoise', markersize=12, alpha=0.4),
    Line2D([0], [0], marker='o', color='w', label='Country Risk Premium',
           markerfacecolor='red', markersize=8, alpha=0.4)
]
ax.legend(handles=legend_elements, loc='upper right', prop={'size': 12})

plt.title('AI-related Service Exports and Country Risk Premium in 2022', fontsize=20)
plt.show()
