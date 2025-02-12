#%% Setup
import os
import json
from joblib import dump
from joblib import load
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROCESSED_DATA_DIR = "../data/processed"
OUTPUT_DIR = "../output"
YEARS = range(2008, 2022+1)
MOODYS_SCALE = [
    "Aaa", "Aa1", "Aa2", "Aa3",
    "A1", "A2", "A3",
    "Baa1", "Baa2", "Baa3",
    "Ba1", "Ba2", "Ba3",
    "B1", "B2", "B3",
    "Caa1", "Caa2", "Caa3",
    "Ca", "C"
]
VALIDATED_SKIP = ["PRK", "MHL", "NRU", "PLW", "TKM", "TUV", "SLB"]

#%%  Load Data
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

#%% Prepare trade and tariff data
agg_partner_trade_tariff = pd.read_csv(
    f"{PROCESSED_DATA_DIR}/agg_partner_hs2_trade_tariff.csv", 
    dtype={
        'exporter_cca3': 'str', 'importer_cca3': 'str', 
        'year': 'int', 'hs2': 'str', 'trade_value': 'float', 
        'customs_duty': 'float', 'wavg_ad_valorem': 'float'})

agg_wide_by_exporter_year = (agg_partner_trade_tariff
    .assign(custom_duty=lambda x: x.trade_value * x.wavg_ad_valorem)
    .groupby(['year', 'exporter_cca3', 'importer_cca3'])
    .agg({"trade_value": "sum", "custom_duty": "sum"})
    .reset_index()
    .assign(wavg_ad_valorem=lambda x: x.custom_duty / x.trade_value)
    .pivot_table(
        index=["exporter_cca3", "year"],
        columns=["importer_cca3"],
         values=["trade_value", "custom_duty", "wavg_ad_valorem"],
        aggfunc="first",
        fill_value=0
    )
)
agg_wide_by_exporter_year.columns = [f"{col[0]}_{col[1]}" for col in agg_wide_by_exporter_year.columns]
agg_wide_by_exporter_year = agg_wide_by_exporter_year.reset_index()


agg_toplv_by_exporter_year = (agg_partner_trade_tariff
    .assign(custom_duty=lambda x: x.trade_value * x.wavg_ad_valorem)
    .groupby(['year', 'exporter_cca3'])
    .agg({"trade_value": "sum", "custom_duty": "sum"})
    .reset_index()
    .assign(wavg_ad_valorem=lambda x: x.custom_duty / x.trade_value)
)

# %% Merge in target and other features
# Target
sovereign_default_risk = pd.read_csv(f"{PROCESSED_DATA_DIR}/sovereign_default_risk.csv").dropna(subset=['cca3'])
exporter_year_moodys_rating = (sovereign_default_risk
 [['cca3', 'year', 'moodys_rating']]
 .query('moodys_rating in @MOODYS_SCALE')
 .assign(moodys_rating=lambda x: pd.Categorical(x.moodys_rating, categories=MOODYS_SCALE, ordered=True))
)

# Other features
gdp = pd.read_csv(f"{PROCESSED_DATA_DIR}/gdp.csv").dropna(subset=['cca3']).drop(columns=['country'])
gdp_per_capita = pd.read_csv(f"{PROCESSED_DATA_DIR}/gdp_per_capita.csv").dropna(subset=['cca3']).drop(columns=['country'])
govt_debt = pd.read_csv(f"{PROCESSED_DATA_DIR}/govt_debt.csv").dropna(subset=['cca3']).drop(columns=['country'])


exporter_year_features = (
    gdp
    .merge(gdp_per_capita, on=['cca3', 'year'], how='outer')
    .merge(govt_debt, on=['cca3', 'year'], how='outer')
    .query('year in @YEARS')
    .groupby('cca3', group_keys=True)
    .apply(lambda x: x.interpolate(method='quadratic'))
    .reset_index(drop=True)
)

mldf_wide = (agg_wide_by_exporter_year
    .merge(
        exporter_year_moodys_rating,
        left_on=['exporter_cca3', 'year'],
        right_on=['cca3', 'year'],
        how='left',
    )
    .drop(columns=['cca3'])
    .dropna(subset=['moodys_rating'])
    .merge(
        exporter_year_features,
        left_on=['exporter_cca3', 'year'],
        right_on=['cca3', 'year'],
        how='left'
    )
    .drop(columns=['cca3'])
    .dropna()
)

mldf_toplv = (agg_toplv_by_exporter_year
    .merge(
        exporter_year_moodys_rating,
        left_on=['exporter_cca3', 'year'],
        right_on=['cca3', 'year'],
        how='left',
    )
    .drop(columns=['cca3'])
    .dropna(subset=['moodys_rating'])
    .merge(
        exporter_year_features,
        left_on=['exporter_cca3', 'year'],
        right_on=['cca3', 'year'],
        how='left'
    )
    .drop(columns=['cca3'])
    .dropna()
    .assign(trade_to_gdp=lambda x: x.trade_value / x.gdp_billion_usd*1e9)
)

# %% Hypothetical Scenario
hyp_usplus10_agg_partner_trade_tariff  = pd.read_csv(f"{PROCESSED_DATA_DIR}/hyp_usplus10_agg_partner_hs2_trade_tariff.csv")

hyp_wide_by_exporter_year = (hyp_usplus10_agg_partner_trade_tariff
    .assign(year = 2022)
    .assign(custom_duty=lambda x: x.trade_value * x.wavg_ad_valorem)
    .groupby(['year', 'exporter_cca3', 'importer_cca3'])
    .agg({"trade_value": "sum", "custom_duty": "sum"})
    .reset_index()
    .assign(wavg_ad_valorem=lambda x: x.custom_duty / x.trade_value)
    .pivot_table(
        index=["exporter_cca3", "year"],
        columns=["importer_cca3"],
         values=["trade_value", "custom_duty", "wavg_ad_valorem"],
        aggfunc="first",
        fill_value=0
    )
 )
hyp_wide_by_exporter_year.columns = [f"{col[0]}_{col[1]}" for col in hyp_wide_by_exporter_year.columns]
hyp_wide_by_exporter_year = hyp_wide_by_exporter_year.reset_index()

hyp_toplv_by_exporter_year = (hyp_usplus10_agg_partner_trade_tariff
    .assign(custom_duty=lambda x: x.trade_value * x.wavg_ad_valorem)
    .groupby(['year', 'exporter_cca3'])
    .agg({"trade_value": "sum", "custom_duty": "sum"})
    .reset_index()
    .assign(wavg_ad_valorem=lambda x: x.custom_duty / x.trade_value)
)

hyp_mldf_wide = (hyp_wide_by_exporter_year
    .merge(
        exporter_year_moodys_rating,
        left_on=['exporter_cca3', 'year'],
        right_on=['cca3', 'year'],
        how='left',
    )
    .drop(columns=['cca3'])
    .dropna(subset=['moodys_rating'])
    .merge(
        exporter_year_features,
        left_on=['exporter_cca3', 'year'],
        right_on=['cca3', 'year'],
        how='left'
    )
    .drop(columns=['cca3'])
    .dropna()
)

hyp_mldf_toplv = (hyp_toplv_by_exporter_year
    .assign(year = 2022)
    .merge(
        exporter_year_moodys_rating,
        left_on=['exporter_cca3', 'year'],
        right_on=['cca3', 'year'],
        how='left',
    )
    .drop(columns=['cca3'])
    .dropna(subset=['moodys_rating'])
    .merge(
        exporter_year_features,
        left_on=['exporter_cca3', 'year'],
        right_on=['cca3', 'year'],
        how='left'
    )
    .drop(columns=['cca3'])
    .dropna()
    .assign(trade_to_gdp=lambda x: x.trade_value / x.gdp_billion_usd*1e9)
)



# %% Prepare Data for machine learning ---------------------------------------
mldf_wide = mldf_wide.assign(
    moodys_rating_encoded=lambda x: x.moodys_rating.cat.codes
)
X_wide = mldf_wide.drop(columns=['moodys_rating', 'moodys_rating_encoded'])
y_wide = mldf_wide['moodys_rating_encoded']

# Preprocessor
categorical_cols = ['year', 'exporter_cca3']
numeric_cols_wide = [col for col in X_wide.columns if col not in categorical_cols]
preprocessor_wide = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols_wide),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)


# Split data
X_wide_train, X_wide_test, y_wide_train, y_wide_test = train_test_split(
    X_wide, y_wide,
    test_size=0.2,
    random_state=RANDOM_SEED
)

#
mldf_toplv = mldf_toplv.assign(
    moodys_rating_encoded=lambda x: x.moodys_rating.cat.codes
)
X_toplv = mldf_toplv.drop(columns=['moodys_rating', 'moodys_rating_encoded'])
y_toplv = mldf_toplv['moodys_rating_encoded']

# Preprocessor
categorical_cols = ['year', 'exporter_cca3']
numeric_cols_toplv = [col for col in X_toplv.columns if col not in categorical_cols]
preprocessor_toplv = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols_toplv),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Split data
X_toplv_train, X_toplv_test, y_toplv_train, y_toplv_test = train_test_split(
    X_toplv, y_toplv,
    test_size=0.2,
    random_state=RANDOM_SEED
)




# %% Random Forest
# Pipiline
pipeline_wide_rf = Pipeline(steps=[
    ('preprocessor', preprocessor_wide),
    ('svd', TruncatedSVD(n_components=40, random_state=RANDOM_SEED)),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED))
])
pipeline_toplv_rf = Pipeline(steps=[
    ('preprocessor', preprocessor_toplv),
    ('svd', TruncatedSVD(n_components=40, random_state=RANDOM_SEED)),
    ('classifier', RandomForestClassifier(n_estimators=20, random_state=RANDOM_SEED))
])

# Fit Random Forest model
pipeline_wide_rf.fit(X_wide_train, y_wide_train)
pipeline_toplv_rf.fit(X_toplv_train, y_toplv_train)

# Evaluate Random Forest model
y_wide_pred = pipeline_wide_rf.predict(X_wide_test)
y_toplv_pred = pipeline_toplv_rf.predict(X_toplv_test)

print("Wide Classification Report:")
print(classification_report(y_wide_test, y_wide_pred))

print("Top Level Classification Report:")
print(classification_report(y_toplv_test, y_toplv_pred))

accuracy_wide = accuracy_score(y_wide_test, y_wide_pred)
accuracy_toplv = accuracy_score(y_toplv_test, y_toplv_pred)
print("Wide Accuracy: {:.2f}%".format(accuracy_wide * 100))
print("Top Level Accuracy: {:.2f}%".format(accuracy_toplv * 100))


# %% Neural Network ------------------
mlp_classifier_wide = MLPClassifier(
    hidden_layer_sizes=(20, 5),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=RANDOM_SEED
)
mlp_classifier_toplv = MLPClassifier(
    hidden_layer_sizes=(20, 10),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=RANDOM_SEED
)

pipeline_wide_nn = Pipeline(steps=[
    ('preprocessor', preprocessor_wide),
    ('classifier', mlp_classifier_wide)
])

pipeline_toplv_nn = Pipeline(steps=[
    ('preprocessor', preprocessor_toplv),
    ('classifier', mlp_classifier_toplv)
])

pipeline_wide_nn.fit(X_wide_train, y_wide_train)
pipeline_toplv_nn.fit(X_toplv_train, y_toplv_train)

y_wide_pred_nn = pipeline_wide_nn.predict(X_wide_test)
y_toplv_pred_nn = pipeline_toplv_nn.predict(X_toplv_test)

print("Wide Neural Network Classification Report:")
print(classification_report(y_wide_test, y_wide_pred_nn))

print("Top Level Neural Network Classification Report:")
print(classification_report(y_toplv_test, y_toplv_pred_nn))

accuracy_wide_nn = accuracy_score(y_wide_test, y_wide_pred_nn)
accuracy_toplv_nn = accuracy_score(y_toplv_test, y_toplv_pred_nn)
print("Wide Neural Network Accuracy: {:.2f}%".format(accuracy_wide_nn * 100))
print("Top Level Neural Network Accuracy: {:.2f}%".format(accuracy_toplv_nn * 100))


# %% Hypothetical Scenario---------------------------------------
hyp_mldf_wide = hyp_mldf_wide.assign(
    moodys_rating_encoded=lambda x: x.moodys_rating.cat.codes
)
hyp_mldf_wide = hyp_mldf_wide[list(set(hyp_mldf_wide.columns) & set(mldf_wide.columns))]
hyp_mldf_wide = hyp_mldf_wide.reindex(columns=mldf_wide.columns, fill_value=0)
X_hyp_wide = hyp_mldf_wide.drop(columns=['moodys_rating', 'moodys_rating_encoded'])
y_2022_wide = hyp_mldf_wide['moodys_rating_encoded']

hyp_mldf_toplv = hyp_mldf_toplv.assign(
    moodys_rating_encoded=lambda x: x.moodys_rating.cat.codes
)
hyp_mldf_toplv = hyp_mldf_toplv[list(set(hyp_mldf_toplv.columns) & set(mldf_toplv.columns))]
hyp_mldf_toplv = hyp_mldf_toplv.reindex(columns=mldf_toplv.columns, fill_value=0)
X_hyp_toplv = hyp_mldf_toplv.drop(columns=['moodys_rating', 'moodys_rating_encoded'])
y_2022_toplv = hyp_mldf_toplv['moodys_rating_encoded']

# Predict
y_hyp_wide_pred_rf = pipeline_wide_rf.predict(X_hyp_wide)
y_hyp_toplv_pred_rf = pipeline_toplv_rf.predict(X_hyp_toplv)
y_hyp_wide_pred_nn = pipeline_wide_nn.predict(X_hyp_wide)
y_hyp_toplv_pred_nn = pipeline_toplv_nn.predict(X_hyp_toplv)

y_hyp_wide_diff_rf = y_hyp_wide_pred_rf - y_2022_wide
y_hyp_toplv_diff_rf = y_hyp_toplv_pred_rf - y_2022_toplv
y_hyp_wide_diff_nn = y_hyp_wide_pred_nn - y_2022_wide
y_hyp_toplv_diff_nn = y_hyp_toplv_pred_nn - y_2022_toplv

print(f"{y_hyp_wide_diff_rf.mean()}, {y_hyp_toplv_diff_rf.mean()}, {y_hyp_wide_diff_nn.mean()}, {y_hyp_toplv_diff_nn.mean()}")



# %%
