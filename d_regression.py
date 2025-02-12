# %% Setup
import os
import json
import numpy as np
import pandas as pd
import itertools
from patsy import dmatrices
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from stargazer.stargazer import Stargazer
import statsmodels.api as sm
from stargazer.stargazer import Stargazer, LineLocation
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Constants
PROCESSED_DATA_DIR = "../data/processed"
OUTPUT_DIR = "../output"
YEARS = range(2008, 2022+1)


# %% Load Data
data_agg_toplv = pd.read_csv(f"{PROCESSED_DATA_DIR}/data_agg_toplv.csv")
# %% Data Processing
data_agg_toplv = (data_agg_toplv
    .assign(market_type=lambda x: np.where(x.market_type.isin(["China", "India"]), "China and India", x.market_type))
    .assign(
        default_spread = lambda x: x.default_spread * 1e4,
        equity_risk_premium = lambda x: x.equity_risk_premium * 1e2,
        country_risk_premium = lambda x: x.country_risk_premium * 1e2,
        dds_ex_to_trade = lambda x: x.dds_ex_to_trade * 1e2,
        wavg_ad_valorem = lambda x: x.wavg_ad_valorem * 1e2
    )
    .astype({'year', 'int'})
 )



# %%
model1 = smf.ols("country_risk_premium ~ 0 +  wavg_ad_valorem*dds_ex_to_trade + C(market_type) + C(year) -1", data=data_agg_toplv).fit()
model1.summary()


# %%
model2 = smf.ols("country_risk_premium ~ wavg_ad_valorem*dds_ex_to_trade + C(year)", data=data_agg_toplv.query('market_type == "Developed"')).fit()
model2.summary()
# %%
model3 = smf.ols("country_risk_premium ~ wavg_ad_valorem*dds_ex_to_trade", data=data_agg_toplv.query('market_type == "China and India"')).fit()
model3.summary()

# %%
model4 = smf.ols("country_risk_premium ~ wavg_ad_valorem*dds_ex_to_trade + C(year)", data=data_agg_toplv.query('market_type == "Other Emerging"')).fit()
model4.summary()

# %%
stargazer = Stargazer([model1, model2, model3, model4])

stargazer.custom_columns(['All APAC', 'Developed', 'China and India', 'Other Emerging'], [1, 1, 1, 1])
stargazer.covariate_order(['wavg_ad_valorem', 'dds_ex_to_trade',  'wavg_ad_valorem:dds_ex_to_trade', 'C(market_type)[Developed]', 'C(market_type)[China and India]', 'C(market_type)[Other Emerging]','Intercept'])
stargazer.rename_covariates({
    'C(market_type)[Developed]': 'MarketType: Developed Economy',
    'C(market_type)[China and India]': 'MarketType: China and India',
    'C(market_type)[Other Emerging]': 'MarketType: Other Emerging Market',
    'wavg_ad_valorem': 'Weighted avgerage Ad Valorem',
    'dds_ex_to_trade': 'Ratio of AI-related service to trade (%)',
    'wavg_ad_valorem:dds_ex_to_trade': 'Ratio of AI service * W.Avg Ad Valorem'
})
stargazer.significant_digits(2)
stargazer.dependent_variable_name("Country Risk Premium (%)")
stargazer.add_line('Year Fixed Effect', ['Yes', 'Yes', 'No', 'Yes'], LineLocation.FOOTER_TOP)
stargazer.show_degrees_of_freedom(False)

stargazer.add_custom_notes(['1. All APAC is estimated with a regression through the origin to obtain the mean of each market type', '2. Year fixed effects are omitted for the China and India regression', 'due of high degree of multicollinearity with Weighted avgerage Ad Valorem and Ratio of AI-related service to trade'])
reg_table_html = stargazer.render_html()

# save to output dir
with open(f'{OUTPUT_DIR}/reg_table.html', 'w') as f:
    f.write(reg_table_html)


# %%
model = smf.ols("country_risk_premium ~ wavg_ad_valorem*dds_ex_to_trade", data=data_agg_toplv.query('market_type == "China and India"')).fit()
model.summary()
# conduct vif
y, X = dmatrices('country_risk_premium ~ wavg_ad_valorem*dds_ex_to_trade + C(year)', data=data_agg_toplv.query('market_type == "Developed"'), return_type='dataframe')
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
# %%
