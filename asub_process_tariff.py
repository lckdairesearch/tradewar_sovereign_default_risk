#%% Setup
import os
import re
import glob
import itertools
import numpy as np
import pandas as pd

RAW_DATA_DIR = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"
LARGE_RAW_DATA_DIR = "/Volumes/SSKat/Blog/1/data/raw"
tariff_dir = os.path.join(LARGE_RAW_DATA_DIR, "tariff")

country_info = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "country_info.csv"), dtype='str')
country_cca3_ccn3_dict = dict(zip(country_info.cca3, country_info.ccn3))

# %% Non standard code to CCN3
if not os.path.exists(os.path.join(RAW_DATA_DIR, "code_conversion_unwto_to_ccn3.csv")):
    tariff_files = glob.glob(os.path.join(tariff_dir, f"MAcMap-*_Tariff_HS6_min.txt"))
    country_cca3s = set([
        re.search(r"MAcMap-([A-Z]{3})_[0-9]{4}_Tariff_HS6_min.txt", os.path.basename(tariff_file)).group(1)
        for tariff_file in tariff_files
    ])
    country_code_cca3_dict = dict()
    for cca3 in country_cca3s:
        country_files = glob.glob(os.path.join(tariff_dir, f"MAcMap-{cca3}_*_Tariff_HS6_min.txt"))
        country_file = country_files[0]
        country_code = pd.read_csv(country_file, sep="\t", dtype='str', nrows=2).iloc[0]['ReportingCountry']
        country_code_cca3_dict[country_code] = cca3
    unwto_code = pd.DataFrame(country_code_cca3_dict.items(), columns=["unwto_code", "cca3"])
    country_info = pd.read_csv("../data/processed/country_info.csv", dtype='str')
    code_conversion_unwto_to_ccn3 = (unwto_code
        .assign(cca3=lambda x: x.cca3.replace({"ROM":"ROU"}))
        .merge(
            country_info[['ccn3', 'cca3']],
            left_on="cca3",
            right_on="cca3",
            how="left"
        )
        .assign(is_different =lambda x: x.unwto_code != x.ccn3)
        .query("not ccn3.isnull()")
        .query("is_different")
        .set_index("unwto_code")
        .ccn3
        .to_dict()
    )
    if not os.path.exists(os.path.join(RAW_DATA_DIR, "code_conversion_unwto_to_ccn3.csv")):
        pd.DataFrame(code_conversion_unwto_to_ccn3.items(), columns=["unwto_code", "ccn3"]).to_csv(
            os.path.join(RAW_DATA_DIR, "code_conversion_unwto_to_ccn3.csv"),
            index=False
        )
else:
    code_conversion_unwto_to_ccn3 = pd.read_csv(os.path.join(RAW_DATA_DIR, "code_conversion_unwto_to_ccn3.csv"), dtype='str').set_index("unwto_code").ccn3.to_dict()
    
code_conversion_to_unwto_code = {v:k for k, v in code_conversion_unwto_to_ccn3.items()}



# %% Get Tariff Data
def get_processed_tariff_data_sub(reporter = None, partner = None, year = None):
    global LARGE_RAW_DATA_DIR
    global tariff_dir
    global code_conversion_unwto_to_ccn3
    global country_info
    global country_cca3_ccn3_dict

    assert reporter is None or isinstance(reporter, list) or isinstance(reporter, str)
    if isinstance(reporter, str):
        reporter = [reporter]
    if isinstance(reporter, list):
        assert all([r in country_info.cca3.values for r in reporter])
    assert partner is None or isinstance(partner, list) or isinstance(partner, str)
    if isinstance(partner, str):
        partner = [partner]
    if isinstance(partner, list):
        assert all([p in country_info.cca3.values for p in partner])
    assert year is None or isinstance(year, list)  or isinstance(year, str) or isinstance(year, int)
    if isinstance(year, str) or isinstance(year, int):
        year = [str(year)]
    if isinstance(year, list):
        assert all([re.match("[0-9]{4}", y) for y in year])


    if reporter is None:
        reporter_regex = "[A-Z]{3}"
    else:
        reporter_regex = f"({'|'.join(reporter)})"
    if partner is None:
        partner_regex = "[0-9]{3}"
        partner_ccn3s = country_cca3_ccn3_dict.values()
        partner_unwto_codes = pd.Series(partner_ccn3s).replace(code_conversion_to_unwto_code).to_list()
    else:
        partner_ccn3s = [country_cca3_ccn3_dict[p] for p in partner]
        partner_unwto_codes = pd.Series(partner_ccn3s).replace(code_conversion_to_unwto_code).to_list()
    if year is None:
        year_regex = "[0-9]{4}"
    else:
        year_regex = f"({'|'.join(year)})"

    tariff_files = glob.glob(os.path.join(tariff_dir, f"MAcMap-*_Tariff_HS6_min.txt"))
    tariff_files = [
        tariff_file
        for tariff_file in tariff_files
        if re.match(f"MAcMap-{reporter_regex}_{year_regex}_Tariff_HS6_min.txt", os.path.basename(tariff_file))
    ]
    # create an empty dataframe
    tariff_data = pd.DataFrame(columns=['hs_nomenclature', 'importer_ccn3', 'year', 'hs6', 'exporter_ccn3',
        'no_of_tariff_lines', 'min_ad_valorem', 'max_ad_valorem', 'is_imputed', 'imputed_from'])
    for tariff_file in tariff_files:
        tariff_data_ = (pd.read_csv(tariff_file, sep="\t", dtype='str')
            # rename columns to snake case
            .rename(columns={
                "Revision": "hs_nomenclature",
                "ReportingCountry": "importer_code_unwto",
                "PartnerCountry": "exporter_code_unwto",
                "ProductCode": "hs6",
                "MinAve": "min_ad_valorem",
                "MaxAve": "max_ad_valorem",
            })
            .rename(columns=lambda x: re.sub(r'(?<!^)(?=[A-Z])', '_', x).lower())
            .query("exporter_code_unwto in @partner_unwto_codes")
            .assign(hs6 = lambda x: x['hs6'].apply(lambda y: str(y).zfill(6)))
            .assign(is_imputed = 0)
            .assign(imputed_from = np.NaN)
            .astype({"min_ad_valorem": "float", "max_ad_valorem": "float"})
            .assign(importer_ccn3 = lambda x: x['importer_code_unwto'].replace(code_conversion_unwto_to_ccn3))
            .assign(exporter_ccn3 = lambda x: x['exporter_code_unwto'].replace(code_conversion_unwto_to_ccn3))
            .drop(columns=["source"])
            .reset_index(drop=True)
            [['hs_nomenclature', 'importer_ccn3', 'year', 'hs6', 'exporter_ccn3',
            'no_of_tariff_lines', 'min_ad_valorem', 'max_ad_valorem', 'is_imputed', 'imputed_from']]
        )

        tariff_data = pd.concat([tariff_data, tariff_data_], ignore_index=True)

    return tariff_data



def get_processed_tariff_data(reporter = None, partner = None, year = None):
    tariff_data = get_processed_tariff_data_sub(reporter=reporter, partner=partner, year=year)

    # Check if data if missing and impute
    if reporter is None:
        reporter = country_cca3_ccn3_dict.keys()
    if isinstance(reporter, str):
        reporter = [reporter]
    if isinstance(partner, str):
        partner = [partner]
    if year is None:
        year = map(str, range(2008, 2022))
    if isinstance(year, str) or isinstance(year, int):
        year = [str(year)]
    for r, y in itertools.product(reporter, year):
        is_data_empty = (tariff_data
            [lambda x: (x.importer_ccn3 == country_cca3_ccn3_dict[r]) & (x.year == y)]
            .shape[0]
        ) == 0
        if is_data_empty:
            intrapolating_years = [str(int(y)-1), str(int(y)+1)]
            tariff_data_ = get_processed_tariff_data_sub(reporter=r, partner = partner, year=intrapolating_years)
            if tariff_data_.shape[0] == 0:
                continue
            tariff_data_ = (tariff_data_
                .groupby(['importer_ccn3', 'exporter_ccn3', 'hs6'])
                .agg(min_ad_valorem=('min_ad_valorem', 'mean'), max_ad_valorem=('max_ad_valorem', 'mean'),
                     imputed_from=('year', lambda x: ",".join(x)))
                .reset_index()
                .assign(year=y)
                .assign(is_imputed=1)
            )
            tariff_data = pd.concat([tariff_data, tariff_data_], ignore_index=True)
    return tariff_data

# %%
