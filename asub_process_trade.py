#%% Setup
import os
import shutil
import glob
import re
import numpy as np
import pandas as pd
import pyreadr

RAW_DATA_DIR = "../data/raw"
LARGE_RAW_DATA_DIR = "/Volumes/SSKat/Blog/1/data/raw"
PROCESSED_DATA_DIR = "../data/processed"

code_conversion_unwto_to_ccn3 = pd.read_csv(os.path.join(RAW_DATA_DIR, "code_conversion_unwto_to_ccn3.csv"), dtype='str').set_index("unwto_code").ccn3.to_dict()
code_conversion_to_unwto_code = {v:k for k, v in code_conversion_unwto_to_ccn3.items()}
country_info = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "country_info.csv"), dtype='str')
country_cca3_ccn3_dict = dict(zip(country_info.cca3, country_info.ccn3))

#%% Get Trade Data

def get_processed_trade_data(reporter=None, partner=None, year=None, trade_flow="EX", inc_world=True):
    assert reporter is None or isinstance(reporter, list) or isinstance(reporter, str), "reporter must be either None, list, or str"
    if reporter is None:
        reporter = country_info.cca3.values.tolist()
        reporter_regex = "[A-Z]{3}"
    if isinstance(reporter, str):
        reporter = [reporter]
    if isinstance(reporter, list):
        assert all([r in country_info.cca3.values for r in reporter]), "reporter must be a ISO 3 letter country code"
        reporter_regex = "|".join(reporter)

    assert partner is None or isinstance(partner, list) or isinstance(partner, str), "partner must be either None, list, or str"
    if partner is None:
        partner = country_info.cca3.values.tolist()
        partner_regex = "[A-Z]{3}"
    if isinstance(partner, str):
        partner = [partner]
    if isinstance(partner, list):
        assert all([p in country_info.cca3.values for p in partner]), "partner must be a ISO 3 letter country code"
        partner_ = [p if p != "TWN" else "S19" for p in partner] + (["W00"] if inc_world else [])
        partner_regex = "|".join(partner)

    assert year is None or isinstance(year, list) or isinstance(year, str) or isinstance(year, int), "year must be either None, list, str, or int"
    if year is None:
        year_regex = "[0-9]{4}"
    if isinstance(year, str) or isinstance(year, int):
        year = [str(year)]
    if isinstance(year, list):
        assert all([re.match("[0-9]{4}", y) for y in year]), "year must be 4 digit"
        year_regex = "|".join(year)
    assert trade_flow in ["EX", "IM"], "trade_flow must be either 'EX' or 'IM'"

    trade_files = glob.glob(os.path.join(LARGE_RAW_DATA_DIR, "trade", f"*_{trade_flow}.tsv"))
    trade_files = [
        file
        for file in trade_files
        if re.match(f"{reporter_regex}_{trade_flow}.tsv", os.path.basename(file))
    ]

    trade_data = pd.DataFrame(columns=["year", "exporter_cca3", "importer_cca3", "hs_nomenclature", "hs6", "trade_value", "is_world"])
    for trade_file in trade_files:
        trade_data_ = (
            pd.read_csv(trade_file, sep='\t', dtype='str')
            .drop_duplicates()
            .rename(columns={
                "classificationCode": "hs_nomenclature",
                "cmdCode": "hs6",
                "primaryValue": "trade_value",
                "reporterISO": "reporter_cca3",
                "partnerISO": "partner_cca3",
            })
            .astype({"year": str, "trade_value": np.float64})
            .rename(columns = lambda x: x.replace("ISO", "Iso"))
            .rename(columns = lambda x: re.sub(r'(?<!^)(?=[A-Z])', '_', x).lower())
            .assign(
                reporter_cca3 = lambda x: x.reporter_cca3.replace({"S19": "TWN"}),
                partner_cca3 = lambda x: x.partner_cca3.replace({"S19": "TWN"})
            )
            .assign(is_world = lambda x: x.partner_cca3 == "W00")
            [lambda x: x.partner_cca3.isin(partner_)]
            [lambda x: x.year.isin(year)]
            .rename(columns = {
                col: 
                    col.replace("reporter", "exporter").replace("partner", "importer")
                    if trade_flow == "EX" else
                    col.replace("reporter", "importer").replace("partner", "exporter")
                for col in ["reporter_cca3", "partner_cca3"]
            })
            [["year", "exporter_cca3", "importer_cca3", "hs_nomenclature", "hs6", "trade_value", "is_world"]]
            .reset_index(drop=True)
        )
        trade_data = pd.concat([trade_data, trade_data_], ignore_index=True)

    return trade_data
