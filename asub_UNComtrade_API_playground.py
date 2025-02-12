#%% Setup
import os
import time
import json
import numpy as np
import pandas as pd
import comtradeapicall

RAW_DATA_DIR = "../data/raw"
LARGE_RAW_DATA_DIR = "/Volumes/SSKat/Blog/1/data/raw"
COMTRADE_API_KEY = os.environ.get('COMTRADE_API_KEY')
COMTRADE_API_KEY = "5ec046cfa52c45b8bf314df79697b96c"
VALIDATED_SKIP = ["PRK", "MHL", "NRU", "PLW", "TKM", "TUV", "SLB"]
output_dir  = f"{LARGE_RAW_DATA_DIR}/trade"

# %% Data
with open("../data/processed/apac_country.json", "r") as f:
    apac_countries = json.load(f)
apac_countries = (
    pd.DataFrame(apac_countries)
    .assign(ccn3=lambda x: x['ccn3'].apply(lambda y: str(y).zfill(3))))

country_info = pd.read_csv("../data/processed/country_info.csv", dtype='str')
country_ccn3_cca3_dict = dict(zip(country_info.ccn3, country_info.cca3))

code_conversion_unwto_to_ccn3 = pd.read_csv(os.path.join(RAW_DATA_DIR, "code_conversion_unwto_to_ccn3.csv"), dtype='str').set_index("unwto_code").ccn3.to_dict()
code_conversion_to_unwto_code = {v:k for k, v in code_conversion_unwto_to_ccn3.items()}


if not os.path.exists(f"{output_dir}/trade_data_tracker_temp.tsv"):
    with open(f"{output_dir}/trade_data_tracker_temp.tsv", "w") as f:
        f.write("reporterCode\treporterISO\tyear\n")
        f.close()

# %% Function
def fetch_yearly_export_data(reporter, year):
    global output_dir
    global COMTRADE_API_KEY
    global country_ccn3_cca3_dict
    global code_conversion_to_unwto_code

    reporter_ = pd.Series(reporter).replace(code_conversion_to_unwto_code).iat[0]
    reporter_cca3 = country_ccn3_cca3_dict[reporter]

    # skip if the data is already fetched
    with open(f"{output_dir}/trade_data_tracker_temp.tsv", "r") as f:
        traker = f.readlines()
        f.close()
    if f"{reporter_}\t{reporter_cca3}\t{year}\n" in traker:
        return

    # get the partners if partially fetched
    if os.path.exists(f"{output_dir}/{reporter_cca3}_EX.tsv"):
        fetched_year_partners = pd.read_csv(f"{output_dir}/{reporter_cca3}_EX.tsv", sep="\t", usecols=["year", "partnerCode"]).drop_duplicates()
        fetched_partners = fetched_year_partners.query(f"year == {year}").partnerCode.to_list()
    else:
        fetched_partners = []
    
    # get top 30 export partners
    top_export_partners = comtradeapicall.getFinalData(
        COMTRADE_API_KEY, 
        typeCode='C', 
        freqCode='A', 
        clCode='HS', 
        period=str(year),
        reporterCode=int(reporter_),
        cmdCode='Total', 
        flowCode="X", 
        partnerCode=comtradeapicall.convertCountryIso3ToCode("ALL"), 
        partner2Code=None, 
        customsCode=None,
        motCode=None,
        breakdownMode='classic', 
        includeDesc=True)
    if top_export_partners is None:
        raise Exception("Out of quota")
    if top_export_partners.empty:
        print(f"!!! Inconsistent with avaliablitity, {reporter_cca3} {year} has no data available")
        return
    
    top30_export_partner = (top_export_partners
            .query('partnerISO.str.match("([A-Z]{3}|S19)")', engine='python')
            .sort_values('primaryValue', ascending=False)
            .head(30)
    )
    top30_export_partnerCodes = top30_export_partner.partnerCode.to_list()

    # Fetch the data
    partnerCodes = list(set(top30_export_partnerCodes) - set(fetched_partners))
    fetch_calls = 0
    while fetch_calls*10 < len(partnerCodes):
        partnerCodes_ = partnerCodes[fetch_calls*10:(fetch_calls+1)*10]
        # Add World
        if fetch_calls == 0:
            partnerCodes_ = [0] + partnerCodes_
        
        # Fetch
        reporter_partner_export_value = comtradeapicall.getFinalData(
            COMTRADE_API_KEY, 
            typeCode='C', 
            freqCode='A', 
            clCode='HS', 
            period=str(year),
            reporterCode=int(reporter_),
            cmdCode='AG6',      
            flowCode="X", 
            partnerCode=",".join(map(str, partnerCodes_)),
            partner2Code=None, 
            customsCode=None,
            motCode=None,
            breakdownMode='classic', 
            includeDesc=True,
        )

        # Check and save
        if reporter_partner_export_value.empty:
            print(f"{reporter_cca3} {year} failed to fetch data")
            raise Exception("Empty Data")
        if "reporterCode" not in reporter_partner_export_value.columns:
            print(f"{reporter_cca3} {year} failed to fetch data")
            raise Exception("Empty Data")
        reporter_partner_export_value_subset = (reporter_partner_export_value
                    [['reporterCode', 'reporterISO', 
                    'partnerCode', 'partnerISO',
                    'classificationCode', 'cmdCode', 
                    'primaryValue']]
                    .assign(year=year)
                    )
        assert reporter_partner_export_value_subset.shape[0] <= 1_000_000
        reporter_partner_export_value_subset.to_csv(f"{output_dir}/{reporter_cca3}_EX.tsv", sep="\t", mode='a', header=False, index=False)
        fetch_calls += 1 

    # Update the tracker
    with open(f"{output_dir}/trade_data_tracker_temp.tsv", "a") as f:
        f.write(f"{reporter_}\t{reporter_cca3}\t{year}\n")
        f.close()
    return

def fetch_export_data(reporter):
    global output_dir
    global COMTRADE_API_KEY
    global country_ccn3_cca3_dict
    global code_conversion_to_unwto_code

    reporter_ = pd.Series(reporter).replace(code_conversion_to_unwto_code).iat[0]
    reporter_cca3 = country_ccn3_cca3_dict[reporter]

    if not os.path.exists(f"{output_dir}/{reporter_cca3}_EX.tsv"):
        with open(f"{output_dir}/{reporter_cca3}_EX.tsv", "w") as f:
            f.write("reporterCode\treporterISO\tpartnerCode\tpartnerISO\tclassificationCode\tcmdCode\tprimaryValue\tyear\n")
            f.close()

    fetch_tracker = pd.read_csv(f"{output_dir}/trade_data_tracker_temp.tsv", sep="\t")
    fetched_years = fetch_tracker.query(f"reporterCode == '{reporter_}'").year.to_list()
    years = range(2008, 2022+1)
    reporter_data_availability = comtradeapicall._getFinalDataAvailability(
        typeCode='C',
        freqCode='A',
        clCode='HS',
        period=','.join(map(str, years)),
        reporterCode=int(reporter_),
    )
    if reporter_data_availability.empty:
        print(f"{reporter_cca3} has no data available")
        return
    years_available = reporter_data_availability.period.to_list()
    fetching_years = list(set(map(int,years_available)) - set(map(int,fetched_years)))

    for year in fetching_years:
        tries = 0
        while tries < 2:
            try:
                fetch_yearly_export_data(reporter, year)
                break
            except Exception as e:
                print(f"{reporter_cca3} {year} failed with error: {e}")
                if "Out of quota" in str(e):
                    quit()
                time.sleep(60)
                tries += 1
        if tries == 2:
            print(f"{reporter_cca3} {year} failed")
            continue
    return

#%% Main
apac_countries[lambda x: (x.ccn3.apply(lambda y: country_ccn3_cca3_dict[y] not in VALIDATED_SKIP))].ccn3.apply(fetch_export_data)

