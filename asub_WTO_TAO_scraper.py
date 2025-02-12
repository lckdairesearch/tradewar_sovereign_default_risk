# %% Setup
import json
import os
import re
import time
from itertools import product

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

os.chdir("/Users/leodai/Library/CloudStorage/OneDrive-Personal/Work/Blog/1/code")

# %% Environment variables
WTO_TAO_USERNAME1 = "moots.doctor_7z@icloud.com"
WTO_TAO_USERNAME2 = "lariat13.elands@icloud.com"
WTO_TAO_USERNAME3 = "hamster16791@spinly.net"
WTO_TAO_USERNAME4 = "gyxege@dreamclarify.org"
WTO_TAO_PASSWORD = os.environ.get("WTO_TAO_PASSWORD")

# %% Data
# APAC countries
with open("../data/raw/APAC_countries.json", "r") as f:
    apac_countries = json.load(f)
apac_countries = (
    pd.DataFrame(apac_countries)
    .assign(ccn3=lambda x: x['ccn3'].apply(lambda y: str(y).zfill(3))))

# Options for the report
year_option = list(range(1996, 2025+1))
country_option = pd.read_csv("../data/raw/WTO_TAO_country_option.csv")
country_name_code_dict = country_option.set_index("country_name")['country_code'].to_dict()
country_code_name_dict = {v: k for k, v in country_name_code_dict.items()}
chapter_heading_option = pd.read_csv("../data/raw/WTO_TAO_chapter_heading_option.csv", dtype={"chapter_code": str, "heading_code": str})

# Output
if not os.path.exists("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer.tsv"):
    with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer.tsv", "w") as f:
        f.write("exporter_code\timporter_code\tyear\ths6\tvalue\tad_val_duty_mfn\tad_val_duty_best\tn_tariff_lines\n")
if not os.path.exists("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_tracker.tsv"):
    with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_tracker.tsv", "w") as f:
        f.write("exporter_code\tyear\tchapter\theading\n")
if not os.path.exists("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_fail_tracker.tsv"):
    with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_fail_tracker.tsv", "w") as f:
        f.write("exporter_code\tyear\tchapter\theading\n")
if not os.path.exists("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_validated_skip_tracker.tsv"):
    with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_validated_skip_tracker.tsv", "w") as f:
        f.write("exporter_code\tyear\tchapter\theading\n")


# %% Functions

def login_to_wto_tao(driver, n_tries = 3):
    WTO_TAO_USERNAME1 = globals().get("WTO_TAO_USERNAME1", None)
    WTO_TAO_USERNAME2 = globals().get("WTO_TAO_USERNAME2", None)
    WTO_TAO_USERNAME3 = globals().get("WTO_TAO_USERNAME3", None)
    WTO_TAO_USERNAME4 = globals().get("WTO_TAO_USERNAME4", None)

    tries = 0
    while tries < n_tries:
        try: 
            user_name = [WTO_TAO_USERNAME1, WTO_TAO_USERNAME2, WTO_TAO_USERNAME3, WTO_TAO_USERNAME4][np.random.randint(0, 4)]
            globals()['user_name'] = user_name

            WTO_TAO_PASSWORD = globals().get("WTO_TAO_PASSWORD", None)
            if driver.current_url != "https://tao.wto.org":
                driver.get("https://tao.wto.org/welcome.aspx?ReturnUrl=%2fdefault.aspx")
            username_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "ctl00_c_ctrLogin_UserName"))
            )
            username_input.clear()
            username_input.send_keys(user_name)
            password_input = driver.find_element(By.ID, "ctl00_c_ctrLogin_Password")
            password_input.clear()
            password_input.send_keys(WTO_TAO_PASSWORD)
            login_button = driver.find_element(By.ID, "ctl00_c_ctrLogin_LoginButton")
            login_button.click()
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, "ctl00_LeftNavigationMenu1_mLeftMenun0")))
            cookies = driver.get_cookies()
            aspxausth = [i for i in cookies if i['name'] == ".ASPXAUTH"][0]['value']
            asp_net_sessionid = [i for i in cookies if i['name'] == "ASP.NET_SessionId"][0]['value']
            globals()['aspxausth'] = aspxausth
            globals()['asp_net_sessionid'] = asp_net_sessionid
            duties_faced_by_exporter_link = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "ctl00_c_rptIDBReports_ctl08_lbReportName"))
            )
            duties_faced_by_exporter_link.click()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//span[contains(text(), "Duties faced in export markets")]'))
            )
            break
        except:
            time.sleep(abs(np.random.normal(10, 5)))
            tries += 1
    if tries == n_tries:
        raise ValueError("Failed to login to WTO TAO.")
    return driver

def post_tariff_by_yr_ex_chp_hdg(year, exporter, chapter, heading):
    aspxauth = globals().get("aspxausth", None)
    asp_net_sessionid = globals().get("asp_net_sessionid", None)

    url = "https://tao.wto.org/report/ExportMarketV2.aspx"
    payload = {
        "ctl00$manager": "ctl00$c$pnlCrit|ctl00$c$cmdRunReport",
        "ctl00$LeftNavigationMenu1$pop_top": "150px",
        "ctl00$LeftNavigationMenu1$pop_left": "250px",
        "ctl00$LeftNavigationMenu1$pop_visible": "",
        "ctl00$c$cboStartYear": "2000",
        "ctl00$c$cboEndYear": "2000",
        "ctl00$c$cboExporter": f"{country_name_code_dict[exporter]}",
        "ctl00$c$cboImporter": "A000",
        "ctl00$c$cboChapter": f"{str(chapter)}",
        "ctl00$c$rbTariffTradeLines": "rbAllTariffLines",
        "ctl00$c$cboHeading": f"{str(heading)}",
        "ctl00$c$cboSubHeading": "0",
        "ctl00$c$rbReportType": "rbDetail",
        "ctl00$c$cboDetailLevel": "SubHeading",
        "ctl00$c$hdnContentIsBig": "false",
        "ctl00$c$hdnContentOwn": "",
        "hdnContent": "",
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "__LASTFOCUS": "",
        "__VIEWSTATE": "",
        "__VSTATE": "29c56bb8-6a4e-4eff-9c3c-5db65d4e4c6d",
        "__ASYNCPOST": "true",
        "ctl00$c$cmdRunReport": "Display",
    }

    # Headers
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Cookie": f"ASP.NET_SessionId={asp_net_sessionid}; .ASPXAUTH={aspxauth}",
        "Host": "tao.wto.org",
        "Origin": "https://tao.wto.org",
        "Referer": "https://tao.wto.org/report/ExportMarketV2.aspx",
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "X-MicrosoftAjax": "Delta=true",
        "X-Requested-With": "XMLHttpRequest",
    }

    response = requests.post(url, headers=headers, data=payload)
    response.raise_for_status()
    return response.content

def get_tariff_interactively_by_yr_ex_chp_hdg(driver, year, exporter, chapter, heading):
    global country_name_code_dict
    start_year_select_element =WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "ctl00_c_cboStartYear")))
    start_year_select_element.send_keys(str(year))
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.XPATH, '//span[@class="updateProgress"]')))
    
    end_year_option = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, f'//*[@id="ctl00_c_cboEndYear"]/option[@value="{year}"]')))
    end_year_option.click()
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.XPATH, '//span[@class="updateProgress"]')))
    
    exporter_option = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, f'//*[@id="ctl00_c_cboExporter"]/option[@value="{country_name_code_dict[exporter]}"]')))
    exporter_option.click()
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.XPATH, '//span[@class="updateProgress"]')))

    chapter_option = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, f'//*[@id="ctl00_c_cboChapter"]/option[@value="{chapter}"]')))
    chapter_option.click()
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.XPATH, '//span[@class="updateProgress"]')))

    try:
        heading_option = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f'//*[@id="ctl00_c_cboHeading"]/option[@value="{heading}"]')))
    except:
        return driver, driver.page_source
    heading_option.click()
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.XPATH, '//span[@class="updateProgress"]')))

    subheading_option = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, f'//*[@id="ctl00_c_cboSubHeading"]/option[@value="0"]')))
    subheading_option.click()
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.XPATH, '//span[@class="updateProgress"]')))

    detailed_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "ctl00_c_rbDetail")))
    detailed_button.click()
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.XPATH, '//span[@class="updateProgress"]')))

    detailed_option = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_c_cboDetailLevel"]/option[@value="SubHeading"]')))
    detailed_option.click()
    WebDriverWait(driver, 10).until(
        EC.invisibility_of_element_located((By.XPATH, '//span[@class="updateProgress"]')))

    run_report_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "ctl00_c_cmdRunReport")))
    run_report_button.click()
    WebDriverWait(driver, 180).until(
        EC.invisibility_of_element_located((By.XPATH, '//span[@class="updateProgress"]')))
    time.sleep(1)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "ctl00_c_summaryReportDiv"))
    )

    return driver, driver.page_source

def parse_tariff_by_yr_ex_chp_hdg(soup, exporter, exporter_code):

    summary_report_div = soup.find(id="ctl00_c_summaryReportDiv")
    summary_table = summary_report_div.find("table")
    summary_table_rows = summary_table.find_all("tr")
    summary_table_country_rows  = [row for row in summary_table_rows if row.get("class","") in [["table2"], ["table3"]]]
    for row in summary_table_country_rows:
        if row.get("style") == "FONT-WEIGHT: bolder":
            row_tds = row.find_all("td")
            importer = row_tds[0].text
            importer_code = country_name_code_dict[importer]
            year = row_tds[1].text
        else:
            row_tds = row.find_all("td")
            commodity = row_tds[0].text
            value = row_tds[1].text
            ad_val_duty_mfn = row_tds[2].text
            ad_val_duty_best = row_tds[3].text
            n_tariff_lines = row_tds[4].text
            commodity_pattern = r'.*(\d{6}).*([A-Za-z].*)$'
            match = re.match(commodity_pattern, commodity)
            if match:
                hs6 = match.group(1)
                description = match.group(2).strip()
                hs6, description
            else:
                raise ValueError(f"Input string does not have a valid HS6 code or format. {commodity}")
            with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer.tsv", "a") as f:
                f.write(f"{exporter_code}\t{importer_code}\t{year}\t{hs6}\t{value}\t{ad_val_duty_mfn}\t{ad_val_duty_best}\t{n_tariff_lines}\n")
    return None


def get_tariff_by_yr_ex_chp_hdg(driver, year, exporter, chapter, heading, n_tries = 3):

    global critical_failures
    global posts
    global country_name_code_dict
    chapter = str(chapter).zfill(2)
    heading = str(heading).zfill(4)
    exporter_code = country_name_code_dict[exporter]

    skip = False
    with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_tracker.tsv", "r") as f:
        tracker = f.read()
    if f"{exporter_code}\t{year}\t{chapter}\t{heading}" in tracker:
        skip = True
    with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_validated_skip_tracker.tsv", "r") as f:
        validated_skip_tracker = f.read()
    if f"{exporter_code}\t{year}\t{chapter}\t{heading}" in validated_skip_tracker:
        skip = True
    if skip == True:
        return driver
    
    tries = 0
    while tries < n_tries:
        try:
            driver, content = get_tariff_interactively_by_yr_ex_chp_hdg(driver, year, exporter, chapter, heading)

            assert not re.search(r"pageRedirect.*default.aspx", content)
            soup = BeautifulSoup(content, "html.parser")
            is_chapter_selected = soup.find(id="ctl00_c_cboChapter").find("option", value=chapter, selected="selected") is not None
            is_heading_available = soup.find(id="ctl00_c_cboHeading").find("option", value=heading) is not None
            if is_chapter_selected and not is_heading_available:
                with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_validated_skip_tracker.tsv", "a") as f:
                    f.write(f"{exporter_code}\t{year}\t{chapter}\t{heading}\n")
                break
            assert soup.find(id="ctl00_c_summaryReportDiv")

            parse_tariff_by_yr_ex_chp_hdg(soup, exporter, exporter_code)

            print(f"{exporter_code}\t{year}\t{chapter}\t{heading} succeeded.")
            with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_tracker.tsv", "a") as f:
                f.write(f"{exporter_code}\t{year}\t{chapter}\t{heading}\n")
            time.sleep(abs(np.random.normal(0.5, 0.25)))
            posts += 1
            break
        except Exception as e:
            print(type(e).__name__)
            print(f"{exporter_code}\t{year}\t{chapter}\t{heading} failed. Retrying...")
            time.sleep(abs(np.random.normal(5, 2)))
            driver = login_to_wto_tao(driver)
            tries += 1
    if tries == n_tries:
        print(f"{exporter_code}\t{year}\t{chapter}\t{heading} failed. Critical failure.")
        with open("../data/raw/WTO_TAO_tariff_hs6_by_year_exporter_importer_fail_tracker.tsv", "a") as f:
            f.write(f"{exporter_code}\t{year}\t{chapter}\t{heading}\n")
        critical_failures += 1
    return driver

# %% Main
years = range(2023, 1999, -1)
exporter_codes = apac_countries.ccn3.apply(lambda x: "C" + x).tolist()
exporter_codes = [c for c in exporter_codes if c in country_code_name_dict.keys()]

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://tao.wto.org")
driver = login_to_wto_tao(driver)

critical_failures = 0
posts = 0
for year in years:

    for exporter_code in exporter_codes:
        exporter = country_code_name_dict[exporter_code]

        for chapter in chapter_heading_option['chapter_code'].unique():
            
            headings = chapter_heading_option.query(f"chapter_code == '{chapter}'")['heading_code'].unique().tolist()
            for heading in headings:
                
                driver = get_tariff_by_yr_ex_chp_hdg(driver, year, exporter, chapter, heading)                
               
                if (posts != 0) & (posts % 30 == 0) &( posts % 150 != 0):
                    time.sleep(abs(np.random.normal(10, 5)))
                    driver = login_to_wto_tao(driver)
                if (posts != 0) & (posts % 150 == 0):
                    driver.quit()
                    time.sleep(abs(np.random.normal(180, 5)))
                    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
                    driver.get("https://tao.wto.org")
                    driver = login_to_wto_tao(driver)
                if critical_failures > 3:
                    driver.quit()
                    raise ValueError("Critical failure. Exiting...")

driver.quit()



# %% Report Options
# Commented out as this is already done

# # Year
# WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_c_cboStartYear"]/option')))
# year_option_elements = driver.find_elements(By.XPATH, '//*[@id="ctl00_c_cboStartYear"]/option')
# year_option = []
# # simply put them in a list instead of a df by taking the value
# for option in year_option_elements:
#     year_option.append(option.get_attribute('value'))

# driver.find_element(By.XPATH, '//*[@id="ctl00_c_cboStartYear"]/option[@value="2020"]').click()
# WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_c_cboEndYear"]/option[@value="2020"]'))
# ).click()

# Get Exporter Options
# WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_c_cboExporter"]/option')))
# exporter_option_elements = driver.find_elements(By.XPATH, '//*[@id="ctl00_c_cboExporter"]/option')
# countries = []
# for option in exporter_option_elements[1:]:
#     country_code = option.get_attribute('value')
#     country_name = option.text
#     countries.append({"country_code": country_code, "country_name": country_name})
# country_option = pd.DataFrame(countries)
# country_option.to_csv("../data/raw/WTO_TAO_country_option.csv", index=False)
# country_name_code_dict = country_option.set_index("country_name")['country_code'].to_dict()
# driver.find_element(By.XPATH, f'//*[@id="ctl00_c_cboExporter"]/option[@value="{country_name_code_dict["China"]}"]').click()

# # Get Chapter Options
# WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_c_cboChapter"]/option')))
# chapter_option_elements = driver.find_elements(By.XPATH, '//*[@id="ctl00_c_cboChapter"]/option')
# chapters = []
# for option in chapter_option_elements[1:]:
#     chapter_code = option.get_attribute('value')
#     chapter_desc = option.text
#     chapters.append({"chapter_code": chapter_code, "chapter_description": chapter_desc})
# chapter_option = pd.DataFrame(chapters)

# headings = []
# for chapter_code in chapter_option['chapter_code']:
#     driver.find_element(By.XPATH, f'//*[@id="ctl00_c_cboChapter"]/option[@value="{chapter_code}"]').click()
#     time.sleep(np.random.normal(1, 0.5))
#     WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_c_cboHeading"]/option')))
#     heading_option_elements = driver.find_elements(By.XPATH, '//*[@id="ctl00_c_cboHeading"]/option')

#     for option in heading_option_elements[1:]:
#         heading_code = option.get_attribute('value')
#         heading_desc = option.text
#         headings.append({"chapter_code": chapter_code, "heading_code": heading_code, "heading_description": heading_desc})

# heading_option = pd.DataFrame(headings)
# chapter_heading_option = chapter_option.merge(heading_option, on = 'chapter_code', how="left")
# chapter_heading_option .to_csv("../data/raw/WTO_TAO_chapter_heading_option.csv", index=False)




