from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

districts = ["ramanagara", "bagalkot", "ballari"]

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
chrome_options.add_argument("--window-size=1920,1080")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

def get_data(label):
    try:
        if label == "Y-Power Score":
            xpaths = [
                "//p[contains(text(), 'Y-Power Score')]/following-sibling::span[1]",
                "//span[contains(text(), 'Y-Power Score')]/following::span[contains(@class, 'font-bold')][1]",
                "//p[contains(text(), 'Y-Power Score')]/following::span[1]",
                "//*[contains(@class, 'text-[#E26F45]')]"
            ]
            for xpath in xpaths:
                try:
                    el = driver.find_element(By.XPATH, xpath)
                    val = el.text.strip()
                    if val and (val.isdigit() or "/" in val):
                        if "/" in val: val = val.split("/")[0].strip()
                        return val
                except:
                    continue
            return "N/A"

        # General approach for most metrics
        xpaths = [
            f"//*[text()='{label}']/following::div[contains(@class, 'font-bold') or contains(@class, 'text-2xl')][1]",
            f"//*[contains(text(), '{label}')]/following::div[contains(@class, 'font-bold') or contains(@class, 'text-2xl')][1]",
            f"//*[text()='{label}']/ancestor::div[1]/following-sibling::div[1]",
            f"//*[contains(text(), '{label}')]/ancestor::div[1]/following-sibling::div[1]"
        ]

        for xpath in xpaths:
            try:
                element = driver.find_element(By.XPATH, xpath)
                val = element.text.strip()
                if val:
                    if "Avg:" in val:
                        val = val.split("Avg:")[0].strip()
                    if val != "i":
                        return val
            except:
                continue

        return "N/A"
    except Exception:
        return "N/A"

all_data = []

for dist in districts:
    print(f"Scraping data for: {dist}...", flush=True)
    url = f"https://youthpower.in/scorecard?state=karnataka&district={dist}"
    driver.get(url)

    try:
        # Wait for the page to load
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Y-Power Score')]")))

        # Give it some time to finish animations/loading
        time.sleep(5)

        row = {
            "District": dist.title(),
            "Y-Power Score": get_data("Y-Power Score"),
            "Total Population": get_data("Total Population"),
            "Youth Population": get_data("Total Youth Population"),
            "Savings per Individual": get_data("Savings per Working Age Individual"),
            "Mudra Loan Ratio": get_data("Mudra Loan to Labor Force Ratio"),
            "CSR Spending": get_data("CSR Spending per Capita (Rs.)"),
            "GDP Growth Rate": get_data("GDP Growth Rate (%)"),
            "Per Capita Income": get_data("Per Capita Income (Rs.)"),
            "MSMEs per 10k": get_data("MSMEs per 10k Population"),
            "EPFO Firms": get_data("EPFO Firms")
        }
        all_data.append(row)
        print(f"  Captured {dist}: Score {row['Y-Power Score']}")

    except Exception as e:
        print(f"  Failed to scrape {dist}: {e}")

driver.quit()

# Final export
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv("karnataka_opportunity_complete.csv", index=False)
    print(f"\nCSV saved: karnataka_opportunity_complete.csv\n")
    print(df.to_string(index=False))
else:
    print("No data captured.")
