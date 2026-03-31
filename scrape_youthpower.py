from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# CONFIGURATION
DISTRICTS = [
    'bagalkot', 'ballari', 'belagavi', 'bengaluru-rural', 'bengaluru-urban',
    'bidar', 'chamarajanagar', 'chikballapur', 'chikkamagaluru', 'chitradurga',
    'dakshina-kannada', 'davangere', 'dharwad', 'gadag', 'hassan', 'haveri',
    'kalaburagi', 'kodagu', 'kolar', 'koppal', 'mandya', 'mysuru', 'raichur',
    'ramanagara', 'shivamogga', 'tumakuru', 'udupi', 'uttara-kannada',
    'vijayanagara', 'vijayapura', 'yadgir'
]

METRICS_H3 = [
    "Total Population", "Total Youth Population", "Savings per Working Age Individual",
    "Mudra Loan to Labor Force Ratio", "CSR Spending per Capita (Rs.)", "CSR Share (% of State)",
    "GDP Growth Rate (%)", "Per Capita Income (Rs.)", "Trains per Week per 1000 sqkm",
    "MSMEs per 10k Population", "EPFO Firms", "Number of Jobs", "Employment in all MSMEs",
    "Registered Unorganised Workers", "Labor force participation (%)", "Unemployment rate (%)",
    "EPFO Coverage Rate (% of the Labor Force)", "Women Hostels", "Schools", "Enrollment Ratio",
    "GER, Higher Education", "Test scores (%)", "Number of Colleges", "Libraries",
    "ITI Seats per 1 lac Youth", "Seats in Top 3 Trades", "Sanctioned Trainers",
    "PMKVY Enrollment per 1 Lac Youth", "PMKVY Enrollments in Top 3 Jobs (%)",
    "Apprentices per 1 Lac Youth", "Enrolment (6-12 average)", "ITI Certified Trainers",
    "PMKVY Trainees Certified %", "Learning Outcomes (average)", "ITI Seats Vacancy %",
    "ITI Trainer Vacancy %"
]

SCORES = [
    "Y-Power Score", "Opportunity", "Workforce", "Education", "Readiness and Skills"
]

def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--window-size=1920,3000")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def get_value(driver, label):
    try:
        # Categorical Scores (Typically Y-Power Score, etc.)
        # These are usually followed by a span with the score
        score_xpaths = [
            f"//*[text()='{label}']/following::span[1]",
            f"//*[contains(text(), '{label}')]/following::span[1]"
        ]

        # Grid Metrics (Typically in h3 cards)
        metric_xpaths = [
            f"//*[text()='{label}']/following::div[contains(@class, 'font-bold') or contains(@class, 'text-2xl')][1]",
            f"//*[contains(text(), '{label}')]/following::div[contains(@class, 'font-bold') or contains(@class, 'text-2xl')][1]",
            f"//*[text()='{label}']/ancestor::div[1]/following-sibling::div[1]"
        ]

        # Try Score XPaths first if it's a score label
        if label in SCORES:
            for xpath in score_xpaths:
                try:
                    el = driver.find_element(By.XPATH, xpath)
                    val = el.text.strip()
                    if val and val != "i":
                        if "/" in val: val = val.split("/")[0].strip()
                        if val.isdigit() or (val.replace('.','',1).isdigit()):
                            return val
                except:
                    continue

        # Try Metric XPaths
        for xpath in metric_xpaths:
            try:
                el = driver.find_element(By.XPATH, xpath)
                val = el.text.strip()
                if val and val != "i":
                    if "Avg:" in val:
                        val = val.split("Avg:")[0].strip()
                    return val
            except:
                continue

        return "N/A"
    except:
        return "N/A"

def scrape_all():
    driver = get_driver()
    all_data = []

    try:
        for idx, dist in enumerate(DISTRICTS):
            print(f"[{idx+1}/{len(DISTRICTS)}] Scraping: {dist}...", flush=True)
            url = f"https://youthpower.in/scorecard?state=karnataka&district={dist}"
            driver.get(url)

            try:
                # Wait for main content
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Y-Power Score')]")))

                # Wait specifically for the main score to populate (usually a digit)
                # This ensures dynamic JS components have finished rendering values
                try:
                    WebDriverWait(driver, 10).until(lambda d: d.find_element(By.XPATH, "//p[contains(text(), 'Y-Power Score')]/following-sibling::span[1]").text.strip().isdigit())
                except:
                    time.sleep(2) # Fallback sleep if digit wait fails (it might be 'N/A')

                row = {"District": dist.replace("-", " ").title()}

                # Capture all desired data
                for label in SCORES + METRICS_H3:
                    row[label] = get_value(driver, label)

                all_data.append(row)
                print(f"  Done. Score: {row['Y-Power Score']}")

            except Exception as e:
                print(f"  Failed to scrape {dist}: {e}")

            if (idx + 1) % 5 == 0:
                pd.DataFrame(all_data).to_csv("karnataka_complete_scorecard_partial.csv", index=False)

    finally:
        driver.quit()

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv("karnataka_complete_scorecard.csv", index=False)
        print("\nSUCCESS: Data saved to karnataka_complete_scorecard.csv")
    else:
        print("\nERROR: No data was captured.")

if __name__ == "__main__":
    scrape_all()
