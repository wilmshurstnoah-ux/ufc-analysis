# 02_scrape.py - scrapes recent UFC events from UFCStats.com
# run from project root: python scripts/02_scrape.py

import pandas as pd
import requests
import time
import os
from bs4 import BeautifulSoup

url = "http://www.ufcstats.com/statistics/events/completed"
output_path = "data/raw/scraped_events.csv"

print(f"Scraping from: {url}")

# get the page
try:
    response = requests.get(url, timeout=10)
except Exception as e:
    print(f"Could not reach the site - {e}")
    exit()

if response.status_code != 200:
    print(f"Something went wrong, status code: {response.status_code}")
    exit()

time.sleep(1) # dont spam the server

soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table")

if not table:
    print("Couldnt find the table on the page")
    exit()

# loop through each row and grab the event info
rows = table.find_all("tr")
events = []

for row in rows:
    cols = row.find_all("td")
    if len(cols) < 2:
        continue

    link = cols[0].find("a")
    if not link:
        continue
    event_name = link.get_text(strip=True)

    date_span = cols[0].find("span")
    if date_span:
        event_date = date_span.get_text(strip=True)
    else:
        event_date = ""

    event_location = cols[1].get_text(strip=True) # second column is location

    events.append({"event_name": event_name, "date": event_date, "location": event_location})

# save to csv
df = pd.DataFrame(events)

if df.empty:
    print("No events found - page layout might have changed")
    exit()

os.makedirs("data/raw", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\nScraped {len(df)} events and saved to {output_path}")
print(df.head())
