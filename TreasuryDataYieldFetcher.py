import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver



url = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202501'
headers = {'User-Agent': 'Mozilla/5.0'}  # Some servers require a user-agent header
response = requests.get(url, headers=headers)
if response.status_code == 200:
    html_content = response.text
else:
    raise Exception(f'Failed to retrieve the page. Status code: {response.status_code}')

soup = BeautifulSoup(html_content, 'html.parser')

# Example: Find the table containing the yield curve data
table = soup.find('table', {'class': 'data-table'})  # Adjust the class or id as per the actual HTML

# Extract table headers
headers = [header.text.strip() for header in table.find_all('th')]

# Extract table rows
rows = []
for row in table.find_all('tr')[1:]:  # Skip the header row
    columns = row.find_all('td')
    row_data = [col.text.strip() for col in columns]
    rows.append(row_data)

# Set up the Selenium WebDriver (ensure you have the appropriate driver installed)
driver = webdriver.Chrome()  # Or use webdriver.Firefox() for Firefox

driver.get(url)
html_content = driver.page_source

# Proceed with BeautifulSoup parsing as before
soup = BeautifulSoup(html_content, 'html.parser')
driver.quit()