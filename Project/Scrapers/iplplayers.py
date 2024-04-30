from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import time

driver = webdriver.Chrome()

urls = [
    "https://www.iplt20.com/teams/chennai-super-kings",
    "https://www.iplt20.com/teams/delhi-capitals",
    "https://www.iplt20.com/teams/gujarat-titans",
    "https://www.iplt20.com/teams/kolkata-knight-riders",
    "https://www.iplt20.com/teams/lucknow-super-giants",
    "https://www.iplt20.com/teams/mumbai-indians",
    "https://www.iplt20.com/teams/punjab-kings",
    "https://www.iplt20.com/teams/rajasthan-royals",
    "https://www.iplt20.com/teams/royal-challengers-bangalore",
    "https://www.iplt20.com/teams/sunrisers-hyderabad"
]

all_texts = []

for url in urls:
    driver.get(url)
    
    time.sleep(3) 

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    elements = soup.find_all('div', class_='ih-p-name')
    texts = [h2.text for div in elements for h2 in div.find_all('h2')]
    
    joined_texts = ', '.join(texts)
    print(f"Texts from {url}: {joined_texts}")
    
    all_texts.extend(texts)

driver.quit()

final_joined_texts = ', '.join(all_texts)
print(f"\nFinal aggregated texts: {final_joined_texts}")

with open("iplplayers.txt", "w") as file:
    file.write(final_joined_texts)

print("Data extracted and saved to 'iplplayers.txt'.")
