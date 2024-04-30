from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import time
import json  # Import JSON library to handle JSON data

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

team_players = {}  # Dictionary to store team names and players

for url in urls:
    driver.get(url)
    
    time.sleep(3)  # Wait for the page to fully load

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Extract the team name from the URL
    team_name = url.split('/')[-1]  # Splits the URL and takes the last part
    
    elements = soup.find_all('div', class_='ih-p-name')
    texts = [h2.text.strip() for div in elements for h2 in div.find_all('h2')]
    
    # Store texts in dictionary under their team name
    print(texts)
    team_players[team_name] = texts

driver.quit()

# Saving the dictionary to a JSON file
with open("ipl_teams_players.json", "w") as json_file:
    json.dump(team_players, json_file, indent=4)  # Pretty printing for better readability

print("Data extracted and saved to 'ipl_teams_players.json'.")
