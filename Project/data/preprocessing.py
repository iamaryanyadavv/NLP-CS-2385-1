import pandas as pd

# Sample DataFrame loading
# Assuming df is your DataFrame with a 'text' column for tweets and 'sentiment' for sentiment
df = pd.read_csv('sentiment_labeled_tweets.csv')
# Remove duplicates based on the 'text' column
df = df.drop_duplicates(subset=['text'], keep='first')  # keep='first' keeps the first occurrence

playerString = 'Ravindra Jadeja, MS Dhoni, Moeen Ali, Ruturaj Gaikwad, Robin Uthappa, Dwayne Bravo, Ambati Rayudu, Deepak Chahar, KM Asif, Tushar Deshpande, Shivam Dube, Maheesh Theekshana, Rajvardhan Hangargekar, Simarjeet Singh, Devon Conway, Dwaine Pretorius, Mitchell Santner, Adam Milne, Subhranshu Senapati, Mukesh Choudhary, Prashant Solanki, C Hari Nishaanth, N Jagadeesan, Chris Jordan, K Bhagath Varma, Rishabh Pant, Prithvi Shaw, Axar Patel, Anrich Nortje, David Warner, Mitch Marsh, Shardul Thakur, Mustafizur Rahman, Kuldeep Yadav, Ashwin Hebbar, Kamlesh Nagarkoti, KS Bharat, Sarfaraz Khan, Mandeep Singh, Syed Khaleel Ahmed, Chetan Sakariya, Lalit Yadav, Ripal Patel, Yash Dhull, Rovman Powell, Pravin Dubey, Lungi Ngidi, Tim Seifert, Vicky Ostwal, Hardik Pandya, Shubman Gill, Rashid Khan, Rahmanullah Gurbaz, Mohammad Shami, Lockie Ferguson, Abhinav Sadarangani, Rahul Tewatia, Noor Ahmad, R Sai Kishore, Dominic Drakes, Jayant Yadav, Viijay Shankar, Darshan Nalkande, Yash Dayal, Alzarri Joseph, Prandeep Sangwan, David Miller, Wriddhiman Saha, Matthew Wade, Gurkeerat Singh, Varun Aaron, Sunil Narine, Andre Russell, Varun Chakravarthy, Venkatesh Iyer, Pat Cummins, Nitish Rana, Shreyas Iyer, Shivam Mavi, Sheldon Jackson, Ajinkya Rahane, Rinku Singh, Anukul Roy, Rasikh Dar, Baba Indrajith, Chamika Karunaratne, Abhijeet Tomar, Pratham Singh, Ashok Sharma, Sam Billings, Aaron Finch, Tim Southee, Ramesh Kumar, Mohammad Nabi, Umesh Yadav, KL Rahul, Marcus Stoinis, Ravi Bishnoi, Quinton de Kock, Manish Pandey, Jason Holder, Deepak Hooda, Krunal Pandya, Mark Wood (injured/replaced by Andrew Tye), Avesh Khan, Ankit Rajpoot, K Gowtham, Dushmanta Chameera, Shahbaz Nadeem, Manan Vohra, Mohsin Khan, Ayush Badoni, Kyle Mayers, Karan Sharma, Evin Lewis, Mayank Yadav, B Sai Sudharsan, Rohit Sharma, Jasprit Bumrah, Suryakumar Yadav, Kieron Pollard, Ishan Kishan, Dewald Brevis, Basil Thampi, Murugan Ashwin, Jaydev Unadkat, Mayank Markande, N Tilak Varma, Sanjay Yadav, Jofra Archer, Daniel Sams, Tymal Mills, Tim David, Riley Meredith, Mohammad Arshad Khan (injured/replaced Kumar Kartikeya Singh), Anmolpreet Singh, Ramandeep Singh, Rahul Buddhi, Hrithik Shokeen, Arjun Tendulkar, Aryan Juyal, Fabian Allen, Mayank Agarwal, Arshdeep Singh, Shikhar Dhawan, Kagiso Rabada, Jonny Bairstow, Rahul Chahar, Shahrukh Khan, Harpreet Brar, Prabhsimran Singh, Jitesh Sharma, Ishan Porel, Liam Livingstone, Odean Smith, Sandeep Sharma, Raj Bawa, Rishi Dhawan, Prerak Mankad, Vaibhav Arora, Writtick Chatterjee, Baltej Dhanda, Ansh Patel, Nathan Ellis, Atharva Taide, Bhanuka Rajapaksa, Benny Howell, Sanju Samson, Jos Buttler, Yashasvi Jaiswal, R Ashwin, Trent Boult, Shimron Hetmyer, Devdutt Padikkal, Prasidh Krishna, Yuzvendra Chahal, Riyan Parag, KC Cariappa, Navdeep Saini, Obed McCoy, Anunay Singh, Kuldeep Sen, Karun Nair, Dhruv Jurel, Tejas Baroka, Kuldip Yadav, Shubham Garhwal, Jimmy Neesham, Nathan Coulter-Nile (injured/replaced by Corbin Bosch), Rassie van der Dussen, Daryl Mitchell, Virat Kohli, Glenn Maxwell, Mohammed Siraj, Faf du Plessis, Harshal Patel, Wanindu Hasaranga, Dinesh Karthik, Anuj Rawat, Shahbaz Ahamad, Akash Deep, Josh Hazlewood, Mahipal Lomror, Finn Allen, Sherfane Rutherford, Jason Behrendorff, Suyash Prabhudessai, Chama Milind, Aneeshwar Gautam, Karn Sharma, Siddharth Kaul, Luvnith Sisodia (injury replacement: Rajat Patidar), David Willey, Kane Williamson, Abdul Samad, Umran Malik, Washington Sundar, Nicholas Pooran, T Natarajan, Bhuvneshwar Kumar, Priyam Garg, Rahul Tripathi, Abhishek Sharma, Kartik Tyagi, Shreyas Gopal, Jagadeesha Suchith, Aiden Markram, Marco Jansen, Romario Shepherd, Sean Abbott, R Samarth, Shashank Singh, Saurabh Dubey, Vishnu Vinod, Glenn Phillips, Fazalhaq Farooqi'
playerList = playerString.split(', ')

# Normalize player names for case-insensitive matching
players = [player.lower() for player in playerList]

# Initialize a dictionary to hold tweets for each player
player_tweets = {player: [] for player in players}

# Iterate over each row in the DataFrame
for _, row in df.iterrows():
    tweet = row['text'].lower()  # Convert tweet to lowercase for case-insensitive comparison
    sentiment = row['Sentiment']
    
    # Check if tweet mentions any players
    for player in players:
        if player in tweet:  # Check if the player's name is in the tweet
            # Append the tweet along with its sentiment to the appropriate player's list
            player_tweets[player].append((tweet, sentiment))

# Create a new DataFrame from the dictionary
# Each entry in the list becomes a row; if different players have different numbers of tweets, this can be an issue,
# so you might need to handle varying lengths if it causes problems
player_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in player_tweets.items()]))

print(player_df.head())

# Save to CSV
player_df.to_csv('player_tweets.csv', index=False)
