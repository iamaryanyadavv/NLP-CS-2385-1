import pandas as pd
import re

players = 'Ravindra Jadeja, MS Dhoni, Moeen Ali, Ruturaj Gaikwad, Robin Uthappa, Dwayne Bravo, Ambati Rayudu, Deepak Chahar, KM Asif, Tushar Deshpande, Shivam Dube, Maheesh Theekshana, Rajvardhan Hangargekar, Simarjeet Singh, Devon Conway, Dwaine Pretorius, Mitchell Santner, Adam Milne, Subhranshu Senapati, Mukesh Choudhary, Prashant Solanki, C Hari Nishaanth, N Jagadeesan, Chris Jordan, K Bhagath Varma, Rishabh Pant, Prithvi Shaw, Axar Patel, Anrich Nortje, David Warner, Mitch Marsh, Shardul Thakur, Mustafizur Rahman, Kuldeep Yadav, Ashwin Hebbar, Kamlesh Nagarkoti, KS Bharat, Sarfaraz Khan, Mandeep Singh, Syed Khaleel Ahmed, Chetan Sakariya, Lalit Yadav, Ripal Patel, Yash Dhull, Rovman Powell, Pravin Dubey, Lungi Ngidi, Tim Seifert, Vicky Ostwal, Hardik Pandya, Shubman Gill, Rashid Khan, Rahmanullah Gurbaz, Mohammad Shami, Lockie Ferguson, Abhinav Sadarangani, Rahul Tewatia, Noor Ahmad, R Sai Kishore, Dominic Drakes, Jayant Yadav, Viijay Shankar, Darshan Nalkande, Yash Dayal, Alzarri Joseph, Prandeep Sangwan, David Miller, Wriddhiman Saha, Matthew Wade, Gurkeerat Singh, Varun Aaron, Sunil Narine, Andre Russell, Varun Chakravarthy, Venkatesh Iyer, Pat Cummins, Nitish Rana, Shreyas Iyer, Shivam Mavi, Sheldon Jackson, Ajinkya Rahane, Rinku Singh, Anukul Roy, Rasikh Dar, Baba Indrajith, Chamika Karunaratne, Abhijeet Tomar, Pratham Singh, Ashok Sharma, Sam Billings, Aaron Finch, Tim Southee, Ramesh Kumar, Mohammad Nabi, Umesh Yadav, KL Rahul, Marcus Stoinis, Ravi Bishnoi, Quinton de Kock, Manish Pandey, Jason Holder, Deepak Hooda, Krunal Pandya, Mark Wood (injured/replaced by Andrew Tye), Avesh Khan, Ankit Rajpoot, K Gowtham, Dushmanta Chameera, Shahbaz Nadeem, Manan Vohra, Mohsin Khan, Ayush Badoni, Kyle Mayers, Karan Sharma, Evin Lewis, Mayank Yadav, B Sai Sudharsan, Rohit Sharma, Jasprit Bumrah, Suryakumar Yadav, Kieron Pollard, Ishan Kishan, Dewald Brevis, Basil Thampi, Murugan Ashwin, Jaydev Unadkat, Mayank Markande, N Tilak Varma, Sanjay Yadav, Jofra Archer, Daniel Sams, Tymal Mills, Tim David, Riley Meredith, Mohammad Arshad Khan (injured/replaced Kumar Kartikeya Singh), Anmolpreet Singh, Ramandeep Singh, Rahul Buddhi, Hrithik Shokeen, Arjun Tendulkar, Aryan Juyal, Fabian Allen, Mayank Agarwal, Arshdeep Singh, Shikhar Dhawan, Kagiso Rabada, Jonny Bairstow, Rahul Chahar, Shahrukh Khan, Harpreet Brar, Prabhsimran Singh, Jitesh Sharma, Ishan Porel, Liam Livingstone, Odean Smith, Sandeep Sharma, Raj Bawa, Rishi Dhawan, Prerak Mankad, Vaibhav Arora, Writtick Chatterjee, Baltej Dhanda, Ansh Patel, Nathan Ellis, Atharva Taide, Bhanuka Rajapaksa, Benny Howell, Sanju Samson, Jos Buttler, Yashasvi Jaiswal, R Ashwin, Trent Boult, Shimron Hetmyer, Devdutt Padikkal, Prasidh Krishna, Yuzvendra Chahal, Riyan Parag, KC Cariappa, Navdeep Saini, Obed McCoy, Anunay Singh, Kuldeep Sen, Karun Nair, Dhruv Jurel, Tejas Baroka, Kuldip Yadav, Shubham Garhwal, Jimmy Neesham, Nathan Coulter-Nile (injured/replaced by Corbin Bosch), Rassie van der Dussen, Daryl Mitchell, Virat Kohli, Glenn Maxwell, Mohammed Siraj, Faf du Plessis, Harshal Patel, Wanindu Hasaranga, Dinesh Karthik, Anuj Rawat, Shahbaz Ahamad, Akash Deep, Josh Hazlewood, Mahipal Lomror, Finn Allen, Sherfane Rutherford, Jason Behrendorff, Suyash Prabhudessai, Chama Milind, Aneeshwar Gautam, Karn Sharma, Siddharth Kaul, Luvnith Sisodia (injury replacement: Rajat Patidar), David Willey, Kane Williamson, Abdul Samad, Umran Malik, Washington Sundar, Nicholas Pooran, T Natarajan, Bhuvneshwar Kumar, Priyam Garg, Rahul Tripathi, Abhishek Sharma, Kartik Tyagi, Shreyas Gopal, Jagadeesha Suchith, Aiden Markram, Marco Jansen, Romario Shepherd, Sean Abbott, R Samarth, Shashank Singh, Saurabh Dubey, Vishnu Vinod, Glenn Phillips, Fazalhaq Farooqi'
playerList = players.split(', ')

# Load the CSV file into a DataFrame
df = pd.read_csv('IPL_2022_tweets.csv')

# Function to check if any player's name is in the text
def contains_player(text):
    if pd.isna(text):  # Check if the text is NaN
        return False
    return any(player in text for player in playerList)

# Filter the DataFrame to include only rows where the 'text' contains any of the player names
df = df[df['text'].apply(contains_player)]

# Clean the text in the DataFrame
def clean_text(text):
    # Convert any non-string input to string
    text = str(text)
    # Convert text to lowercase
    text = text.lower()
    # Remove links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Remove characters not allowed (keep only alphanumerics, hashtags, @, and common punctuation)
    text = re.sub(r'[^\w\s#@,.!?\'\"]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Keep only the 'text' column if it's not already the only column
df = df[['text']]

# Display the head of the DataFrame to verify results
print(df.head())

# Optionally save to a new CSV
df.to_csv('filtered_and_cleaned_tweets.csv', index=False)