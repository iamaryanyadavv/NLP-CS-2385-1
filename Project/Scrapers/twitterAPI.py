import tweepy
import pandas as pd
import json

# API Keys and Tokens
consumer_key = "XweKoLpTfTOPEHhGgGfNrKlyc" #Your API/Consumer key 
consumer_secret = "cGzLNo7lh3xe8lhWsPuFlDShkwWrnMWJjZ0u1x9GDehziEDOEl" #Your API/Consumer Secret Key
access_token = "1657436525381156866-vz9CKCsnYOaKzsqyZtzGOEADzcOuH8" #Your Access token key
access_token_secret = "iB8Atg0VsxFmBW5vLImaPDdmo0uQfSrIfdLGsHYNPgOmq" #Your Access token Secret key

#Pass in our twitter API authentication key
auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret,
    access_token, access_token_secret
)

#Instantiate the tweepy API
api = tweepy.API(auth, wait_on_rate_limit=True)

# Load JSON data
with open("ipl_teams_players.json", "r") as file:
    players_data = json.load(file)
    csk_players = players_data["chennai-super-kings"]

# Calculate number of tweets to pull per player
tweets_per_player = 1500 // len(csk_players)

# Data container for all tweets
all_tweets_data = []

for player in csk_players:
    search_query = f" '{player.replace(' ', '')}' '#{player.replace(' ', '')}' -filter:retweets AND -filter:links"
    
    try:
        # Retrieve tweets
        tweets = api.search_tweets(q=search_query, lang="en", count=tweets_per_player, tweet_mode='extended')
        
        # Extract attributes from each tweet
        for tweet in tweets:
            tweet_data = {
                'Player': player,
                'Tweet Text': tweet.full_text,
                'Date of Post': tweet.created_at,
                'Likes': tweet.favorite_count,
                'Source': tweet.source,
                'Comments Count': tweet.reply_count  # Requires elevated access in Twitter API v2
            }
            all_tweets_data.append(tweet_data)
            
    except BaseException as e:
        print(f'Status Failed On {player}, Error: {str(e)}')

# Create DataFrame from the collected tweets data
tweets_df = pd.DataFrame(all_tweets_data)

# Save the DataFrame to a CSV file
tweets_df.to_csv("ipl_tweets.csv", index=False)
print("Data extracted and saved to 'ipl_tweets.csv'.")
