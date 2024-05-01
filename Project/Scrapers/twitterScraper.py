import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "Ruturaj Gaikwad until:2024-04-29 since:2024-01-01 -filter:links"
tweets = []
limit = 1

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content])
        
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
print(df)
