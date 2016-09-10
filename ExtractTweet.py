__author__ = 'User'

import tweepy
import csv

# OAuth Keys
ckey = 'HdTXYIsGeBp5hwjoQT2xmE5Cs'
csecret = 'sUAxQZ9GjZZ9XyWJUMTSboKRJ3pjmJp2zgcLQOPLBPVMWFhwBz'
atoken = '603634840-O9oMqvR69dZWCet2rYZPmHl9etI0RyXztfnJVBAj'
asecret = '2oqkgE9ZiKofjjCVpBHfgOxIjEShSVOTDV5ztFP1i51nZ'

# Authentication
auth = tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api = tweepy.API(auth)
tweets = []

# tweet = api.get_status(723901641645297664)
# print(tweet.text.encode('UTF-8'))

# Write tweets of a user into .csv file
with open('demo_1.csv', 'w') as csvfile:
    fieldnames = ['Timestamp','Tweet ID', 'Tweet']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()

    for status in tweepy.Cursor(api.user_timeline, user_id='2569997099',
                                exclude_replies='false', include_rts='false').items(200):
    # for status in tweepy.Cursor(api.search, q=':) -RT', lang='en').items(10):
        writer.writerow({'Timestamp':status.created_at, 'Tweet ID':status.id_str.encode('UTF-8'),
                         'Tweet':status.text.encode('UTF-8')})
        tweets += [status.text]
        print (status.text.encode('UTF-8'))
