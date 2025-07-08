# # import tweepy
# import pandas as pd
# import time
# import os
# from datetime import datetime, timedelta
# import logging
# from config import Config
# import json

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TwitterMiner:
#     """Dedicated Twitter/X mining for USC mentions"""
    
#     def __init__(self):
#         self.config = Config()
#         self.client = None
#         self.setup_twitter_client()
        
#         # USC-specific search terms
#         self.usc_keywords = [
#             "University of San Carlos",
#             "USC Cebu",
#             "USC Philippines", 
#             "USC Talamban",
#             "USC Downtown",
#             "Carolinian USC"
#         ]
        
#         # Post-pandemic focus (March 2020 onwards)
#         self.start_date = datetime(2020, 3, 1)
    
#     def setup_twitter_client(self):
#         """Initialize Twitter API client"""
#         try:
#             self.client = tweepy.Client(
#                 bearer_token=self.config.TWITTER_BEARER_TOKEN,
#                 consumer_key=self.config.TWITTER_API_KEY,
#                 consumer_secret=self.config.TWITTER_API_SECRET,
#                 access_token=self.config.TWITTER_ACCESS_TOKEN,
#                 access_token_secret=self.config.TWITTER_ACCESS_TOKEN_SECRET,
#                 wait_on_rate_limit=True
#             )
#             logger.info(" Twitter client initialized successfully")
#         except Exception as e:
#             logger.error(f" Error setting up Twitter client: {e}")
#             raise
    
#     def search_tweets(self, keyword, max_results=100):
#         """Search for tweets containing USC keywords"""
#         tweets_data = []
        
#         try:
#             # Build search query
#             query = f'"{keyword}" -is:retweet lang:en'
            
#             logger.info(f" Searching for: {keyword}")
            
#             # Search recent tweets (last 7 days with basic access)
#             tweets = tweepy.Paginator(
#                 self.client.search_recent_tweets,
#                 query=query,
#                 max_results=100,
#                 tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations', 'geo', 'lang'],
#                 user_fields=['name', 'username', 'location', 'verified'],
#                 expansions=['author_id'],
#                 limit=max_results//100
#             ).flatten()
            
#             for tweet in tweets:
#                 tweet_data = {
#                     'tweet_id': tweet.id,
#                     'author_id': tweet.author_id,
#                     'text': tweet.text,
#                     'created_at': tweet.created_at,
#                     'like_count': tweet.public_metrics['like_count'],
#                     'retweet_count': tweet.public_metrics['retweet_count'],
#                     'reply_count': tweet.public_metrics['reply_count'],
#                     'quote_count': tweet.public_metrics['quote_count'],
#                     'url': f"https://twitter.com/user/status/{tweet.id}",
#                     'keyword_searched': keyword,
#                     'language': tweet.lang
#                 }
#                 tweets_data.append(tweet_data)
                
#             logger.info(f" Found {len(tweets_data)} tweets for '{keyword}'")
            
#         except Exception as e:
#             logger.error(f" Error searching tweets for '{keyword}': {e}")
        
#         return tweets_data
    
#     def mine_all_keywords(self, tweets_per_keyword=200):
#         """Mine tweets for all USC keywords"""
#         all_tweets = []
        
#         logger.info(" Starting Twitter mining for USC...")
        
#         for keyword in self.usc_keywords:
#             tweets = self.search_tweets(keyword, tweets_per_keyword)
#             all_tweets.extend(tweets)
            
#             # Rate limiting - be nice to the API
#             time.sleep(2)
        
#         # Remove duplicates based on tweet_id
#         unique_tweets = []
#         seen_ids = set()
        
#         for tweet in all_tweets:
#             if tweet['tweet_id'] not in seen_ids:
#                 unique_tweets.append(tweet)
#                 seen_ids.add(tweet['tweet_id'])
        
#         logger.info(f"🎯 Total unique tweets collected: {len(unique_tweets)}")
#         return unique_tweets
    
#     def save_to_csv(self, tweets_data, filename=None):
#         """Save tweets to CSV file"""
#         if not tweets_data:
#             logger.warning(" No tweets to save")
#             return
        
#         # Create DataFrame
#         df = pd.DataFrame(tweets_data)
        
#         # Generate filename if not provided
#         if not filename:
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             filename = f'usc_twitter_data_{timestamp}.csv'
        
#         # Save to raw data directory
#         filepath = os.path.join('data', 'raw', filename)
#         df.to_csv(filepath, index=False, encoding='utf-8')
        
#         logger.info(f" Saved {len(tweets_data)} tweets to {filepath}")
        
#         # Generate summary
#         self.generate_summary(df, filename)
        
#         return filepath
    
#     def generate_summary(self, df, filename):
#         """Generate summary statistics"""
#         summary = {
#             'total_tweets': len(df),
#             'date_range': {
#                 'start': df['created_at'].min().isoformat(),
#                 'end': df['created_at'].max().isoformat()
#             },
#             'engagement_stats': {
#                 'total_likes': df['like_count'].sum(),
#                 'total_retweets': df['retweet_count'].sum(),
#                 'total_replies': df['reply_count'].sum(),
#                 'avg_likes': df['like_count'].mean(),
#                 'avg_retweets': df['retweet_count'].mean()
#             },
#             'keyword_breakdown': df['keyword_searched'].value_counts().to_dict(),
#             'top_engaging_tweets': df.nlargest(5, 'like_count')[['text', 'like_count', 'retweet_count']].to_dict('records')
#         }
        
#         # Save summary
#         summary_filename = filename.replace('.csv', '_summary.json')
#         summary_filepath = os.path.join('data', 'processed', summary_filename)
        
#         with open(summary_filepath, 'w', encoding='utf-8') as f:
#             json.dump(summary, f, indent=2, default=str)
        
#         logger.info(f" Summary saved to {summary_filepath}")
        
#         # Print quick stats
#         print("\n" + "="*50)
#         print(" TWITTER MINING SUMMARY")
#         print("="*50)
#         print(f"Total tweets: {summary['total_tweets']}")
#         print(f"Total likes: {summary['engagement_stats']['total_likes']:,}")
#         print(f"Total retweets: {summary['engagement_stats']['total_retweets']:,}")
#         print(f"Average likes per tweet: {summary['engagement_stats']['avg_likes']:.1f}")
#         print("\nKeyword breakdown:")
#         for keyword, count in summary['keyword_breakdown'].items():
#             print(f"  • {keyword}: {count} tweets")
#         print("="*50)

# def main():
#     """Main function to run Twitter mining"""
#     try:
#         # Initialize miner
#         miner = TwitterMiner()
        
#         # Mine tweets
#         tweets = miner.mine_all_keywords(tweets_per_keyword=200)
        
#         # Save results
#         if tweets:
#             miner.save_to_csv(tweets)
#             print(" Twitter mining completed successfully!")
#         else:
#             print(" No tweets found. Check your API credentials and search terms.")
            
#     except Exception as e:
#         logger.error(f" Error in main execution: {e}")
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     main()