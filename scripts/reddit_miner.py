# reddit_miner.py
# Main Reddit Data Mining Script for USC Thesis
# Place this file in: scripts/reddit_miner.py

import praw
import pandas as pd
import datetime
from datetime import datetime, timedelta
import time
import json
import re
import os
import sys
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Import configuration
from config import REDDIT_CONFIG, USC_KEYWORDS, TARGET_SUBREDDITS, COLLECTION_PARAMS

class RedditDataMiner:
    def __init__(self):
        """
        Initialize Reddit API connection using config
        """
        self.reddit = praw.Reddit(
            client_id=REDDIT_CONFIG['client_id'],
            client_secret=REDDIT_CONFIG['client_secret'],
            user_agent=REDDIT_CONFIG['user_agent']
        )
        
        # Test connection
        try:
            print(f"Connected to Reddit as: {self.reddit.user.me()}")
        except:
            print("Connected to Reddit (read-only mode)")
        
    def search_usc_posts(self, keywords=None, subreddits=None, time_filter='all', limit=100):
        """
        Search for USC-related posts across Reddit
        """
        if keywords is None:
            keywords = USC_KEYWORDS
        if subreddits is None:
            subreddits = TARGET_SUBREDDITS
            
        posts_data = []
        
        # Combine keywords into search query
        search_query = ' OR '.join([f'"{keyword}"' for keyword in keywords])
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                print(f"Searching in r/{subreddit_name}...")
                
                # Search posts
                posts_found = 0
                for submission in subreddit.search(search_query, time_filter=time_filter, limit=limit):
                    # Filter for post-pandemic content
                    post_date = datetime.fromtimestamp(submission.created_utc)
                    if post_date >= datetime.strptime(COLLECTION_PARAMS['start_date'], '%Y-%m-%d'):
                        
                        post_data = {
                            'post_id': submission.id,
                            'subreddit': subreddit_name,
                            'title': submission.title,
                            'text': submission.selftext,
                            'score': submission.score,
                            'upvote_ratio': submission.upvote_ratio,
                            'num_comments': submission.num_comments,
                            'created_utc': submission.created_utc,
                            'created_date': post_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'author': str(submission.author),
                            'url': submission.url,
                            'permalink': f"https://reddit.com{submission.permalink}"
                        }
                        posts_data.append(post_data)
                        posts_found += 1
                        
                        # Add delay to respect rate limits
                        time.sleep(0.1)
                
                print(f"  Found {posts_found} posts in r/{subreddit_name}")
                        
            except Exception as e:
                print(f"Error searching r/{subreddit_name}: {e}")
                continue
                
        return posts_data
    
    def get_post_comments(self, post_id, limit=None):
        """
        Get comments for a specific post
        """
        if limit is None:
            limit = COLLECTION_PARAMS['comments_per_post']
            
        comments_data = []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    comment_data = {
                        'comment_id': comment.id,
                        'post_id': post_id,
                        'text': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'created_date': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'author': str(comment.author),
                        'parent_id': comment.parent_id,
                        'is_submitter': comment.is_submitter
                    }
                    comments_data.append(comment_data)
                    
        except Exception as e:
            print(f"Error getting comments for post {post_id}: {e}")
            
        return comments_data
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Categorize sentiment
            if polarity > 0.1:
                sentiment_category = 'positive'
            elif polarity < -0.1:
                sentiment_category = 'negative'
            else:
                sentiment_category = 'neutral'
                
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment_category': sentiment_category
            }
        except:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment_category': 'neutral'
            }
    
    def collect_and_analyze_data(self, output_file=None):
        """
        Complete data collection and analysis pipeline
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create absolute paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(current_dir)
            output_file = os.path.join(project_dir, 'data', 'raw', f'usc_reddit_data_{timestamp}.csv')
        
        print("Starting USC Reddit data collection...")
        print(f"Target keywords: {USC_KEYWORDS}")
        print(f"Target subreddits: {TARGET_SUBREDDITS}")
        print(f"Collection period: From {COLLECTION_PARAMS['start_date']} onwards")
        
        # Collect posts
        posts = self.search_usc_posts(limit=COLLECTION_PARAMS['posts_per_subreddit'])
        print(f"\nFound {len(posts)} posts total")
        
        if len(posts) == 0:
            print("No posts found! Try different keywords or subreddits.")
            return None
        
        # Collect comments for posts
        all_comments = []
        print("Collecting comments...")
        for i, post in enumerate(posts[:20]):  # Limit to first 20 posts for comments
            print(f"Processing post {i+1}/{min(20, len(posts))}: {post['title'][:50]}...")
            comments = self.get_post_comments(post['post_id'])
            all_comments.extend(comments)
            time.sleep(0.2)  # Rate limiting
            
        print(f"Found {len(all_comments)} comments total")
        
        # Combine and analyze data
        all_data = []
        
        # Process posts
        print("Analyzing sentiment for posts...")
        for post in posts:
            combined_text = f"{post['title']} {post['text']}"
            sentiment = self.analyze_sentiment(combined_text)
            
            post.update({
                'content_type': 'post',
                'combined_text': combined_text,
                **sentiment
            })
            all_data.append(post)
        
        # Process comments
        print("Analyzing sentiment for comments...")
        for comment in all_comments:
            sentiment = self.analyze_sentiment(comment['text'])
            comment.update({
                'content_type': 'comment',
                'combined_text': comment['text'],
                **sentiment
            })
            all_data.append(comment)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        except Exception as e:
            print(f"Error creating directory {output_dir}: {e}")
            # Fallback to current directory
            output_file = f'usc_reddit_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            print(f"Saving to current directory: {output_file}")
        
        # Save to CSV
        try:
            df = pd.DataFrame(all_data)
            df.to_csv(output_file, index=False)
            print(f"\nData saved to {output_file}")
            print(f"File size: {os.path.getsize(output_file)} bytes")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
            return None
        
        # Generate statistics and visualizations
        self.generate_basic_stats(df)
        self.create_visualizations(df, output_file)
        
        return df
    
    def generate_basic_stats(self, df):
        """
        Generate basic statistics about the collected data
        """
        print("\n" + "="*50)
        print("USC REDDIT DATA COLLECTION STATISTICS")
        print("="*50)
        
        total_posts = len(df[df['content_type'] == 'post'])
        total_comments = len(df[df['content_type'] == 'comment'])
        
        print(f"Total posts collected: {total_posts}")
        print(f"Total comments collected: {total_comments}")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        
        # Subreddit distribution
        if total_posts > 0:
            print(f"\nPosts by subreddit:")
            subreddit_counts = df[df['content_type'] == 'post']['subreddit'].value_counts()
            for subreddit, count in subreddit_counts.items():
                print(f"  r/{subreddit}: {count} posts")
        
        # Sentiment distribution
        sentiment_counts = df['sentiment_category'].value_counts()
        print(f"\nOverall sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Average scores
        if total_posts > 0:
            avg_post_score = df[df['content_type'] == 'post']['score'].mean()
            print(f"\nAverage post score: {avg_post_score:.2f}")
        
        if total_comments > 0:
            avg_comment_score = df[df['content_type'] == 'comment']['score'].mean()
            print(f"Average comment score: {avg_comment_score:.2f}")
        
        print("="*50)
    
    def create_visualizations(self, df, data_file_path):
        """
        Create visualizations of the collected data
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('USC Reddit Data Analysis', fontsize=16, fontweight='bold')
            
            # Sentiment distribution
            sentiment_counts = df['sentiment_category'].value_counts()
            colors = ['lightgreen', 'lightcoral', 'lightblue']
            axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                          autopct='%1.1f%%', colors=colors)
            axes[0, 0].set_title('Sentiment Distribution')
            
            # Posts vs Comments
            content_counts = df['content_type'].value_counts()
            axes[0, 1].bar(content_counts.index, content_counts.values, 
                          color=['skyblue', 'lightgreen'])
            axes[0, 1].set_title('Posts vs Comments')
            axes[0, 1].set_ylabel('Count')
            
            # Sentiment over time
            df['created_date'] = pd.to_datetime(df['created_date'])
            df['month_year'] = df['created_date'].dt.to_period('M')
            
            if len(df['month_year'].unique()) > 1:
                monthly_sentiment = df.groupby(['month_year', 'sentiment_category']).size().unstack(fill_value=0)
                monthly_sentiment.plot(kind='bar', ax=axes[1, 0], stacked=True)
                axes[1, 0].set_title('Sentiment Over Time (Monthly)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].legend(title='Sentiment')
            else:
                axes[1, 0].text(0.5, 0.5, 'Not enough data\nfor time analysis', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Sentiment Over Time')
            
            # Polarity distribution
            axes[1, 1].hist(df['polarity'], bins=20, alpha=0.7, color='purple')
            axes[1, 1].set_title('Sentiment Polarity Distribution')
            axes[1, 1].set_xlabel('Polarity Score (-1 to 1)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Create results directory using same base path as data file
            if os.path.dirname(data_file_path):
                project_dir = os.path.dirname(os.path.dirname(data_file_path))
                results_dir = os.path.join(project_dir, 'results')
            else:
                results_dir = 'results'
            
            try:
                os.makedirs(results_dir, exist_ok=True)
                viz_filename = os.path.join(results_dir, f'usc_reddit_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {viz_filename}")
            except Exception as e:
                print(f"Error saving visualization: {e}")
                # Fallback to current directory
                viz_filename = f'usc_reddit_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to current directory: {viz_filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            print("Continuing without visualizations...")

def main():
    """
    Main function to run the Reddit data mining
    """
    print("USC Thesis - Reddit Data Mining")
    print("================================")
    
    # Initialize miner
    try:
        miner = RedditDataMiner()
        
        # Collect and analyze data
        df = miner.collect_and_analyze_data()
        
        if df is not None:
            print(f"\nData collection completed successfully!")
            print(f"Total records collected: {len(df)}")
            print(f"You can find your data in the 'data/raw' folder")
            print(f"Visualizations are saved in the 'results' folder")
        else:
            print("\nData collection failed. Please check your configuration.")
            
    except Exception as e:
        print(f"Error during data collection: {e}")
        print("Please check your Reddit API credentials in config.py")

if __name__ == "__main__":
    main()