# enhanced_reddit_miner.py
# Enhanced Reddit Data Mining Script with Code-Switching Detection
# Focus on USC-related posts with Bisaya, Tagalog, and Conyo code-switching

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
from collections import Counter
import numpy as np
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enhanced configuration for code-switching detection
class CodeSwitchingDetector:
    def __init__(self):
        # Filipino/Bisaya/Tagalog keywords and patterns
        self.bisaya_words = [
            'uy', 'bay', 'bai', 'dong', 'daw', 'gud', 'naa', 'wala', 'bitaw', 'mao',
            'kay', 'unya', 'unsa', 'asa', 'kinsa', 'kanus-a', 'ngano', 'ayaw', 'wa',
            'nya', 'ra', 'man', 'pud', 'pod', 'sad', 'gani', 'lagi', 'jud', 'gyud',
            'kaayo', 'lisod', 'sayon', 'dili', 'ayaw', 'pwede', 'murag', 'para',
            'mga', 'diha', 'diri', 'nganong', 'unsaon', 'pila', 'daghan', 'gamay',
            'kusog', 'hinay', 'bisan', 'kung', 'pero', 'unya', 'dayon', 'usa',
            'duha', 'tulo', 'upat', 'lima', 'walo', 'syete', 'siyam', 'napulo',
            'buntag', 'hapon', 'gabii', 'ugma', 'gahapon', 'karon', 'kaniadto',
            'lami', 'humot', 'baho', 'init', 'bugnaw', 'ulan', 'adlaw', 'buwan',
            'tuig', 'semana', 'oras', 'minuto', 'segundo', 'balay', 'kwarto',
            'kusina', 'banyo', 'sala', 'higdaanan', 'lamesa', 'lingkoranan',
            'bisaya', 'cebuano', 'binisaya'
        ]
        
        self.tagalog_words = [
            'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'tayo', 'natin', 'namin',
            'ninyo', 'nila', 'kita', 'mo', 'ko', 'ka', 'niya', 'namin', 'ninyo',
            'nila', 'ito', 'iyan', 'iyon', 'dito', 'diyan', 'doon', 'nandito',
            'nandiyan', 'nandoon', 'hindi', 'oo', 'opo', 'hindi', 'wala', 'may',
            'meron', 'mayroon', 'kung', 'kapag', 'pag', 'para', 'dahil', 'kasi',
            'pero', 'ngunit', 'at', 'o', 'na', 'nga', 'naman', 'din', 'rin',
            'lang', 'lamang', 'pa', 'na', 'eh', 'ah', 'oh', 'hay', 'naku',
            'talaga', 'totoo', 'totoong', 'ganda', 'maganda', 'pangit', 'tama',
            'mali', 'mabuti', 'masama', 'ayos', 'okay', 'sige', 'tara', 'halika',
            'pumunta', 'umuwi', 'umalis', 'bumalik', 'kumain', 'uminom', 'matulog',
            'gumising', 'maligo', 'magbihis', 'mag-aral', 'magtrabaho', 'maglaro',
            'manood', 'makinig', 'magbasa', 'magsulat', 'maglakad', 'tumakbo',
            'tagalog', 'filipino', 'pinoy', 'pilipino'
        ]
        
        self.conyo_words = [
            'like', 'kasi', 'ganon', 'ganun', 'yung', 'yun', 'naman', 'talaga',
            'super', 'grabe', 'sobrang', 'ang', 'sa', 'ng', 'mga', 'yung',
            'ano', 'ba', 'eh', 'kaya', 'lang', 'naman', 'diba', 'di ba',
            'omg', 'wtf', 'lol', 'haha', 'hehe', 'hihi', 'char', 'chos',
            'naks', 'oks', 'pwede', 'sige', 'tara', 'go', 'sus', 'sis',
            'bro', 'kuya', 'ate', 'tito', 'tita', 'lola', 'lolo', 'mama',
            'papa', 'nanay', 'tatay', 'anak', 'kapatid', 'pinsan', 'barkada',
            'tropa', 'bestie', 'bff', 'crush', 'jowa', 'syota', 'kras',
            'kilig', 'bitter', 'emo', 'jologs', 'baduy', 'sosyal', 'burgis',
            'jejemon', 'feelingera', 'feelingero', 'chismosa', 'chismoso',
            'maarte', 'malandi', 'pakipot', 'torpe', 'assuming', 'deadma',
            'ghosting', 'seen zone', 'friendzone', 'conyo', 'jeje'
        ]
        
        # Mixed patterns (English + Filipino)
        self.mixed_patterns = [
            r'\b(kasi|pero|tapos|then|and then|kaya|so|like)\b.*\b(english|tagalog|bisaya)\b',
            r'\b(i|you|we|they)\b.*\b(naman|talaga|nga|ba|eh)\b',
            r'\b(super|very|really)\b.*\b(ganda|pangit|ayos|okay)\b',
            r'\b(my|your|our|their)\b.*\b(ate|kuya|tito|tita)\b',
            r'\b(grabe|sobrang|ang)\b.*\b(nice|good|bad|cool)\b'
        ]
        
        # USC-specific terms
        self.usc_terms = [
            'usc', 'university of san carlos', 'san carlos', 'carolinian', 'carolinians',
            'talamban', 'downtown', 'main', 'north', 'south', 'campus', 'tc', 'dc',
            'usc-tc', 'usc-dc', 'usc main', 'usc north', 'usc south', 'usc talamban',
            'casaa', 'engineering', 'business', 'education', 'arts', 'sciences',
            'medicine', 'nursing', 'pharmacy', 'dentistry', 'law', 'architecture',
            'carolinian', 'green and gold', 'warriors', 'usc warriors'
        ]
    
    def detect_code_switching(self, text):
        """
        Detect code-switching in text and return analysis
        """
        text_lower = text.lower()
        
        # Count occurrences of each language
        bisaya_count = sum(1 for word in self.bisaya_words if word in text_lower)
        tagalog_count = sum(1 for word in self.tagalog_words if word in text_lower)
        conyo_count = sum(1 for word in self.conyo_words if word in text_lower)
        
        # Check for mixed patterns
        mixed_patterns_found = sum(1 for pattern in self.mixed_patterns if re.search(pattern, text_lower))
        
        # Check for USC relevance
        usc_relevance = sum(1 for term in self.usc_terms if term in text_lower)
        
        # Determine if code-switching is present
        total_filipino_words = bisaya_count + tagalog_count + conyo_count
        has_code_switching = total_filipino_words >= 2 or mixed_patterns_found > 0
        
        # Determine dominant language mix
        language_mix = []
        if bisaya_count > 0:
            language_mix.append('bisaya')
        if tagalog_count > 0:
            language_mix.append('tagalog')
        if conyo_count > 0:
            language_mix.append('conyo')
        
        return {
            'has_code_switching': has_code_switching,
            'bisaya_count': bisaya_count,
            'tagalog_count': tagalog_count,
            'conyo_count': conyo_count,
            'mixed_patterns_found': mixed_patterns_found,
            'total_filipino_words': total_filipino_words,
            'language_mix': language_mix,
            'usc_relevance_score': usc_relevance,
            'code_switching_score': total_filipino_words + mixed_patterns_found
        }

class EnhancedRedditMiner:
    def __init__(self):
        """Initialize with enhanced configuration"""
        # Reddit API configuration - you'll need to add your credentials
        self.reddit = praw.Reddit(
            client_id='TG2zWWhi5QJNS8kIT5HgRQ',
            client_secret='Pk8S4MauTq0lQtg5IXfrnpIvRi_Rrw',
            user_agent='USC_Thesis_Research_v1.0 by u/Dizzy-Language-6383'
        )
        
        self.code_detector = CodeSwitchingDetector()
        
        # Enhanced keywords for USC
        self.usc_keywords = [
            'university of san carlos', 'usc', 'san carlos university',
            'carolinian', 'carolinians', 'usc talamban', 'usc downtown',
            'usc main', 'usc tc', 'usc dc', 'usc warriors', 'green and gold',
            'usc cebu', 'usc philippines', 'casaa', 'usc engineering',
            'usc business', 'usc medicine', 'usc nursing', 'usc law',
            'usc architecture', 'usc education', 'usc arts', 'usc sciences'
        ]
        
        # Expanded subreddit list
        self.target_subreddits = [
            'Philippines', 'Cebu', 'studentsph', 'CollegeStudentsph',
            'UniversityOfThePhilippines', 'phinvest', 'phr4r', 'phclassifieds',
            'phcareers', 'phgamers', 'phbooks', 'phtech', 'phtravel',
            'casualph', 'askph', 'phstudents', 'phuniversity', 'phcollege',
            'cebuano', 'bisaya', 'visayas', 'mindanao', 'luzon'
        ]
        
        # Collection parameters
        self.collection_params = {
            'posts_per_subreddit': 200,
            'comments_per_post': 50,
            'max_posts_total': 1000,
            'max_comments_total': 5000,
            'start_date': '2020-01-01',
            'rate_limit_delay': 0.5
        }
        
        self.collected_data = []
        self.stats = {
            'total_posts': 0,
            'total_comments': 0,
            'code_switching_posts': 0,
            'code_switching_comments': 0,
            'usc_relevant_posts': 0,
            'usc_relevant_comments': 0
        }
    
    def search_posts_with_code_switching(self):
        """Enhanced search for posts with code-switching"""
        all_posts = []
        
        # Search with different keyword combinations
        search_strategies = [
            # Direct USC searches
            ['university of san carlos', 'usc cebu', 'carolinian'],
            # Code-switching + USC
            ['usc kasi', 'usc pero', 'usc tapos', 'usc sige'],
            # Campus-specific searches
            ['usc talamban', 'usc downtown', 'usc main'],
            # Academic-related
            ['usc engineering', 'usc medicine', 'usc business', 'usc law'],
            # Student life
            ['usc student', 'usc freshie', 'usc graduate'],
            # Filipino context
            ['university cebu', 'college cebu', 'school cebu']
        ]
        
        for strategy in search_strategies:
            for subreddit_name in self.target_subreddits:
                posts = self._search_subreddit(subreddit_name, strategy)
                all_posts.extend(posts)
                
                if len(all_posts) >= self.collection_params['max_posts_total']:
                    break
                    
                time.sleep(self.collection_params['rate_limit_delay'])
            
            if len(all_posts) >= self.collection_params['max_posts_total']:
                break
        
        # Filter and deduplicate
        unique_posts = self._deduplicate_posts(all_posts)
        filtered_posts = self._filter_relevant_posts(unique_posts)
        
        return filtered_posts[:self.collection_params['max_posts_total']]
    
    def _search_subreddit(self, subreddit_name, keywords):
        """Search a specific subreddit with keywords"""
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for keyword in keywords:
                try:
                    # Search with different time filters
                    for time_filter in ['year', 'all']:
                        for submission in subreddit.search(
                            keyword, 
                            time_filter=time_filter, 
                            limit=50,
                            sort='relevance'
                        ):
                            post_date = datetime.fromtimestamp(submission.created_utc)
                            if post_date >= datetime.strptime(self.collection_params['start_date'], '%Y-%m-%d'):
                                
                                # Quick relevance check
                                combined_text = f"{submission.title} {submission.selftext}".lower()
                                code_analysis = self.code_detector.detect_code_switching(combined_text)
                                
                                post_data = {
                                    'post_id': submission.id,
                                    'subreddit': subreddit_name,
                                    'title': submission.title,
                                    'text': submission.selftext,
                                    'combined_text': combined_text,
                                    'score': submission.score,
                                    'upvote_ratio': submission.upvote_ratio,
                                    'num_comments': submission.num_comments,
                                    'created_utc': submission.created_utc,
                                    'created_date': post_date.strftime('%Y-%m-%d %H:%M:%S'),
                                    'author': str(submission.author),
                                    'url': submission.url,
                                    'permalink': f"https://reddit.com{submission.permalink}",
                                    'search_keyword': keyword,
                                    **code_analysis
                                }
                                posts.append(post_data)
                                
                        time.sleep(0.2)  # Rate limiting
                        
                except Exception as e:
                    print(f"Error searching '{keyword}' in r/{subreddit_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error accessing r/{subreddit_name}: {e}")
            
        return posts
    
    def _deduplicate_posts(self, posts):
        """Remove duplicate posts"""
        seen_ids = set()
        unique_posts = []
        
        for post in posts:
            if post['post_id'] not in seen_ids:
                seen_ids.add(post['post_id'])
                unique_posts.append(post)
                
        return unique_posts
    
    def _filter_relevant_posts(self, posts):
        """Filter posts for relevance to USC and code-switching"""
        relevant_posts = []
        
        for post in posts:
            # Must have USC relevance OR code-switching
            if post['usc_relevance_score'] > 0 or post['has_code_switching']:
                relevant_posts.append(post)
                
        return relevant_posts
    
    def collect_comments_parallel(self, posts):
        """Collect comments using parallel processing"""
        all_comments = []
        
        def get_comments_for_post(post):
            return self.get_post_comments(post['post_id'])
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_post = {
                executor.submit(get_comments_for_post, post): post 
                for post in posts[:50]  # Limit to prevent rate limiting
            }
            
            for future in as_completed(future_to_post):
                post = future_to_post[future]
                try:
                    comments = future.result()
                    all_comments.extend(comments)
                    print(f"Collected {len(comments)} comments from: {post['title'][:50]}...")
                except Exception as e:
                    print(f"Error collecting comments for {post['post_id']}: {e}")
                    
        return all_comments
    
    def get_post_comments(self, post_id):
        """Get comments for a specific post with code-switching analysis"""
        comments_data = []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:self.collection_params['comments_per_post']]:
                if hasattr(comment, 'body') and comment.body not in ['[deleted]', '[removed]']:
                    
                    # Analyze code-switching
                    code_analysis = self.code_detector.detect_code_switching(comment.body)
                    
                    comment_data = {
                        'comment_id': comment.id,
                        'post_id': post_id,
                        'text': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'created_date': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'author': str(comment.author),
                        'parent_id': comment.parent_id,
                        'is_submitter': comment.is_submitter,
                        **code_analysis
                    }
                    comments_data.append(comment_data)
                    
        except Exception as e:
            print(f"Error getting comments for post {post_id}: {e}")
            
        return comments_data
    
    def analyze_sentiment_enhanced(self, text):
        """Enhanced sentiment analysis considering Filipino context"""
        try:
            # Basic TextBlob analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Adjust for Filipino positive/negative expressions
            filipino_positive = ['ganda', 'ayos', 'okay', 'lami', 'nindot', 'sige', 'go']
            filipino_negative = ['pangit', 'dili', 'hindi', 'ayaw', 'baduy', 'boring']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in filipino_positive if word in text_lower)
            neg_count = sum(1 for word in filipino_negative if word in text_lower)
            
            # Adjust polarity based on Filipino words
            if pos_count > neg_count:
                polarity = min(1.0, polarity + 0.1 * pos_count)
            elif neg_count > pos_count:
                polarity = max(-1.0, polarity - 0.1 * neg_count)
            
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
                'sentiment_category': sentiment_category,
                'filipino_positive_words': pos_count,
                'filipino_negative_words': neg_count
            }
        except:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment_category': 'neutral',
                'filipino_positive_words': 0,
                'filipino_negative_words': 0
            }
    
    def run_complete_collection(self):
        """Run the complete data collection process"""
        print("Starting Enhanced USC Reddit Data Collection with Code-Switching Detection")
        print("=" * 80)
        
        # Step 1: Collect posts
        print("Step 1: Collecting posts...")
        posts = self.search_posts_with_code_switching()
        print(f"Found {len(posts)} relevant posts")
        
        # Step 2: Collect comments
        print("Step 2: Collecting comments...")
        comments = self.collect_comments_parallel(posts)
        print(f"Found {len(comments)} comments")
        
        # Step 3: Analyze sentiment
        print("Step 3: Analyzing sentiment...")
        all_data = []
        
        # Process posts
        for post in posts:
            sentiment = self.analyze_sentiment_enhanced(post['combined_text'])
            post.update({
                'content_type': 'post',
                **sentiment
            })
            all_data.append(post)
            
        # Process comments
        for comment in comments:
            sentiment = self.analyze_sentiment_enhanced(comment['text'])
            comment.update({
                'content_type': 'comment',
                **sentiment
            })
            all_data.append(comment)
        
        # Step 4: Save data
        print("Step 4: Saving data...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'usc_code_switching_data_{timestamp}.csv'
        
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        
        # Step 5: Generate enhanced statistics
        self.generate_enhanced_stats(df)
        
        print(f"\nData collection completed!")
        print(f"Total records: {len(df)}")
        print(f"Data saved to: {filename}")
        
        return df
    
    def generate_enhanced_stats(self, df):
        """Generate enhanced statistics"""
        print("\n" + "=" * 60)
        print("USC CODE-SWITCHING ANALYSIS RESULTS")
        print("=" * 60)
        
        # Basic counts
        posts_df = df[df['content_type'] == 'post']
        comments_df = df[df['content_type'] == 'comment']
        
        print(f"Total posts: {len(posts_df)}")
        print(f"Total comments: {len(comments_df)}")
        
        # Code-switching statistics
        cs_posts = posts_df[posts_df['has_code_switching'] == True]
        cs_comments = comments_df[comments_df['has_code_switching'] == True]
        
        print(f"\nCode-switching detection:")
        print(f"Posts with code-switching: {len(cs_posts)} ({len(cs_posts)/len(posts_df)*100:.1f}%)")
        print(f"Comments with code-switching: {len(cs_comments)} ({len(cs_comments)/len(comments_df)*100:.1f}%)")
        
        # Language mix analysis
        all_cs_data = df[df['has_code_switching'] == True]
        if len(all_cs_data) > 0:
            print(f"\nLanguage mix in code-switching content:")
            bisaya_count = len(all_cs_data[all_cs_data['bisaya_count'] > 0])
            tagalog_count = len(all_cs_data[all_cs_data['tagalog_count'] > 0])
            conyo_count = len(all_cs_data[all_cs_data['conyo_count'] > 0])
            
            print(f"Contains Bisaya: {bisaya_count}")
            print(f"Contains Tagalog: {tagalog_count}")
            print(f"Contains Conyo: {conyo_count}")
        
        # USC relevance
        usc_relevant = df[df['usc_relevance_score'] > 0]
        print(f"\nUSC relevance:")
        print(f"USC-relevant posts: {len(usc_relevant)} ({len(usc_relevant)/len(df)*100:.1f}%)")
        
        # Sentiment analysis
        print(f"\nSentiment distribution:")
        sentiment_counts = df['sentiment_category'].value_counts()
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment.capitalize()}: {count} ({count/len(df)*100:.1f}%)")
        
        # USC-specific sentiment
        usc_sentiment = usc_relevant['sentiment_category'].value_counts()
        print(f"\nUSC-specific sentiment:")
        for sentiment, count in usc_sentiment.items():
            print(f"{sentiment.capitalize()}: {count} ({count/len(usc_relevant)*100:.1f}%)")
        
        print("=" * 60)

def main():
    """Main execution function"""
    try:
        miner = EnhancedRedditMiner()
        df = miner.run_complete_collection()
        
        print("\n🎉 Data collection completed successfully!")
        print(f"📊 You now have {len(df)} records with code-switching analysis")
        print("📁 Check the CSV file for detailed results")
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        print("Please check your Reddit API credentials and internet connection")

if __name__ == "__main__":
    main()