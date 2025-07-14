import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Required installations:
# pip install transformers torch sentencepiece
# pip install textblob vaderSentiment
# pip install langdetect polyglot

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from langdetect import detect, DetectorFactory
    import torch
except ImportError as e:
    print(f"Missing required packages. Please install: {e}")
    print("Run: pip install transformers torch textblob vaderSentiment langdetect")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seed for reproducible language detection
DetectorFactory.seed = 0

# File paths
PROCESSED_PATH = r"c:\Users\siobh\Usc_Thesis_1\data\processed\usc_code_switching_data_processed.csv"
SENTIMENT_PATH = r"c:\Users\siobh\Usc_Thesis_1\data\processed\usc_code_switching_sentiment_analysis.csv"
SENTIMENT_STATS_PATH = r"c:\Users\siobh\Usc_Thesis_1\data\processed\sentiment_analysis_stats.json"

class CodeSwitchingSentimentAnalyzer:
    def __init__(self):
        self.stats = {
            'total_posts': 0,
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'avg_confidence_scores': {},
            'language_sentiment_breakdown': {},
            'code_switching_sentiment_patterns': {},
            'processing_time': 0
        }
        
        # Initialize sentiment analyzers
        self.analyzers = {}
        self._initialize_analyzers()
        
        # Filipino/Tagalog sentiment lexicons (basic)
        self.filipino_positive_words = {
            'maganda', 'ganda', 'astig', 'cool', 'nice', 'okay', 'ok', 'salamat', 'thank you',
            'masaya', 'saya', 'happy', 'love', 'mahal', 'winner', 'solid', 'goods', 'sulit',
            'worth it', 'perfect', 'amazing', 'wow', 'galing', 'best', 'favorite', 'pogi',
            'gandang', 'magaling', 'talented', 'blessing', 'blessed', 'grateful', 'thankful'
        }
        
        self.filipino_negative_words = {
            'pangit', 'bad', 'hate', 'ayaw', 'galit', 'angry', 'mad', 'disappointed', 'sad',
            'malungkot', 'problema', 'problem', 'hirap', 'mahirap', 'difficult', 'stressed',
            'pagod', 'tired', 'boring', 'nakakainit', 'nakakainis', 'annoying', 'walang kwenta',
            'worst', 'terrible', 'awful', 'failed', 'failure', 'disappointed', 'frustrating'
        }
        
        # Common Filipino intensifiers
        self.filipino_intensifiers = {
            'sobrang': 2.0, 'very': 1.5, 'really': 1.5, 'super': 2.0, 'grabe': 1.8,
            'talaga': 1.3, 'lagi': 1.2, 'always': 1.4, 'never': -1.5, 'hindi': -1.0,
            'di': -1.0, 'wala': -1.2, 'walang': -1.2
        }

    def _initialize_analyzers(self):
        """Initialize different sentiment analysis models."""
        try:
            # VADER (good for social media text)
            self.analyzers['vader'] = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
            
            # RoBERTa-based model (good for context)
            if torch.cuda.is_available():
                device = 0
                logger.info("Using GPU for sentiment analysis")
            else:
                device = -1
                logger.info("Using CPU for sentiment analysis")
            
            self.analyzers['roberta'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=device,
                return_all_scores=True
            )
            logger.info("RoBERTa sentiment analyzer initialized")
            
            # Multilingual model
            self.analyzers['multilingual'] = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=device,
                return_all_scores=True
            )
            logger.info("Multilingual sentiment analyzer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing analyzers: {e}")
            # Fallback to basic analyzers
            self.analyzers['vader'] = SentimentIntensityAnalyzer()

    def detect_language_segments(self, text: str) -> List[Dict[str, Any]]:
        """Detect language segments in code-switched text."""
        if not text or len(text.strip()) < 10:
            return [{'text': text, 'language': 'unknown', 'confidence': 0.0}]
        
        # Split text into sentences for better language detection
        sentences = re.split(r'[.!?]+', text)
        segments = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
                
            try:
                detected_lang = detect(sentence)
                # Map language codes to readable names
                lang_mapping = {
                    'en': 'english',
                    'tl': 'tagalog',
                    'es': 'spanish',
                    'ceb': 'cebuano'
                }
                
                language = lang_mapping.get(detected_lang, detected_lang)
                segments.append({
                    'text': sentence,
                    'language': language,
                    'confidence': 0.8  # Placeholder confidence
                })
            except:
                segments.append({
                    'text': sentence,
                    'language': 'unknown',
                    'confidence': 0.0
                })
        
        return segments if segments else [{'text': text, 'language': 'mixed', 'confidence': 0.5}]

    def analyze_filipino_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using Filipino-specific lexicon."""
        if not text:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_score = 0
        negative_score = 0
        total_sentiment_words = 0
        
        for i, word in enumerate(words):
            # Check for intensifiers
            intensifier = 1.0
            if i > 0 and words[i-1] in self.filipino_intensifiers:
                intensifier = self.filipino_intensifiers[words[i-1]]
            
            # Check sentiment words
            if word in self.filipino_positive_words:
                positive_score += intensifier
                total_sentiment_words += 1
            elif word in self.filipino_negative_words:
                negative_score += abs(intensifier)  # Make sure it's positive for negative sentiment
                total_sentiment_words += 1
        
        if total_sentiment_words == 0:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Normalize scores
        total_score = positive_score + negative_score
        if total_score == 0:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        pos_norm = positive_score / total_score
        neg_norm = negative_score / total_score
        compound = (positive_score - negative_score) / total_score
        
        return {
            'compound': compound,
            'positive': pos_norm,
            'negative': neg_norm,
            'neutral': max(0, 1 - pos_norm - neg_norm)
        }

    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        if not text:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        try:
            scores = self.analyzers['vader'].polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    def analyze_with_roberta(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using RoBERTa model."""
        if not text or 'roberta' not in self.analyzers:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        try:
            results = self.analyzers['roberta'](text[:512])  # Limit text length
            scores = {}
            for result in results[0]:
                label = result['label'].lower()
                if label == 'label_0':
                    scores['negative'] = result['score']
                elif label == 'label_1':
                    scores['neutral'] = result['score']
                elif label == 'label_2':
                    scores['positive'] = result['score']
            
            return {
                'positive': scores.get('positive', 0.0),
                'negative': scores.get('negative', 0.0),
                'neutral': scores.get('neutral', 1.0)
            }
        except Exception as e:
            logger.warning(f"RoBERTa analysis failed: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    def analyze_with_multilingual(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using multilingual model."""
        if not text or 'multilingual' not in self.analyzers:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        try:
            results = self.analyzers['multilingual'](text[:512])
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label or label in ['5 stars', '4 stars']:
                    scores['positive'] = max(scores['positive'], score)
                elif 'negative' in label or label in ['1 star', '2 stars']:
                    scores['negative'] = max(scores['negative'], score)
                else:
                    scores['neutral'] = max(scores['neutral'], score)
            
            return scores
        except Exception as e:
            logger.warning(f"Multilingual analysis failed: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    def ensemble_sentiment_analysis(self, text: str, language_mix: List[str] = None) -> Dict[str, Any]:
        """Combine multiple sentiment analysis methods for better accuracy."""
        if not text:
            return {
                'final_sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                'individual_scores': {}
            }
        
        # Get scores from different analyzers
        vader_scores = self.analyze_with_vader(text)
        roberta_scores = self.analyze_with_roberta(text)
        multilingual_scores = self.analyze_with_multilingual(text)
        filipino_scores = self.analyze_filipino_sentiment(text)
        
        individual_scores = {
            'vader': vader_scores,
            'roberta': roberta_scores,
            'multilingual': multilingual_scores,
            'filipino': filipino_scores
        }
        
        # Weight the scores based on language mix
        weights = {'vader': 0.25, 'roberta': 0.25, 'multilingual': 0.25, 'filipino': 0.25}
        
        if language_mix:
            # Increase Filipino weight if Filipino languages are present
            filipino_langs = {'tagalog', 'filipino', 'cebuano', 'bisaya', 'hiligaynon'}
            if any(lang in filipino_langs for lang in language_mix):
                weights['filipino'] = 0.4
                weights['multilingual'] = 0.3
                weights['vader'] = 0.2
                weights['roberta'] = 0.1
        
        # Calculate ensemble scores
        ensemble_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        for analyzer, weight in weights.items():
            scores = individual_scores[analyzer]
            ensemble_scores['positive'] += scores.get('positive', 0.0) * weight
            ensemble_scores['negative'] += scores.get('negative', 0.0) * weight
            ensemble_scores['neutral'] += scores.get('neutral', 0.0) * weight
        
        # Normalize ensemble scores
        total = sum(ensemble_scores.values())
        if total > 0:
            ensemble_scores = {k: v/total for k, v in ensemble_scores.items()}
        
        # Determine final sentiment
        final_sentiment = max(ensemble_scores, key=ensemble_scores.get)
        confidence = ensemble_scores[final_sentiment]
        
        return {
            'final_sentiment': final_sentiment,
            'confidence': confidence,
            'scores': ensemble_scores,
            'individual_scores': individual_scores
        }

    def analyze_code_switching_sentiment_patterns(self, text: str, language_mix: List[str], 
                                                 sentiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment patterns in code-switched content."""
        if not text or len(language_mix) < 2:
            return {
                'has_sentiment_switching': False,
                'sentiment_switches': 0,
                'dominant_language_sentiment': None,
                'mixed_language_effect': 'none'
            }
        
        # Detect language segments
        segments = self.detect_language_segments(text)
        
        # Analyze sentiment for each segment
        segment_sentiments = []
        for segment in segments:
            if len(segment['text'].strip()) < 5:
                continue
            
            seg_sentiment = self.ensemble_sentiment_analysis(segment['text'], [segment['language']])
            segment_sentiments.append({
                'text': segment['text'],
                'language': segment['language'],
                'sentiment': seg_sentiment['final_sentiment'],
                'confidence': seg_sentiment['confidence']
            })
        
        # Analyze sentiment switching
        sentiment_switches = 0
        prev_sentiment = None
        
        for seg in segment_sentiments:
            if prev_sentiment and prev_sentiment != seg['sentiment']:
                sentiment_switches += 1
            prev_sentiment = seg['sentiment']
        
        # Determine dominant language sentiment
        lang_sentiment_count = {}
        for seg in segment_sentiments:
            if seg['language'] not in lang_sentiment_count:
                lang_sentiment_count[seg['language']] = {'positive': 0, 'negative': 0, 'neutral': 0}
            lang_sentiment_count[seg['language']][seg['sentiment']] += 1
        
        # Determine mixed language effect
        mixed_effect = 'none'
        if len(segment_sentiments) > 1:
            sentiments = [seg['sentiment'] for seg in segment_sentiments]
            if len(set(sentiments)) > 1:
                mixed_effect = 'sentiment_contrast'
            else:
                mixed_effect = 'sentiment_reinforcement'
        
        return {
            'has_sentiment_switching': sentiment_switches > 0,
            'sentiment_switches': sentiment_switches,
            'segment_analysis': segment_sentiments,
            'language_sentiment_breakdown': lang_sentiment_count,
            'mixed_language_effect': mixed_effect
        }

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the entire dataframe for sentiment analysis."""
        start_time = datetime.now()
        
        logger.info(f"Starting sentiment analysis of {len(df)} posts")
        self.stats['total_posts'] = len(df)
        
        # Initialize result columns
        df['sentiment_label'] = ''
        df['sentiment_confidence'] = 0.0
        df['sentiment_scores_positive'] = 0.0
        df['sentiment_scores_negative'] = 0.0
        df['sentiment_scores_neutral'] = 0.0
        df['vader_compound'] = 0.0
        df['roberta_scores'] = ''
        df['multilingual_scores'] = ''
        df['filipino_scores'] = ''
        df['has_sentiment_switching'] = False
        df['sentiment_switches'] = 0
        df['mixed_language_effect'] = 'none'
        df['segment_analysis'] = ''
        
        # Process each row
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing row {idx}/{len(df)}")
            
            text = row.get('combined_text', '')
            language_mix = row.get('language_mix', [])
            
            # Main sentiment analysis
            sentiment_result = self.ensemble_sentiment_analysis(text, language_mix)
            
            # Code-switching sentiment analysis
            cs_analysis = self.analyze_code_switching_sentiment_patterns(text, language_mix, sentiment_result)
            
            # Update dataframe
            df.at[idx, 'sentiment_label'] = sentiment_result['final_sentiment']
            df.at[idx, 'sentiment_confidence'] = sentiment_result['confidence']
            df.at[idx, 'sentiment_scores_positive'] = sentiment_result['scores']['positive']
            df.at[idx, 'sentiment_scores_negative'] = sentiment_result['scores']['negative']
            df.at[idx, 'sentiment_scores_neutral'] = sentiment_result['scores']['neutral']
            df.at[idx, 'vader_compound'] = sentiment_result['individual_scores']['vader'].get('compound', 0.0)
            df.at[idx, 'roberta_scores'] = json.dumps(sentiment_result['individual_scores']['roberta'])
            df.at[idx, 'multilingual_scores'] = json.dumps(sentiment_result['individual_scores']['multilingual'])
            df.at[idx, 'filipino_scores'] = json.dumps(sentiment_result['individual_scores']['filipino'])
            df.at[idx, 'has_sentiment_switching'] = cs_analysis['has_sentiment_switching']
            df.at[idx, 'sentiment_switches'] = cs_analysis['sentiment_switches']
            df.at[idx, 'mixed_language_effect'] = cs_analysis['mixed_language_effect']
            df.at[idx, 'segment_analysis'] = json.dumps(cs_analysis['segment_analysis'])
            
            # Update statistics
            sentiment = sentiment_result['final_sentiment']
            self.stats['sentiment_distribution'][sentiment] += 1
            
            # Update language-specific sentiment stats
            for lang in language_mix:
                if lang not in self.stats['language_sentiment_breakdown']:
                    self.stats['language_sentiment_breakdown'][lang] = {'positive': 0, 'negative': 0, 'neutral': 0}
                self.stats['language_sentiment_breakdown'][lang][sentiment] += 1
        
        # Calculate processing time
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Calculate average confidence scores
        for sentiment in ['positive', 'negative', 'neutral']:
            mask = df['sentiment_label'] == sentiment
            if mask.sum() > 0:
                self.stats['avg_confidence_scores'][sentiment] = df[mask]['sentiment_confidence'].mean()
        
        logger.info(f"Sentiment analysis completed in {self.stats['processing_time']:.2f} seconds")
        return df

    def save_statistics(self):
        """Save sentiment analysis statistics."""
        with open(SENTIMENT_STATS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Sentiment analysis statistics saved to {SENTIMENT_STATS_PATH}")

    def print_summary(self):
        """Print summary of sentiment analysis results."""
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total posts analyzed: {self.stats['total_posts']}")
        print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        print(f"Average time per post: {self.stats['processing_time']/self.stats['total_posts']:.3f} seconds")
        
        print("\nSentiment Distribution:")
        total = sum(self.stats['sentiment_distribution'].values())
        for sentiment, count in self.stats['sentiment_distribution'].items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        print("\nAverage Confidence Scores:")
        for sentiment, confidence in self.stats['avg_confidence_scores'].items():
            print(f"  {sentiment.capitalize()}: {confidence:.3f}")
        
        print("\nLanguage-Specific Sentiment Breakdown:")
        for lang, sentiments in self.stats['language_sentiment_breakdown'].items():
            total_lang = sum(sentiments.values())
            if total_lang > 0:
                print(f"  {lang.capitalize()}:")
                for sentiment, count in sentiments.items():
                    percentage = (count / total_lang) * 100
                    print(f"    {sentiment}: {count} ({percentage:.1f}%)")

def main():
    """Main function to run sentiment analysis."""
    try:
        # Initialize analyzer
        analyzer = CodeSwitchingSentimentAnalyzer()
        
        # Load processed data
        logger.info(f"Loading processed data from {PROCESSED_PATH}")
        df = pd.read_csv(PROCESSED_PATH)
        
        # Convert language_mix from string to list if needed
        if 'language_mix' in df.columns:
            df['language_mix'] = df['language_mix'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
            )
        
        # Process sentiment analysis
        sentiment_df = analyzer.process_dataframe(df)
        
        # Save results
        logger.info(f"Saving sentiment analysis results to {SENTIMENT_PATH}")
        sentiment_df.to_csv(SENTIMENT_PATH, index=False)
        
        # Save statistics and print summary
        analyzer.save_statistics()
        analyzer.print_summary()
        
        logger.info("Sentiment analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()