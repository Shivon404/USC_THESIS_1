# enhanced_usc_tfidf_vectorizer.py

import pandas as pd
import re
import nltk
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import Optional, Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing with configurable options for sentiment analysis
    """
    
    def __init__(self, 
                 use_lemmatization: bool = True,
                 use_stemming: bool = False,
                 remove_stopwords: bool = True,
                 custom_stopwords: Optional[List[str]] = None,
                 min_word_length: int = 2,
                 max_word_length: int = 20,
                 preserve_sentiment_words: bool = True):
        """
        Initialize the advanced text preprocessor
        
        Args:
            use_lemmatization: Whether to use lemmatization
            use_stemming: Whether to use stemming (ignored if lemmatization is True)
            remove_stopwords: Whether to remove stopwords
            custom_stopwords: Additional stopwords to remove
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
            preserve_sentiment_words: Keep important sentiment words even if they're stopwords
        """
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming and not use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.preserve_sentiment_words = preserve_sentiment_words
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize stopwords with sentiment word preservation
        self.stop_words = self._initialize_stopwords(custom_stopwords)
        
        # Initialize stemmer/lemmatizer
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        elif self.use_stemming:
            self.stemmer = PorterStemmer()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        nltk_downloads = ['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger', 'omw-1.4']
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {item}: {e}")
    
    def _initialize_stopwords(self, custom_stopwords: Optional[List[str]] = None) -> set:
        """Initialize stopwords with sentiment preservation"""
        if not self.remove_stopwords:
            return set()
        
        stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        
        # Remove important sentiment words from stopwords if preserving
        if self.preserve_sentiment_words:
            sentiment_words = {
                'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere',
                'but', 'however', 'although', 'though', 'yet', 'except',
                'very', 'really', 'quite', 'rather', 'pretty', 'too', 'so',
                'more', 'most', 'much', 'many', 'few', 'little', 'less', 'least'
            }
            stop_words = stop_words - sentiment_words
        
        return stop_words
    
    def preprocess(self, text: str) -> str:
        """
        Comprehensive text preprocessing pipeline optimized for sentiment analysis
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Cleaned and processed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle Reddit-specific patterns
        text = re.sub(r'/u/\w+|/r/\w+', '', text)  # Remove Reddit user/subreddit mentions
        text = re.sub(r'r/\w+', '', text)  # Remove subreddit references
        
        # Handle contractions (important for sentiment)
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "let's": "let us", "that's": "that is",
            "what's": "what is", "here's": "here is", "there's": "there is"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove excessive punctuation but keep some for emphasis
        text = re.sub(r'[!]{2,}', ' very_emphasis ', text)  # Multiple exclamation marks
        text = re.sub(r'[?]{2,}', ' question_emphasis ', text)  # Multiple question marks
        text = re.sub(r'\.{3,}', ' ', text)  # Multiple dots
        
        # Remove remaining punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s_]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        
        # Filter by length and stopwords
        filtered_tokens = []
        for word in tokens:
            if (self.min_word_length <= len(word) <= self.max_word_length and
                word not in self.stop_words):
                filtered_tokens.append(word)
        
        # Apply stemming or lemmatization
        if self.use_lemmatization:
            filtered_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        elif self.use_stemming:
            filtered_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        
        return ' '.join(filtered_tokens)

class TFIDFVectorizer:
    """
    Enhanced TF-IDF vectorizer with configurable parameters for sentiment analysis
    """
    
    def __init__(self,
                 max_features: Optional[int] = 10000,
                 min_df: int = 2,
                 max_df: float = 0.9,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_idf: bool = True,
                 smooth_idf: bool = True,
                 sublinear_tf: bool = True,
                 norm: str = 'l2',
                 preprocessor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TF-IDF vectorizer
        
        Args:
            max_features: Maximum number of features to keep
            min_df: Minimum document frequency (ignore terms with lower frequency)
            max_df: Maximum document frequency (ignore terms with higher frequency)
            ngram_range: Range of n-grams to extract
            use_idf: Enable inverse-document-frequency reweighting
            smooth_idf: Smooth idf weights by adding one to document frequencies
            sublinear_tf: Apply sublinear tf scaling (replace tf with 1 + log(tf))
            norm: Normalization method ('l1', 'l2', or None)
            preprocessor_config: Configuration for text preprocessor
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        
        # Initialize preprocessor
        preprocessor_config = preprocessor_config or {}
        self.preprocessor = AdvancedTextPreprocessor(**preprocessor_config)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            norm=norm,
            token_pattern=r'\b[a-zA-Z][a-zA-Z_]+\b'  # Include words with underscores
        )
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
    def fit_transform(self, texts: pd.Series, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        Fit the vectorizer and transform texts
        
        Args:
            texts: Series of text data
            labels: Series of labels
            
        Returns:
            Tuple of (vectorized_texts, encoded_labels, processed_texts)
        """
        # Preprocess texts
        print("🔄 Preprocessing texts...")
        processed_texts = texts.apply(self.preprocessor.preprocess)
        
        # Remove empty texts
        non_empty_mask = processed_texts.str.len() > 0
        processed_texts = processed_texts[non_empty_mask]
        labels = labels[non_empty_mask]
        
        if len(processed_texts) == 0:
            raise ValueError("No valid texts found after preprocessing")
        
        print(f"📊 Processing {len(processed_texts)} non-empty texts...")
        
        # Vectorize
        print("🔄 Vectorizing texts with TF-IDF...")
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Encode labels
        print("🔄 Encoding labels...")
        y = self.label_encoder.fit_transform(labels)
        
        return X, y, processed_texts
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get comprehensive information about extracted features"""
        feature_names = self.vectorizer.get_feature_names_out()
        idf_scores = self.vectorizer.idf_
        
        # Get top features by IDF score
        feature_idf_pairs = list(zip(feature_names, idf_scores))
        feature_idf_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_features': len(feature_names),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'sample_features': feature_names[:20].tolist(),
            'top_idf_features': [f[0] for f in feature_idf_pairs[:10]],
            'label_classes': self.label_encoder.classes_.tolist(),
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df
        }

def load_data(file_path: str, text_column: str = 'combined_text', 
              label_column: str = 'sentiment_category') -> pd.DataFrame:
    """
    Load and validate the dataset
    
    Args:
        file_path: Path to the CSV file
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        Loaded and validated DataFrame
    """
    print(f"📂 Loading dataset from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"📊 Dataset loaded with shape: {df.shape}")
    
    # Check for required columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataset")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    # Check for missing data
    initial_count = len(df)
    missing_text = df[text_column].isna().sum()
    missing_labels = df[label_column].isna().sum()
    
    print(f"📊 Data quality check:")
    print(f"   - Missing text values: {missing_text}")
    print(f"   - Missing label values: {missing_labels}")
    
    # Remove rows with missing text or labels
    df = df.dropna(subset=[text_column, label_column])
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"⚠️  Removed {removed_count} rows with missing data")
    
    # Check for empty strings
    empty_text = (df[text_column].str.strip() == '').sum()
    if empty_text > 0:
        print(f"⚠️  Found {empty_text} empty text strings")
        df = df[df[text_column].str.strip() != '']
    
    print(f"✅ Final dataset shape: {df.shape}")
    return df

def create_output_directories(base_path: str = "data") -> Dict[str, Path]:
    """Create organized directory structure for outputs"""
    directories = {
        'processed': Path(base_path) / "processed",
        'models': Path(base_path) / "models",
        'reports': Path(base_path) / "reports"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories

def save_vectorization_results(X, y, vectorizer_obj, processed_texts, 
                              feature_info, directories, original_labels):
    """Save all vectorization results in organized structure"""
    
    print("💾 Saving vectorization results...")
    
    # Save processed data
    joblib.dump(X, directories['processed'] / 'X_tfidf.pkl')
    joblib.dump(y, directories['processed'] / 'y_encoded.pkl')
    
    # Save models/vectorizers
    joblib.dump(vectorizer_obj.vectorizer, directories['models'] / 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer_obj.label_encoder, directories['models'] / 'label_encoder.pkl')
    
    # Save processed texts and original labels for reference
    results_df = pd.DataFrame({
        'processed_text': processed_texts,
        'original_label': original_labels,
        'encoded_label': y
    })
    results_df.to_csv(directories['processed'] / 'tfidf_processed_data.csv', index=False)
    
    # Save feature information with TF-IDF scores
    feature_names = vectorizer_obj.vectorizer.get_feature_names_out()
    idf_scores = vectorizer_obj.vectorizer.idf_
    
    feature_analysis = pd.DataFrame({
        'feature_name': feature_names,
        'idf_score': idf_scores,
        'feature_frequency': np.array(X.sum(axis=0)).flatten()
    }).sort_values('idf_score', ascending=False)
    
    feature_analysis.to_csv(directories['reports'] / 'tfidf_feature_analysis.csv', index=False)
    
    # Save comprehensive summary report
    unique_labels, label_counts = np.unique(y, return_counts=True)
    label_distribution = dict(zip(vectorizer_obj.label_encoder.inverse_transform(unique_labels), label_counts))
    
    summary_report = f"""
USC Code-Switching Data - TF-IDF Vectorization Report
===================================================

Dataset Information:
- Total samples: {X.shape[0]}
- Total features: {X.shape[1]}
- Matrix sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%
- Matrix density: {(X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%

Feature Information:
- Vocabulary size: {feature_info['vocabulary_size']}
- N-gram range: {feature_info['ngram_range']}
- Min document frequency: {feature_info['min_df']}
- Max document frequency: {feature_info['max_df']}
- Sample features: {', '.join(feature_info['sample_features'])}
- Top IDF features: {', '.join(feature_info['top_idf_features'])}

Label Information:
- Classes: {', '.join(feature_info['label_classes'])}
- Class distribution: {label_distribution}

TF-IDF Parameters:
- Max features: {vectorizer_obj.max_features}
- Use IDF: {vectorizer_obj.use_idf}
- Smooth IDF: {vectorizer_obj.smooth_idf}
- Sublinear TF: {vectorizer_obj.sublinear_tf}
- Normalization: {vectorizer_obj.norm}

Preprocessing Configuration:
- Lemmatization: {vectorizer_obj.preprocessor.use_lemmatization}
- Stemming: {vectorizer_obj.preprocessor.use_stemming}
- Remove stopwords: {vectorizer_obj.preprocessor.remove_stopwords}
- Preserve sentiment words: {vectorizer_obj.preprocessor.preserve_sentiment_words}
- Min/Max word length: {vectorizer_obj.preprocessor.min_word_length}/{vectorizer_obj.preprocessor.max_word_length}

Files Generated:
- X_tfidf.pkl: TF-IDF feature matrix
- y_encoded.pkl: Encoded labels
- tfidf_vectorizer.pkl: Fitted TF-IDF vectorizer model
- label_encoder.pkl: Label encoder model
- tfidf_processed_data.csv: Processed texts with labels
- tfidf_feature_analysis.csv: Feature analysis with IDF scores
- tfidf_vectorization_summary.txt: This summary report
"""
    
    with open(directories['reports'] / 'tfidf_vectorization_summary.txt', 'w') as f:
        f.write(summary_report)
    
    print("✅ All results saved successfully!")

def main():
    """Main execution function"""
    print("🚀 Starting Enhanced USC Code-Switching TF-IDF Vectorization...")
    
    # Configuration
    config = {
        'input_file': 'data/processed/usc_code_switching_data_processed.csv',
        'text_column': 'combined_text',
        'label_column': 'sentiment_category',
        'vectorizer_params': {
            'max_features': 10000,
            'min_df': 2,
            'max_df': 0.9,
            'ngram_range': (1, 3),  # Include unigrams, bigrams, and trigrams
            'use_idf': True,
            'smooth_idf': True,
            'sublinear_tf': True,
            'norm': 'l2'
        },
        'preprocessor_params': {
            'use_lemmatization': True,
            'use_stemming': False,
            'remove_stopwords': True,
            'preserve_sentiment_words': True,
            'min_word_length': 2,
            'max_word_length': 20
        }
    }
    
    try:
        # Create output directories
        directories = create_output_directories()
        print("📁 Created output directories")
        
        # Load and validate dataset
        df = load_data(
            config['input_file'],
            config['text_column'],
            config['label_column']
        )
        
        # Initialize TF-IDF vectorizer
        vectorizer_obj = TFIDFVectorizer(
            **config['vectorizer_params'],
            preprocessor_config=config['preprocessor_params']
        )
        
        # Fit and transform
        X, y, processed_texts = vectorizer_obj.fit_transform(
            df[config['text_column']], 
            df[config['label_column']]
        )
        
        # Get feature information
        feature_info = vectorizer_obj.get_feature_info()
        
        # Display results
        print("\n" + "="*60)
        print("📊 TF-IDF VECTORIZATION RESULTS")
        print("="*60)
        print(f"✅ TF-IDF Matrix Shape: {X.shape}")
        print(f"🧾 Total Features: {feature_info['total_features']}")
        print(f"📝 Sample Features: {', '.join(feature_info['sample_features'])}")
        print(f"🔝 Top IDF Features: {', '.join(feature_info['top_idf_features'])}")
        print(f"🏷️  Label Classes: {', '.join(feature_info['label_classes'])}")
        print(f"💾 Matrix Sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
        print(f"📈 N-gram Range: {feature_info['ngram_range']}")
        
        # Save results
        save_vectorization_results(
            X, y, vectorizer_obj, processed_texts, feature_info, directories,
            df.loc[processed_texts.index, config['label_column']]
        )
        
        print("\n✅ TF-IDF vectorization complete! Files saved in organized structure:")
        print(f"   📁 Processed data: {directories['processed']}")
        print(f"   📁 Models: {directories['models']}")
        print(f"   📁 Reports: {directories['reports']}")
        
    except Exception as e:
        print(f"❌ Error during vectorization: {e}")
        raise

if __name__ == "__main__":
    main()