# enhanced_usc_bow_vectorizer.py

import pandas as pd
import re
import nltk
import os
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from typing import Optional, Dict, Any, Tuple
import numpy as np

class TextPreprocessor:
    """
    Advanced text preprocessing with configurable options for stemming/lemmatization
    """
    
    def __init__(self, 
                 use_lemmatization: bool = True,
                 use_stemming: bool = False,
                 custom_stopwords: Optional[set] = None,
                 min_word_length: int = 2):
        """
        Initialize the text preprocessor
        
        Args:
            use_lemmatization: Whether to use lemmatization
            use_stemming: Whether to use stemming (ignored if lemmatization is True)
            custom_stopwords: Additional stopwords to remove
            min_word_length: Minimum word length to keep
        """
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming and not use_lemmatization
        self.min_word_length = min_word_length
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Initialize stemmer/lemmatizer
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        elif self.use_stemming:
            self.stemmer = PorterStemmer()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        nltk_downloads = ['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger']
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {item}: {e}")
    
    def preprocess(self, text: str) -> str:
        """
        Comprehensive text preprocessing pipeline
        
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
        
        # Remove mentions and hashtags (keep the text part)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation and numbers, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        
        # Filter by length and stopwords
        tokens = [word for word in tokens 
                 if len(word) >= self.min_word_length and word not in self.stop_words]
        
        # Apply stemming or lemmatization
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        elif self.use_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)

class BOWVectorizer:
    """
    Enhanced Bag of Words vectorizer with configurable parameters
    """
    
    def __init__(self,
                 max_features: Optional[int] = 10000,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 1),
                 preprocessor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BoW vectorizer
        
        Args:
            max_features: Maximum number of features to keep
            min_df: Minimum document frequency (ignore terms with lower frequency)
            max_df: Maximum document frequency (ignore terms with higher frequency)
            ngram_range: Range of n-grams to extract
            preprocessor_config: Configuration for text preprocessor
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        # Initialize preprocessor
        preprocessor_config = preprocessor_config or {}
        self.preprocessor = TextPreprocessor(**preprocessor_config)
        
        # Initialize vectorizer
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Words with at least 2 letters
        )
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
    def fit_transform(self, texts: pd.Series, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the vectorizer and transform texts
        
        Args:
            texts: Series of text data
            labels: Series of labels
            
        Returns:
            Tuple of (vectorized_texts, encoded_labels)
        """
        # Preprocess texts
        print("🔄 Preprocessing texts...")
        processed_texts = texts.apply(self.preprocessor.preprocess)
        
        # Remove empty texts
        non_empty_mask = processed_texts.str.len() > 0
        processed_texts = processed_texts[non_empty_mask]
        labels = labels[non_empty_mask]
        
        print(f"📊 Processing {len(processed_texts)} non-empty texts...")
        
        # Vectorize
        print("🔄 Vectorizing texts...")
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Encode labels
        print("🔄 Encoding labels...")
        y = self.label_encoder.fit_transform(labels)
        
        return X, y, processed_texts
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about extracted features"""
        feature_names = self.vectorizer.get_feature_names_out()
        return {
            'total_features': len(feature_names),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'sample_features': feature_names[:20].tolist(),
            'label_classes': self.label_encoder.classes_.tolist()
        }

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
                              feature_info, directories):
    """Save all vectorization results in organized structure"""
    
    # Save processed data
    joblib.dump(X, directories['processed'] / 'X_bow.pkl')
    joblib.dump(y, directories['processed'] / 'y_encoded.pkl')
    
    # Save models/vectorizers
    joblib.dump(vectorizer_obj.vectorizer, directories['models'] / 'bow_vectorizer.pkl')
    joblib.dump(vectorizer_obj.label_encoder, directories['models'] / 'label_encoder.pkl')
    
    # Save processed texts for reference
    processed_texts.to_csv(directories['processed'] / 'bow_processed_data.csv', index=False)
    
    # Save feature information report
    feature_report = pd.DataFrame({
        'feature_name': vectorizer_obj.vectorizer.get_feature_names_out(),
        'feature_count': np.array(X.sum(axis=0)).flatten()
    }).sort_values('feature_count', ascending=False)
    
    feature_report.to_csv(directories['reports'] / 'feature_analysis.csv', index=False)
    
    # Save summary report
    summary_report = f"""
USC Code-Switching Data - BoW Vectorization Report
================================================

Dataset Information:
- Total samples: {X.shape[0]}
- Total features: {X.shape[1]}
- Matrix sparsity: {(X.nnz / (X.shape[0] * X.shape[1]) * 100):.2f}%

Feature Information:
- Vocabulary size: {feature_info['vocabulary_size']}
- Sample features: {', '.join(feature_info['sample_features'])}

Label Information:
- Classes: {', '.join(feature_info['label_classes'])}
- Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}

Vectorization Parameters:
- Max features: {vectorizer_obj.max_features}
- Min document frequency: {vectorizer_obj.min_df}
- Max document frequency: {vectorizer_obj.max_df}
- N-gram range: {vectorizer_obj.ngram_range}

Files Generated:
- X_bow.pkl: Vectorized feature matrix
- y_encoded.pkl: Encoded labels
- bow_vectorizer.pkl: Fitted vectorizer model
- label_encoder.pkl: Label encoder model
- processed_texts.csv: Preprocessed texts
- feature_analysis.csv: Feature frequency analysis
"""
    
    with open(directories['reports'] / 'vectorization_summary.txt', 'w') as f:
        f.write(summary_report)

def main():
    """Main execution function"""
    print("🚀 Starting USC Code-Switching BoW Vectorization...")
    
    # Configuration
    config = {
        'input_file': 'data/processed/usc_code_switching_data_processed.csv',
        'text_column': 'combined_text',
        'label_column': 'sentiment_category',
        'vectorizer_params': {
            'max_features': 10000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 2)  # Include bigrams
        },
        'preprocessor_params': {
            'use_lemmatization': True,
            'use_stemming': False,
            'min_word_length': 2
        }
    }
    
    try:
        # Create output directories
        directories = create_output_directories()
        print("📁 Created output directories")
        
        # Load dataset
        print(f"📂 Loading dataset: {config['input_file']}")
        df = pd.read_csv(config['input_file'])
        print(f"📊 Dataset shape: {df.shape}")
        
        # Check for required columns
        if config['text_column'] not in df.columns:
            raise ValueError(f"Text column '{config['text_column']}' not found in dataset")
        if config['label_column'] not in df.columns:
            raise ValueError(f"Label column '{config['label_column']}' not found in dataset")
        
        # Remove rows with missing text or labels
        initial_count = len(df)
        df = df.dropna(subset=[config['text_column'], config['label_column']])
        print(f"📊 Removed {initial_count - len(df)} rows with missing data")
        
        # Initialize vectorizer
        vectorizer_obj = BOWVectorizer(
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
        print("\n" + "="*50)
        print("📊 VECTORIZATION RESULTS")
        print("="*50)
        print(f"✅ BoW Matrix Shape: {X.shape}")
        print(f"🧾 Total Features: {feature_info['total_features']}")
        print(f"📝 Sample Features: {', '.join(feature_info['sample_features'])}")
        print(f"🏷️  Label Classes: {', '.join(feature_info['label_classes'])}")
        print(f"💾 Matrix Sparsity: {(X.nnz / (X.shape[0] * X.shape[1]) * 100):.2f}%")
        
        # Save results
        save_vectorization_results(X, y, vectorizer_obj, processed_texts, 
                                  feature_info, directories)
        
        print("\n✅ Vectorization complete! Files saved in organized structure:")
        print(f"   📁 Processed data: {directories['processed']}")
        print(f"   📁 Models: {directories['models']}")
        print(f"   📁 Reports: {directories['reports']}")
        
    except Exception as e:
        print(f"❌ Error during vectorization: {e}")
        raise

if __name__ == "__main__":
    main()