import pandas as pd
import re
import logging
from typing import List, Dict, Any
import unicodedata
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_PATH = r"c:\Users\siobh\Usc_Thesis_1\data\raw\usc_reddit_data_20250704_205915.csv"
PROCESSED_PATH = r"c:\Users\siobh\Usc_Thesis_1\data\processed\usc_data_processed.csv"
STATS_PATH = r"c:\Users\siobh\Usc_Thesis_1\data\processed\cleaning_stats.json"

class CodeSwitchingCleaner:
    def __init__(self):
        # Common noise patterns in social media text
        self.noise_patterns = [
            r"http\S+",  # URLs
            r"www\.\S+",  # www links
            r"@\w+",  # mentions (optional - might be relevant for code-switching)
            r"#\w+",  # hashtags (optional - might be relevant)
            r"\b\d{10,}\b",  # long numbers (likely spam)
            r"([a-zA-Z])\1{3,}",  # repeated characters (hahaha -> haha)
            r"RT\s+",  # retweet indicators
        ]
        
        # Patterns for code-switching preservation
        self.preserve_patterns = [
            r"[^\x00-\x7F]+",  # Non-ASCII characters (multilingual content)
        ]
        
        # Statistics tracking
        self.stats = {
            'original_rows': 0,
            'rows_after_cleaning': 0,
            'rows_removed_empty': 0,
            'rows_removed_duplicates': 0,
            'rows_removed_noise': 0,
            'language_distribution': {},
            'avg_text_length_before': 0,
            'avg_text_length_after': 0
        }

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters while preserving multilingual content."""
        if pd.isnull(text):
            return ""
        
        # Normalize Unicode to NFC form (canonical decomposition + canonical composition)
        text = unicodedata.normalize('NFC', text)
        
        # Replace common Unicode variations
        unicode_replacements = {
            '\u2018': "'",  # left single quotation mark
            '\u2019': "'",  # right single quotation mark
            '\u201c': '"',  # left double quotation mark
            '\u201d': '"',  # right double quotation mark
            '\u2013': '-',  # en dash
            '\u2014': '--', # em dash
            '\u2026': '...',  # horizontal ellipsis
            '\u00a0': ' ',  # non-breaking space
        }
        
        for old, new in unicode_replacements.items():
            text = text.replace(old, new)
            
        return text

    def clean_text(self, text: str) -> str:
        """Clean text while preserving code-switching patterns."""
        if pd.isnull(text):
            return ""
        
        original_text = text
        
        # Normalize Unicode first
        text = self.normalize_unicode(text)
        
        # Remove noise patterns
        for pattern in self.noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Clean up excessive punctuation (but preserve some for emotion/emphasis)
        text = re.sub(r'([.!?]){3,}', r'\1\1\1', text)  # Max 3 repeated punctuation
        text = re.sub(r'([,;:]){2,}', r'\1', text)  # Max 1 for these
        
        # Handle repeated characters more carefully (preserve some emphasis)
        text = re.sub(r'([a-zA-Z])\1{4,}', r'\1\1\1', text)  # Max 3 repeated letters
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Preserve case for code-switching analysis (don't lowercase everything)
        return text

    def normalize_language_mix(self, lmix: Any) -> List[str]:
        """Normalize language mix information with better parsing."""
        if pd.isnull(lmix):
            return []
        
        # Convert to string if it's not already
        lmix_str = str(lmix)
        
        # Handle different formats
        if lmix_str.startswith('[') and lmix_str.endswith(']'):
            # List format: ['english', 'tagalog']
            try:
                # Try to parse as JSON first
                langs = json.loads(lmix_str.replace("'", '"'))
                return [lang.strip().lower() for lang in langs if lang.strip()]
            except:
                # Fallback to regex parsing
                lmix_str = re.sub(r"[\[\]']", "", lmix_str)
        
        # Comma-separated format
        langs = [lang.strip().lower() for lang in lmix_str.split(",") if lang.strip()]
        
        # Normalize common language names
        lang_mapping = {
            'eng': 'english',
            'fil': 'filipino',
            'tl': 'tagalog',
            'ceb': 'cebuano',
            'bis': 'bisaya',
            'hil': 'hiligaynon',
            'war': 'waray',
            'pam': 'kapampangan',
            'pan': 'pangasinan',
            'ilo': 'ilocano'
        }
        
        normalized_langs = []
        for lang in langs:
            normalized_langs.append(lang_mapping.get(lang, lang))
        
        return normalized_langs

    def is_high_quality_text(self, text: str, min_length: int = 10, max_length: int = 1000) -> bool:
        """Determine if text meets quality criteria for code-switching analysis."""
        if not text or len(text.strip()) < min_length:
            return False
        
        if len(text) > max_length:
            return False
        
        # Check if text has reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:  # At least 50% alphabetic characters
            return False
        
        # Check for excessive repetition (potential spam)
        words = text.split()
        if len(words) > 3:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
                return False
        
        return True

    def detect_code_switching(self, text: str, language_mix: List[str]) -> Dict[str, Any]:
        """Detect and analyze code-switching patterns."""
        if not text or not language_mix or len(language_mix) < 2:
            return {'has_code_switching': False, 'switch_points': 0, 'languages': language_mix}
        
        # Simple heuristic: look for script changes or language-specific patterns
        script_changes = 0
        prev_script = None
        
        for char in text:
            if char.isalpha():
                # Detect script type
                if ord(char) < 128:  # ASCII (likely English)
                    current_script = 'latin'
                else:
                    current_script = 'non_latin'
                
                if prev_script and prev_script != current_script:
                    script_changes += 1
                prev_script = current_script
        
        return {
            'has_code_switching': len(language_mix) > 1,
            'switch_points': script_changes,
            'languages': language_mix
        }

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the entire dataframe with comprehensive cleaning."""
        logger.info(f"Starting processing of {len(df)} rows")
        self.stats['original_rows'] = len(df)
        
        # Calculate original text length statistics
        if 'combined_text' in df.columns:
            original_lengths = df['combined_text'].dropna().str.len()
            self.stats['avg_text_length_before'] = original_lengths.mean()
        
        # Clean text columns
        text_columns = ['title', 'text', 'combined_text']
        for col in text_columns:
            if col in df.columns:
                logger.info(f"Cleaning column: {col}")
                df[col] = df[col].apply(self.clean_text)
        
        # Normalize language mix
        if 'language_mix' in df.columns:
            logger.info("Normalizing language_mix column")
            df['language_mix'] = df['language_mix'].apply(self.normalize_language_mix)
        
        # Add code-switching analysis
        if 'combined_text' in df.columns and 'language_mix' in df.columns:
            logger.info("Analyzing code-switching patterns")
            cs_analysis = df.apply(
                lambda row: self.detect_code_switching(row['combined_text'], row['language_mix']), 
                axis=1
            )
            df['has_code_switching'] = cs_analysis.apply(lambda x: x['has_code_switching'])
            df['switch_points'] = cs_analysis.apply(lambda x: x['switch_points'])
        
        # Filter out low-quality content
        initial_count = len(df)
        if 'combined_text' in df.columns:
            df = df[df['combined_text'].apply(self.is_high_quality_text)]
        
        self.stats['rows_removed_noise'] = initial_count - len(df)
        
        # Remove rows with empty essential content
        empty_before = len(df)
        df = df[df['combined_text'].str.strip() != ""]
        self.stats['rows_removed_empty'] = empty_before - len(df)
        
        # Remove duplicates
        dup_before = len(df)
        df = df.drop_duplicates(subset=['combined_text'])
        self.stats['rows_removed_duplicates'] = dup_before - len(df)
        
        # Calculate final statistics
        self.stats['rows_after_cleaning'] = len(df)
        if 'combined_text' in df.columns:
            final_lengths = df['combined_text'].str.len()
            self.stats['avg_text_length_after'] = final_lengths.mean()
        
        # Language distribution
        if 'language_mix' in df.columns:
            all_languages = []
            for lang_list in df['language_mix']:
                all_languages.extend(lang_list)
            
            from collections import Counter
            lang_counts = Counter(all_languages)
            self.stats['language_distribution'] = dict(lang_counts)
        
        logger.info(f"Processing complete. Final dataset: {len(df)} rows")
        return df

    def save_statistics(self):
        """Save cleaning statistics to file."""
        with open(STATS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to {STATS_PATH}")

    def print_summary(self):
        """Print a summary of the cleaning process."""
        print("\n" + "="*50)
        print("DATA CLEANING SUMMARY")
        print("="*50)
        print(f"Original rows: {self.stats['original_rows']}")
        print(f"Final rows: {self.stats['rows_after_cleaning']}")
        print(f"Removed (empty): {self.stats['rows_removed_empty']}")
        print(f"Removed (duplicates): {self.stats['rows_removed_duplicates']}")
        print(f"Removed (noise/quality): {self.stats['rows_removed_noise']}")
        print(f"Retention rate: {(self.stats['rows_after_cleaning']/self.stats['original_rows']*100):.1f}%")
        print(f"Avg text length before: {self.stats['avg_text_length_before']:.1f}")
        print(f"Avg text length after: {self.stats['avg_text_length_after']:.1f}")
        
        if self.stats['language_distribution']:
            print("\nLanguage Distribution:")
            for lang, count in sorted(self.stats['language_distribution'].items(), 
                                    key=lambda x: x[1], reverse=True):
                print(f"  {lang}: {count}")

def main():
    """Main processing function."""
    try:
        # Initialize cleaner
        cleaner = CodeSwitchingCleaner()
        
        # Load data
        logger.info(f"Loading data from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
        
        # Process data
        cleaned_df = cleaner.process_dataframe(df)
        
        # Save results
        logger.info(f"Saving cleaned data to {PROCESSED_PATH}")
        cleaned_df.to_csv(PROCESSED_PATH, index=False)
        
        # Save statistics and print summary
        cleaner.save_statistics()
        cleaner.print_summary()
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()