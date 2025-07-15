import pandas as pd
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
from pathlib import Path

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_and_preprocess_text(text):
    """Advanced text preprocessing for better word analysis"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags (common in social media data)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove very short words (less than 2 characters) and numbers
    text = re.sub(r'\b\w{1}\b', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    
    return text

def get_meaningful_tokens(text, stop_words, lemmatizer, min_length=2):
    """Extract meaningful tokens with proper filtering"""
    # Use regex tokenizer for better control
    tokenizer = RegexpTokenizer(r'\b[a-zA-Z]{2,}\b')
    tokens = tokenizer.tokenize(text)
    
    # Filter out stopwords and very short tokens
    meaningful_tokens = []
    for token in tokens:
        if (len(token) >= min_length and 
            token not in stop_words and 
            token.isalpha()):
            # Optional: lemmatize tokens for better grouping
            lemmatized = lemmatizer.lemmatize(token)
            meaningful_tokens.append(lemmatized)
    
    return meaningful_tokens

def filter_words(word_freq, min_freq=3):
    """Filter words based on frequency and meaningfulness"""
    filtered_words = []
    
    for word, freq in word_freq:
        # Skip if frequency is too low
        if freq < min_freq:
            continue
            
        # Skip if word is too short
        if len(word) <= 2:
            continue
            
        # Skip very common words that might not be meaningful for code-switching analysis
        very_common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
        if word in very_common_words:
            continue
            
        filtered_words.append((word, freq))
    
    return filtered_words

def analyze_words_by_document(df, text_column='combined_text'):
    """Analyze words with document-level statistics"""
    # Initialize components
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Add custom stopwords if needed (domain-specific)
    custom_stopwords = {'would', 'could', 'should', 'one', 'two', 'also', 'may', 'might', 'much', 'many', 'get', 'go', 'come', 'say', 'said', 'see', 'know', 'think', 'want', 'need', 'make', 'made', 'take', 'took', 'give', 'gave', 'find', 'found', 'work', 'put', 'use', 'used', 'way', 'time', 'year', 'day', 'people', 'person', 'thing', 'place', 'part', 'good', 'great', 'new', 'old', 'first', 'last', 'long', 'little', 'own', 'right', 'same', 'different', 'still', 'just', 'only', 'even', 'well', 'back', 'up', 'down', 'out', 'over', 'after', 'before', 'here', 'there', 'where', 'when', 'why', 'how', 'what', 'who', 'which', 'this', 'that', 'these', 'those', 'he', 'she', 'it', 'they', 'we', 'you', 'me', 'him', 'her', 'them', 'us', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    stop_words.update(custom_stopwords)
    
    # Process all documents
    all_tokens = []
    document_stats = []
    
    print("📊 Processing documents...")
    for idx, text in enumerate(df[text_column].dropna()):
        # Clean and preprocess
        cleaned_text = clean_and_preprocess_text(text)
        
        if not cleaned_text:
            continue
            
        # Get meaningful tokens
        tokens = get_meaningful_tokens(cleaned_text, stop_words, lemmatizer)
        
        if len(tokens) < 1:  # Skip documents with no tokens
            continue
            
        all_tokens.extend(tokens)
        document_stats.append({
            'doc_id': idx,
            'original_length': len(text),
            'cleaned_length': len(cleaned_text),
            'token_count': len(tokens)
        })
    
    print(f"✅ Processed {len(document_stats)} documents with {len(all_tokens)} total tokens")
    
    return all_tokens, document_stats

def generate_word_analysis(tokens, top_k=100, min_freq=3):
    """Generate word frequency analysis with filtering"""
    print(f"🔍 Generating word frequency analysis...")
    
    # Count word frequencies
    word_freq = Counter(tokens).most_common(top_k * 3)  # Get more initially for filtering
    
    # Filter words
    filtered_words = filter_words(word_freq, min_freq)
    
    # Take top k after filtering
    top_words = filtered_words[:top_k]
    
    print(f"✅ Found {len(top_words)} meaningful words")
    return top_words

def create_results_directory():
    """Create results directory if it doesn't exist"""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    return results_dir

def save_detailed_results(words, document_stats, results_dir):
    """Save detailed results with additional statistics"""
    
    # Convert words to dataframe with additional info
    total_word_count = sum(freq for _, freq in words)
    words_df = pd.DataFrame([
        {
            'word': word,
            'frequency': freq,
            'percentage': round(freq / total_word_count * 100, 2) if total_word_count > 0 else 0,
            'length': len(word)
        }
        for word, freq in words
    ])
    
    # Document statistics
    doc_stats_df = pd.DataFrame(document_stats)
    
    # Save files
    words_df.to_csv(results_dir / 'usc_common_words_improved.csv', index=False)
    doc_stats_df.to_csv(results_dir / 'document_processing_stats.csv', index=False)
    
    # Save summary statistics
    summary_stats = {
        'total_documents_processed': len(document_stats),
        'total_unique_words_found': len(words),
        'total_word_occurrences': total_word_count,
        'avg_tokens_per_doc': doc_stats_df['token_count'].mean() if not doc_stats_df.empty else 0,
        'most_common_word': words[0][0] if words else None,
        'most_common_word_frequency': words[0][1] if words else 0,
        'avg_word_length': words_df['length'].mean() if not words_df.empty else 0
    }
    
    with open(results_dir / 'analysis_summary.txt', 'w') as f:
        f.write("Word Frequency Analysis Summary\n")
        f.write("=" * 32 + "\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
        
        # Additional analysis: word length distribution
        if not words_df.empty:
            length_dist = words_df['length'].value_counts().sort_index()
            f.write(f"\nWord Length Distribution:\n")
            f.write("-" * 25 + "\n")
            for length, count in length_dist.items():
                f.write(f"{length} characters: {count} words\n")
    
    return words_df, doc_stats_df

def main():
    """Main analysis function"""
    # Load data
    try:
        df = pd.read_csv('data/processed/usc_data_processed.csv')
        print(f"📁 Loaded {len(df)} rows from dataset")
    except FileNotFoundError:
        print("❌ Error: Could not find 'data/processed/usc_data_processed.csv'")
        return
    
    # Check for required column
    if 'combined_text' not in df.columns:
        print("❌ Error: Missing 'combined_text' column")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Create results directory
    results_dir = create_results_directory()
    
    # Analyze words
    all_tokens, document_stats = analyze_words_by_document(df)
    
    if not all_tokens:
        print("❌ No tokens found after processing")
        return
    
    # Generate word frequency analysis with improved filtering
    words = generate_word_analysis(all_tokens, top_k=100, min_freq=3)
    
    # Save detailed results
    words_df, doc_stats_df = save_detailed_results(words, document_stats, results_dir)
    
    # Display top results
    print("\n🔝 Top 20 Most Common Words:")
    print("-" * 50)
    for i, (word, freq) in enumerate(words[:20], 1):
        percentage = round(freq / len(all_tokens) * 100, 2) if all_tokens else 0
        print(f"{i:2d}. {word:20} | {freq:4d} occurrences ({percentage:5.2f}%)")
    
    # Additional insights
    if words:
        print(f"\n📊 Analysis Insights:")
        print(f"   - Total unique words analyzed: {len(words)}")
        print(f"   - Most frequent word: '{words[0][0]}' ({words[0][1]} times)")
        print(f"   - Average word length: {words_df['length'].mean():.1f} characters")
        print(f"   - Words with 3-4 characters: {len(words_df[words_df['length'].isin([3, 4])])}")
        print(f"   - Words with 5+ characters: {len(words_df[words_df['length'] >= 5])}")
    
    print(f"\n✅ Results saved to '{results_dir}' folder:")
    print(f"   - usc_common_words_improved.csv")
    print(f"   - document_processing_stats.csv")
    print(f"   - analysis_summary.txt")

if __name__ == "__main__":
    main()