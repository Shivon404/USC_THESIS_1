import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_keyword_dictionary():
    """Create refined keyword dictionary for unigram categorization"""
    
    keywords = {
        'Faculty/Professors': {
            # Faculty titles and references
            'titles': ['prof', 'professor', 'instructor', 'teacher', 'faculty', 'sir', 'maam', 'ma\'am', 'doctor', 'dr', 'mrs', 'mr', 'ms'],
            'descriptors': ['strict', 'kind', 'helpful', 'funny', 'boring', 'smart', 'knowledgeable', 'experienced', 'expert', 'skilled'],
            'faculty_actions': ['lecture', 'lecturing', 'explain', 'explaining', 'demonstrate', 'demonstrating', 'present', 'presenting', 'teach', 'teaching'],
            'faculty_qualities': ['expertise', 'knowledge', 'experience', 'background', 'qualification', 'credential', 'reputation'],
            'faculty_interactions': ['consultation', 'appointment', 'availability', 'discussion', 'conversation', 'guidance', 'mentoring']
        },
        
        'Student Experience': {
            # Student activities and academic work
            'academic_activities': ['study', 'studying', 'learn', 'learning', 'homework', 'assignment', 'project', 'research', 'review', 'preparation'],
            'assessments': ['exam', 'test', 'quiz', 'midterm', 'final', 'grade', 'grading', 'score', 'mark', 'result', 'performance', 'evaluation'],
            'student_actions': ['read', 'reading', 'write', 'writing', 'solve', 'solving', 'answer', 'answering', 'submit', 'submitting'],
            'learning_process': ['understand', 'understanding', 'comprehend', 'grasp', 'memorize', 'remember', 'recall', 'practice', 'master'],
            'student_feelings': ['difficult', 'easy', 'hard', 'challenging', 'tough', 'simple', 'complex', 'confusing', 'clear', 'interesting'],
            'student_outcomes': ['pass', 'fail', 'succeed', 'improve', 'progress', 'achieve', 'accomplish', 'complete', 'finish'],
            'study_methods': ['note', 'notes', 'highlight', 'summarize', 'outline', 'drill', 'repeat', 'concentrate', 'focus']
        },
        
        'School Environment': {
            # Physical spaces and facilities
            'physical_spaces': ['library', 'classroom', 'cafeteria', 'campus', 'building', 'room', 'hall', 'auditorium', 'lab', 'laboratory', 'gym', 'field'],
            'facilities': ['computer', 'wifi', 'internet', 'equipment', 'book', 'resource', 'facility', 'technology', 'projector', 'whiteboard'],
            'social_aspects': ['friend', 'friends', 'classmate', 'student', 'people', 'group', 'club', 'organization', 'community', 'peer'],
            'environment_quality': ['quiet', 'noisy', 'crowded', 'empty', 'comfortable', 'clean', 'dirty', 'new', 'old', 'modern', 'updated'],
            'campus_life': ['event', 'activity', 'gathering', 'celebration', 'festival', 'competition', 'sports', 'recreation', 'break', 'lunch'],
            'atmosphere': ['atmosphere', 'environment', 'culture', 'vibe', 'mood', 'feeling', 'spirit', 'energy', 'ambiance']
        },
        
        'Administrative Services': {
            # Bureaucratic processes and services
            'processes': ['enroll', 'enrollment', 'register', 'registration', 'application', 'apply', 'form', 'paperwork', 'document', 'procedure'],
            'admin_services': ['cashier', 'registrar', 'admission', 'admissions', 'scholarship', 'financial', 'aid', 'billing', 'payment', 'tuition'],
            'admin_staff': ['staff', 'admin', 'administrator', 'clerk', 'officer', 'personnel', 'employee', 'worker', 'secretary', 'assistant'],
            'systems': ['system', 'online', 'portal', 'website', 'platform', 'database', 'record', 'file', 'account', 'login'],
            'service_issues': ['problem', 'issue', 'error', 'mistake', 'delay', 'wait', 'queue', 'line', 'slow', 'fast', 'efficient', 'inefficient'],
            'requirements': ['requirement', 'certificate', 'transcript', 'credential', 'verification', 'approval', 'clearance', 'permit']
        }
    }
    
    return keywords

def create_flat_keyword_dict(keywords):
    """Flatten the keyword dictionary for easier matching"""
    flat_dict = {}
    for category, subcategories in keywords.items():
        all_keywords = []
        for subcat_keywords in subcategories.values():
            all_keywords.extend(subcat_keywords)
        flat_dict[category] = set(all_keywords)
    return flat_dict

def create_context_weights():
    """Create context-based weights for better categorization"""
    
    weights = {
        'Faculty/Professors': {
            'high_confidence': ['prof', 'professor', 'instructor', 'teacher', 'faculty', 'dr', 'doctor'],
            'medium_confidence': ['sir', 'maam', 'ma\'am', 'mrs', 'mr', 'ms', 'lecture', 'teaching'],
            'low_confidence': ['strict', 'kind', 'helpful', 'smart', 'experienced']
        },
        
        'Student Experience': {
            'high_confidence': ['study', 'studying', 'homework', 'exam', 'test', 'grade', 'assignment'],
            'medium_confidence': ['learn', 'learning', 'quiz', 'project', 'research', 'score'],
            'low_confidence': ['difficult', 'easy', 'hard', 'challenging', 'practice']
        },
        
        'School Environment': {
            'high_confidence': ['library', 'classroom', 'cafeteria', 'campus', 'building', 'laboratory'],
            'medium_confidence': ['room', 'hall', 'auditorium', 'gym', 'field', 'facility'],
            'low_confidence': ['quiet', 'noisy', 'crowded', 'comfortable', 'clean']
        },
        
        'Administrative Services': {
            'high_confidence': ['enroll', 'enrollment', 'register', 'registration', 'cashier', 'registrar'],
            'medium_confidence': ['admission', 'admissions', 'tuition', 'scholarship', 'application'],
            'low_confidence': ['staff', 'admin', 'system', 'online', 'portal']
        }
    }
    
    return weights

def enhanced_categorize_unigram(word, keyword_dict, context_weights):
    """Enhanced categorization for unigrams with confidence weighting"""
    word_lower = word.lower().strip()
    
    # Remove common prefixes/suffixes for better matching
    word_clean = re.sub(r'(ing|ed|er|est|ly|tion|ness)$', '', word_lower)
    
    category_scores = defaultdict(float)
    
    # Weight-based scoring
    confidence_weights = {
        'high_confidence': 3.0,
        'medium_confidence': 2.0,  
        'low_confidence': 1.0
    }
    
    # Check against weighted keywords
    for category, confidence_levels in context_weights.items():
        for confidence_level, keywords in confidence_levels.items():
            weight = confidence_weights[confidence_level]
            
            # Exact match
            if word_lower in keywords:
                category_scores[category] += weight
            
            # Partial match for cleaned word
            if word_clean in keywords:
                category_scores[category] += weight * 0.8
            
            # Substring match for longer words
            if len(word_lower) > 4:
                for keyword in keywords:
                    if keyword in word_lower or word_lower in keyword:
                        category_scores[category] += weight * 0.5
    
    # Special handling for ambiguous words
    ambiguous_words = {
        'office': ['Faculty/Professors', 'Administrative Services'],
        'meeting': ['Faculty/Professors', 'Administrative Services'],
        'help': ['Faculty/Professors', 'Student Experience'],
        'work': ['Student Experience', 'Administrative Services'],
        'time': ['Student Experience', 'School Environment'],
        'good': ['Faculty/Professors', 'Student Experience', 'School Environment'],
        'bad': ['Faculty/Professors', 'Student Experience', 'School Environment']
    }
    
    if word_lower in ambiguous_words:
        for category in ambiguous_words[word_lower]:
            category_scores[category] += 0.5
    
    # Return category with highest score
    if category_scores:
        max_score = max(category_scores.values())
        if max_score > 0:
            return max(category_scores, key=category_scores.get)
    
    return 'Uncategorized'

def analyze_unigram_categories(csv_file_path):
    """Main function to analyze and categorize unigrams"""
    
    # Load the unigram data
    try:
        df = pd.read_csv(csv_file_path)
        print(f"📊 Loaded {len(df)} unigrams from {csv_file_path}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{csv_file_path}'")
        return None, None
    
    # Check required columns - adapt for unigram format
    if 'word' in df.columns and 'frequency' in df.columns:
        # Standard unigram format
        word_col = 'word'
        freq_col = 'frequency'
    elif 'unigram' in df.columns and 'frequency' in df.columns:
        # Alternative unigram format
        word_col = 'unigram'
        freq_col = 'frequency'
    elif 'token' in df.columns and 'count' in df.columns:
        # Another common format
        word_col = 'token'
        freq_col = 'count'
    else:
        print(f"❌ Error: Could not find appropriate columns for unigram analysis")
        print(f"Available columns: {list(df.columns)}")
        print("Expected columns: 'word' and 'frequency' (or similar)")
        return None, None
    
    # Create keyword dictionary and context weights
    keywords = create_keyword_dictionary()
    flat_keywords = create_flat_keyword_dict(keywords)
    context_weights = create_context_weights()
    
    print("🔍 Categorizing unigrams with enhanced context analysis...")
    
    # Categorize each unigram
    df['category'] = df[word_col].apply(
        lambda word: enhanced_categorize_unigram(word, flat_keywords, context_weights)
    )
    
    # Create detailed analysis
    category_analysis = df.groupby('category').agg({
        freq_col: ['count', 'sum', 'mean'],
        word_col: 'first'
    }).round(2)
    
    # Flatten column names
    category_analysis.columns = ['word_count', 'total_frequency', 'avg_frequency', 'example_word']
    category_analysis = category_analysis.reset_index()
    
    # Calculate percentages
    total_words = len(df)
    total_frequency = df[freq_col].sum()
    
    category_analysis['percentage_by_count'] = (category_analysis['word_count'] / total_words * 100).round(2)
    category_analysis['percentage_by_frequency'] = (category_analysis['total_frequency'] / total_frequency * 100).round(2)
    
    # Sort by frequency
    category_analysis = category_analysis.sort_values('total_frequency', ascending=False)
    
    return df, category_analysis

def save_results(df, category_analysis, output_dir='results'):
    """Save categorized results to files"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Determine the word column name
    word_col = 'word' if 'word' in df.columns else ('unigram' if 'unigram' in df.columns else 'token')
    freq_col = 'frequency' if 'frequency' in df.columns else 'count'
    
    # Save categorized unigrams
    df_sorted = df.sort_values(['category', freq_col], ascending=[True, False])
    df_sorted.to_csv(output_path / 'unigrams_categorized.csv', index=False)
    
    # Save category summary
    category_analysis.to_csv(output_path / 'unigram_category_analysis_summary.csv', index=False)
    
    # Save detailed breakdown for each category
    for category in df['category'].unique():
        category_data = df[df['category'] == category].sort_values(freq_col, ascending=False)
        safe_category = category.replace('/', '_').replace(' ', '_')
        category_data.to_csv(output_path / f'unigram_category_{safe_category.lower()}.csv', index=False)
    
    print(f"✅ Results saved to '{output_dir}' folder")
    
    return df_sorted

def create_visualizations(category_analysis, output_dir='results'):
    """Create visualizations for the category analysis"""
    
    output_path = Path(output_dir)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Word count by category
    bars1 = ax1.bar(category_analysis['category'], category_analysis['word_count'])
    ax1.set_title('Number of Words by Category', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Number of Words')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. Total frequency by category
    bars2 = ax2.bar(category_analysis['category'], category_analysis['total_frequency'])
    ax2.set_title('Total Frequency by Category', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Total Frequency')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 3. Percentage by frequency (pie chart)
    colors = plt.cm.Set3(range(len(category_analysis)))
    wedges, texts, autotexts = ax3.pie(category_analysis['percentage_by_frequency'], 
                                      labels=category_analysis['category'], 
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      colors=colors)
    ax3.set_title('Distribution by Frequency Percentage', fontsize=14, fontweight='bold')
    
    # 4. Average frequency by category
    bars4 = ax4.bar(category_analysis['category'], category_analysis['avg_frequency'])
    ax4.set_title('Average Frequency by Category', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Average Frequency')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'unigram_category_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Visualization saved to '{output_path}/unigram_category_analysis_plots.png'")

def display_results(category_analysis, df):
    """Display formatted results"""
    
    # Determine column names
    word_col = 'word' if 'word' in df.columns else ('unigram' if 'unigram' in df.columns else 'token')
    freq_col = 'frequency' if 'frequency' in df.columns else 'count'
    
    print("\n" + "="*70)
    print("📈 UNIGRAM CATEGORY ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nTotal words analyzed: {len(df)}")
    print(f"Total frequency: {df[freq_col].sum()}")
    
    print("\n🏆 CATEGORY RANKINGS (by total frequency):")
    print("-"*85)
    print(f"{'Rank':<4} {'Category':<25} {'Words':<10} {'Frequency':<12} {'Freq %':<8} {'Avg Freq':<10}")
    print("-"*85)
    
    for idx, row in category_analysis.iterrows():
        print(f"{idx+1:<4} {row['category']:<25} {row['word_count']:<10} "
              f"{row['total_frequency']:<12} {row['percentage_by_frequency']:<8.1f}% "
              f"{row['avg_frequency']:<10.1f}")
    
    print("\n🔍 TOP WORDS BY CATEGORY:")
    print("-"*55)
    
    for category in category_analysis['category']:
        if category != 'Uncategorized':
            top_words = df[df['category'] == category].head(10)
            print(f"\n📚 {category}:")
            for _, row in top_words.iterrows():
                print(f"  • {row[word_col]:<20} ({row[freq_col]} occurrences)")
    
    # Show some uncategorized examples if they exist
    uncategorized = df[df['category'] == 'Uncategorized']
    if len(uncategorized) > 0:
        print(f"\n❓ Sample Uncategorized words:")
        for _, row in uncategorized.head(10).iterrows():
            print(f"  • {row[word_col]:<20} ({row[freq_col]} occurrences)")

def main():
    """Main execution function"""
    
    # File path - adjust as needed
    csv_file_path = 'results/usc_common_words_improved.csv'
    
  
    # Try to find the file
    file_found = False
    for path in [csv_file_path]:
        if Path(path).exists():
            csv_file_path = path
            file_found = True
            break
    
    if not file_found:
        print(f"❌ Could not find unigram CSV file. Tried:")
        for path in [csv_file_path]:
            print(f"   - {path}")
        print("\nPlease ensure the file exists or update the path in the code.")
        print("Expected format: CSV with columns 'word' and 'frequency' (or similar)")
        return
    
    # Analyze unigrams
    df, category_analysis = analyze_unigram_categories(csv_file_path)
    
    if df is None:
        return
    
    # Save results
    df_sorted = save_results(df, category_analysis)
    
    # Display results
    display_results(category_analysis, df)
    
    # Create visualizations
    try:
        create_visualizations(category_analysis)
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")
        print("Results are still saved to CSV files.")

if __name__ == "__main__":
    main()