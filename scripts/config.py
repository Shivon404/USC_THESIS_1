# Reddit API Configuration
REDDIT_CONFIG = {
    'client_id': 'TG2zWWhi5QJNS8kIT5HgRQ',
    'client_secret': 'Pk8S4MauTq0lQtg5IXfrnpIvRi_Rrw',
    'user_agent': 'USC_Thesis_Research_v1.0 by u/Dizzy-Language-6383'
}

# Enhanced USC Keywords for better search coverage
USC_KEYWORDS = [
    # Official names
    'university of san carlos',
    'usc',
    'usc cebu',
    'san carlos university',
    'san carlos cebu',
    
    # Campus-specific
    'usc talamban',
    'usc downtown',
    'usc main',
    'usc tc',
    'usc dc',
    'usc north campus',
    'usc south campus',
    
    # Student-related
    'carolinian',
    'carolinians',
    'usc student',
    'usc students',
    'usc freshie',
    'usc freshman',
    'usc graduate',
    'usc alumni',
    'usc warrior',
    'usc warriors',
    
    # Academic programs
    'usc engineering',
    'usc business',
    'usc medicine',
    'usc nursing',
    'usc law',
    'usc architecture',
    'usc education',
    'usc arts',
    'usc sciences',
    'usc pharmacy',
    'usc dentistry',
    'casaa',
    
    # Campus life
    'green and gold',
    'usc campus',
    'usc library',
    'usc gym',
    'usc cafeteria',
    'usc dorm',
    'usc dormitory',
    
    # Code-switching combinations
    'usc kasi',
    'usc pero',
    'usc tapos',
    'usc sige',
    'usc naman',
    'usc talaga',
    'usc grabe',
    'usc super',
    'usc ang',
    'usc yung',
    'usc mga',
    'usc ba',
    'usc eh',
    'usc lang',
    'usc din',
    'usc rin',
    'usc nga',
    'usc hay',
    'usc uy',
    'usc bay',
    'usc wa',
    'usc nya',
    'usc man',
    'usc gud',
    'usc bitaw',
    'usc mao',
    'usc unsa',
    'usc ngano',
    'usc murag',
    'usc kaayo',
    'usc jud',
    'usc gyud'
]

# Target Subreddits - expanded list for better coverage
TARGET_SUBREDDITS = [
    # Philippine subreddits
    'Philippines',
    'ph',
    'casualph',
    'askph',
    'phcareers',
    'phinvest',
    'phr4r',
    'phclassifieds',
    'phgamers',
    'phbooks',
    'phtech',
    'phtravel',
    'phfood',
    'phmusicscene',
    'phtelecom',
    'phcovid',
    'phbuildapc',
    'phcycling',
    'phrunning',
    'phfitness',
    
    # Regional subreddits
    'Cebu',
    'cebuano',
    'bisaya',
    'visayas',
    'mindanao',
    'luzon',
    'davao',
    'iloilo',
    'bacolod',
    'baguio',
    'cagayan',
    'dumaguete',
    'tacloban',
    
    # Education-focused subreddits
    'studentsph',
    'CollegeStudentsph',
    'phstudents',
    'phuniversity',
    'phcollege',
    'UniversityOfThePhilippines',
    'ateneo',
    'dlsu',
    'admu',
    'ust',
    'adamson',
    'feu',
    'nu',
    'college',
    'university',
    'GradSchool',
    'premed',
    'medschool',
    'lawschool',
    'engineering',
    'EngineeringStudents',
    'BusinessStudents',
    'nursing',
    'medicalschool'
]

# Collection Parameters
COLLECTION_PARAMS = {
    'posts_per_subreddit': 100,          # Posts to collect per subreddit
    'comments_per_post': 30,             # Comments to collect per post
    'max_posts_total': 1000,             # Maximum total posts to collect
    'max_comments_total': 5000,          # Maximum total comments to collect
    'start_date': '2020-01-01',          # Start date for collection (pandemic onwards)
    'end_date': '2024-12-31',            # End date for collection
    'rate_limit_delay': 0.3,             # Delay between requests (seconds)
    'batch_size': 50,                    # Number of posts to process in each batch
    'max_retries': 3,                    # Maximum retries for failed requests
    'timeout': 30,                       # Request timeout in seconds
    'save_interval': 100,                # Save progress every N records
    'min_score_threshold': -5,           # Minimum score for posts/comments
    'max_text_length': 10000,            # Maximum text length to process
    'language_detection_threshold': 2     # Minimum words needed for language detection
}

# Code-switching Detection Configuration
CODE_SWITCHING_CONFIG = {
    'min_filipino_words': 2,             # Minimum Filipino words to detect code-switching
    'min_mixed_patterns': 1,             # Minimum mixed patterns to detect code-switching
    'confidence_threshold': 0.3,         # Confidence threshold for language detection
    'context_window': 5,                 # Words to consider around each Filipino word
    'weight_bisaya': 1.0,                # Weight for Bisaya words in scoring
    'weight_tagalog': 1.0,               # Weight for Tagalog words in scoring
    'weight_conyo': 0.8,                 # Weight for Conyo words in scoring
    'weight_mixed_patterns': 1.5,        # Weight for mixed patterns in scoring
    'boost_usc_relevance': 1.2           # Boost factor for USC-relevant content
}

# Enhanced Sentiment Analysis Configuration
SENTIMENT_CONFIG = {
    'polarity_threshold_positive': 0.1,   # Threshold for positive sentiment
    'polarity_threshold_negative': -0.1,  # Threshold for negative sentiment
    'subjectivity_threshold': 0.5,        # Threshold for subjective content
    'filipino_sentiment_boost': 0.1,      # Boost factor for Filipino sentiment words
    'context_sensitive': True,            # Enable context-sensitive sentiment analysis
    'normalize_scores': True,             # Normalize sentiment scores
    'use_emoji_sentiment': True,          # Include emoji sentiment analysis
    'custom_sentiment_words': {
        'positive': [
            # English positive words
            'good', 'great', 'excellent', 'amazing', 'awesome', 'perfect', 'wonderful',
            'fantastic', 'brilliant', 'outstanding', 'superb', 'marvelous', 'incredible',
            'love', 'like', 'enjoy', 'happy', 'glad', 'pleased', 'satisfied', 'proud',
            'impressed', 'recommend', 'approve', 'support', 'praise', 'appreciate',
            'thank', 'grateful', 'blessed', 'lucky', 'success', 'achievement', 'victory',
            'win', 'triumph', 'accomplish', 'graduate', 'pass', 'ace', 'excel',
            'smart', 'intelligent', 'talented', 'skilled', 'helpful', 'useful',
            'effective', 'successful', 'beautiful', 'nice', 'cool', 'fun', 'exciting',
            
            # Filipino/Bisaya positive words
            'ganda', 'maganda', 'gandang', 'nindot', 'gwapa', 'guwapo', 'lami',
            'lamian', 'sarap', 'masarap', 'ayos', 'okay', 'oks', 'sige', 'go',
            'tama', 'sakto', 'husto', 'dako', 'dagko', 'daghang', 'bongga',
            'galing', 'husay', 'linaw', 'klaro', 'maayo', 'maayong', 'nais',
            'gusto', 'trip', 'bet', 'pabor', 'suportahan', 'salamat', 'thanks',
            'salamat kaayo', 'thank you', 'gracias', 'appreciated', 'pasalamat',
            'proud', 'praud', 'malipay', 'malipayon', 'happy', 'lipay',
            'successful', 'success', 'graduate', 'nakagraduate', 'nakatapos',
            'passed', 'nakapasa', 'nakalusot', 'scholar', 'dean lister',
            'magna cum laude', 'summa cum laude', 'cum laude', 'honor',
            'award', 'recognition', 'achievement', 'accomplish', 'nakamit',
            'mabuti', 'mabuting', 'maayong', 'sakto', 'tumpak', 'perfect',
            'perpekto', 'kompleto', 'kumpleto', 'buo', 'tibuok',
            
            # Conyo positive expressions
            'super ganda', 'super ayos', 'grabe ang ganda', 'sobrang ganda',
            'ang galing', 'grabe ka', 'super cool', 'ang sarap', 'super sarap',
            'grabe naman', 'sobrang ayos', 'ang ganda naman', 'super nice',
            'grabe ang sarap', 'sobrang galing', 'super proud', 'ang proud',
            'super happy', 'sobrang saya', 'kilig', 'nakakilig', 'cute',
            'ang cute', 'super cute', 'loveit', 'love it', 'love na love',
            'bet ko', 'trip ko', 'gusto ko', 'favorite ko', 'paborito ko',
            
            # Academic positive terms
            'dean list', 'dean lister', 'honor roll', 'scholarship', 'scholar',
            'academic excellence', 'high grades', 'mataas na grades', 'uno',
            'singko', 'passed', 'nakapasa', 'graduate', 'graduation',
            'diploma', 'degree', 'course', 'major', 'program', 'curriculum',
            'professor', 'teacher', 'instructor', 'mentor', 'guide',
            'learn', 'natuto', 'natutunan', 'knowledge', 'kaalaman',
            'education', 'edukasyon', 'pag-aaral', 'study', 'aral',
            'research', 'pananaliksik', 'thesis', 'dissertation',
            'presentation', 'project', 'assignment', 'homework',
            'exam', 'quiz', 'test', 'recitation', 'laboratory', 'lab',
            'library', 'aklatan', 'books', 'libro', 'reference',
            'resource', 'material', 'notes', 'reviewer', 'study guide'
        ],
        'negative': [
            # English negative words
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'angry', 'mad', 'frustrated', 'annoyed', 'disappointed', 'sad',
            'depressed', 'upset', 'worried', 'stressed', 'difficult', 'hard',
            'impossible', 'fail', 'failed', 'failure', 'lose', 'lost',
            'problem', 'issue', 'trouble', 'wrong', 'mistake', 'error',
            'boring', 'dull', 'stupid', 'dumb', 'ugly', 'disgusting',
            'nasty', 'gross', 'sick', 'tired', 'exhausted', 'weak',
            'poor', 'cheap', 'expensive', 'waste', 'useless', 'worthless',
            'broken', 'damaged', 'ruined', 'destroyed', 'cancelled',
            'denied', 'rejected', 'refused', 'ignored', 'abandoned',
            
            # Filipino/Bisaya negative words
            'pangit', 'pangita', 'dili', 'hindi', 'ayaw', 'wala', 'wa',
            'lisod', 'hirap', 'mahirap', 'delikado', 'dangerous', 'peligroso',
            'masama', 'dautang', 'dautan', 'grabeng', 'grabe', 'sobrang',
            'baduy', 'jologs', 'kadiri', 'yuck', 'eww', 'kadiring',
            'boring', 'kapoy', 'kapoya', 'pagod', 'pagoda', 'tired',
            'stress', 'stressed', 'pressure', 'praning', 'takot', 'hadlok',
            'nahadlok', 'natakot', 'scared', 'afraid', 'worried', 'nabalaka',
            'problema', 'problem', 'issue', 'suliran', 'gulo', 'kagulong',
            'sayop', 'mali', 'wrong', 'mistake', 'error', 'kamalian',
            'bagsak', 'bulag', 'napaksa', 'dropout', 'failed', 'failure',
            'wala klase', 'walang klase', 'cancelled', 'suspend', 'suspended',
            'absent', 'late', 'tardy', 'kulang', 'incomplete', 'di kompleto',
            'di kumpleto', 'kulang pa', 'shortage', 'kakulangan',
            
            # Conyo negative expressions
            'super pangit', 'grabe ang pangit', 'sobrang pangit', 'ang pangit',
            'super baduy', 'grabe ang baduy', 'sobrang baduy', 'ang baduy',
            'super boring', 'grabe ang boring', 'sobrang boring', 'ang boring',
            'super hirap', 'grabe ang hirap', 'sobrang hirap', 'ang hirap',
            'super stress', 'grabe ang stress', 'sobrang stress', 'ang stress',
            'ayaw ko', 'hindi ko gusto', 'di ko trip', 'di ko bet',
            'kadiri naman', 'yuck naman', 'eww naman', 'ang kadiri',
            'nakakairita', 'nakakainis', 'annoying', 'irritating',
            'nakakastress', 'nakakapagod', 'nakakasawa', 'nakakaumay',
            
            # Academic negative terms
            'failed', 'bagsak', 'bulag', 'singko', 'incomplete', 'inc',
            'dropped', 'withdrawal', 'kicked out', 'expelled', 'suspended',
            'probation', 'academic probation', 'low grades', 'mababang grades',
            'failing grade', 'bagsak na grade', 'remedial', 'summer class',
            'overload', 'underload', 'delayed', 'irregular', 'shiftee',
            'transferee', 'dropout', 'stop out', 'leave of absence', 'loa',
            'financial problem', 'tuition', 'bayad', 'utang', 'debt',
            'scholarship revoked', 'nawala ang scholarship', 'no scholarship',
            'walang scholarship', 'competition', 'kompetensya', 'rivalry',
            'bully', 'bullying', 'harassment', 'discrimination', 'unfair',
            'bias', 'favored', 'pabor', 'teacher pet', 'sipsip'
        ],
        'neutral': [
            # Common neutral words
            'okay', 'ok', 'fine', 'normal', 'usual', 'regular', 'standard',
            'average', 'typical', 'common', 'ordinary', 'plain', 'simple',
            'basic', 'general', 'moderate', 'medium', 'middle', 'center',
            'balanced', 'equal', 'same', 'similar', 'alike', 'neutral',
            'neither', 'maybe', 'perhaps', 'probably', 'possibly', 'might',
            'could', 'would', 'should', 'seems', 'appears', 'looks like',
            'pwede', 'puwede', 'siguro', 'baka', 'marahil', 'possible',
            'posible', 'kaya', 'maybe', 'baka naman', 'siguro naman',
            'normal lang', 'okay lang', 'ayos lang', 'ganun lang',
            'ganon lang', 'usual lang', 'regular lang', 'standard lang'
        ]
    }
}

# Enhanced Search Strategy Configuration
SEARCH_STRATEGY_CONFIG = {
    'search_combinations': [
        # Direct USC searches
        {
            'keywords': ['university of san carlos', 'usc cebu', 'carolinian'],
            'priority': 'high',
            'expected_relevance': 0.9
        },
        # Code-switching + USC combinations
        {
            'keywords': ['usc kasi', 'usc pero', 'usc tapos', 'usc sige', 'usc naman'],
            'priority': 'high',
            'expected_relevance': 0.8
        },
        # Campus-specific searches
        {
            'keywords': ['usc talamban', 'usc downtown', 'usc main', 'usc tc', 'usc dc'],
            'priority': 'medium',
            'expected_relevance': 0.8
        },
        # Academic program searches
        {
            'keywords': ['usc engineering', 'usc medicine', 'usc business', 'usc law', 'usc nursing'],
            'priority': 'medium',
            'expected_relevance': 0.7
        },
        # Student life searches
        {
            'keywords': ['usc student', 'usc freshie', 'usc graduate', 'usc warriors'],
            'priority': 'medium',
            'expected_relevance': 0.7
        },
        # Sentiment-driven searches
        {
            'keywords': ['usc ganda', 'usc ayos', 'usc pangit', 'usc baduy', 'usc boring'],
            'priority': 'high',
            'expected_relevance': 0.8
        },
        # Regional context searches
        {
            'keywords': ['university cebu', 'college cebu', 'school cebu'],
            'priority': 'low',
            'expected_relevance': 0.4
        }
    ],
    'time_filters': ['year', 'all'],
    'sort_methods': ['relevance', 'new', 'hot'],
    'search_limit_per_keyword': 50,
    'dedupe_similarity_threshold': 0.8
}

# Data Output Configuration
OUTPUT_CONFIG = {
    'output_formats': ['csv', 'json', 'excel'],
    'include_raw_text': True,
    'include_metadata': True,
    'include_sentiment_scores': True,
    'include_code_switching_analysis': True,
    'anonymize_usernames': True,
    'timestamp_format': '%Y-%m-%d %H:%M:%S',
    'filename_prefix': 'usc_code_switching_sentiment',
    'backup_interval': 500,  # Backup every N records
    'compression': False,
    'encoding': 'utf-8'
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'generate_word_clouds': True,
    'generate_sentiment_plots': True,
    'generate_code_switching_stats': True,
    'generate_temporal_analysis': True,
    'generate_subreddit_comparison': True,
    'min_words_for_analysis': 3,
    'max_words_per_cloud': 100,
    'sentiment_trend_window': 30,  # days
    'code_switching_threshold': 0.3,
    'statistical_significance_level': 0.05
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'usc_reddit_miner.log',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_rotation': True,
    'max_log_size': '10MB',
    'backup_count': 5,
    'console_output': True
}

# Error Handling Configuration
ERROR_CONFIG = {
    'max_retries': 3,
    'retry_delay': 1.0,
    'exponential_backoff': True,
    'ignore_deleted_comments': True,
    'ignore_private_subreddits': True,
    'continue_on_api_error': True,
    'save_progress_on_error': True,
    'error_log_file': 'usc_reddit_errors.log'
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_workers': 5,
    'chunk_size': 100,
    'memory_limit': '1GB',
    'cache_enabled': True,
    'cache_size': 1000,
    'cache_ttl': 3600,  # seconds
    'parallel_processing': True,
    'async_enabled': False
}

# Validation Configuration
VALIDATION_CONFIG = {
    'min_text_length': 10,
    'max_text_length': 50000,
    'validate_urls': True,
    'check_spam': True,
    'language_detection': True,
    'content_quality_threshold': 0.5,
    'duplicate_detection': True,
    'profanity_filter': False  # Set to True if needed
}

# Export these configurations for use in the main script
__all__ = [
    'REDDIT_CONFIG',
    'USC_KEYWORDS',
    'TARGET_SUBREDDITS',
    'COLLECTION_PARAMS',
    'CODE_SWITCHING_CONFIG',
    'SENTIMENT_CONFIG',
    'SEARCH_STRATEGY_CONFIG',
    'OUTPUT_CONFIG',
    'ANALYSIS_CONFIG',
    'LOGGING_CONFIG',
    'ERROR_CONFIG',
    'PERFORMANCE_CONFIG',
    'VALIDATION_CONFIG'
]