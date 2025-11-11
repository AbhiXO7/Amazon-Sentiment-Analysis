"""
Phase 1: Data Acquisition & Preprocessing
Complete pipeline for cleaning and normalizing Amazon reviews
Pure NLTK version (no Spacy/PyTorch dependencies)
ENHANCED: Added rating extraction and date parsing
"""

import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except:
    print("⚠️ langdetect not available, language detection disabled")
    LANGDETECT_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except:
    print("⚠️ deep-translator not available, translation disabled")
    TRANSLATOR_AVAILABLE = False

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

class ReviewPreprocessor:
    """
    Complete preprocessing pipeline for Amazon reviews
    Handles: Cleaning, Language Detection, Translation, Tokenization, Lemmatization
    """
    
    def __init__(self):
        """Initialize preprocessor with required models"""
        print("🚀 Initializing Preprocessor...")
        
        # Load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
            print(f"   ✅ Loaded {len(self.stop_words)} stopwords")
        except:
            print("   ⚠️ Stopwords not found. Using basic list.")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                                   'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 
                                   'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
                                   'do', 'does', 'did', 'will', 'would', 'should', 'could',
                                   'may', 'might', 'must', 'can', 'this', 'that', 'these',
                                   'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                                   'what', 'which', 'who', 'when', 'where', 'why', 'how'])
        
        # Initialize lemmatizer
        try:
            self.lemmatizer = WordNetLemmatizer()
            print("   ✅ Lemmatizer initialized")
        except:
            print("   ⚠️ Lemmatizer not available")
            self.lemmatizer = None
        
        print()
        
        # Translation cache
        self.translation_cache = {}
    
    def extract_rating_score(self, rating_text):
        """Extract numeric rating from text like '5.0 out of 5 stars'"""
        if pd.isna(rating_text):
            return np.nan
        
        rating_str = str(rating_text).strip()
        
        # Pattern 1: "5.0 out of 5 stars"
        match = re.search(r'(\d+\.?\d*)\s*out of', rating_str)
        if match:
            return float(match.group(1))
        
        # Pattern 2: "5 stars" or "5.0 stars"
        match = re.search(r'^(\d+\.?\d*)\s*stars?', rating_str)
        if match:
            return float(match.group(1))
        
        # Pattern 3: Just a number
        match = re.search(r'^(\d+\.?\d*)$', rating_str)
        if match:
            return float(match.group(1))
        
        return np.nan
    
    def parse_date(self, date_text):
        """Parse date string to datetime object"""
        if pd.isna(date_text):
            return None
        
        date_str = str(date_text).strip()
        
        # Common date formats on Amazon India
        date_formats = [
            '%d %B %Y',      # "15 January 2024"
            '%d %b %Y',      # "15 Jan 2024"
            '%B %d, %Y',     # "January 15, 2024"
            '%b %d, %Y',     # "Jan 15, 2024"
            '%Y-%m-%d',      # "2024-01-15"
            '%d/%m/%Y',      # "15/01/2024"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        # If no format matched, return None
        return None
    
    def detect_language(self, text):
        """Detect language of text"""
        if not LANGDETECT_AVAILABLE:
            return 'en'  # Assume English if detection unavailable
        
        try:
            if pd.isna(text) or len(str(text).strip()) < 3:
                return 'unknown'
            return detect(str(text))
        except:
            return 'unknown'
    
    def translate_text(self, text, source_lang='hi', target_lang='en'):
        """Translate text from Hindi to English"""
        if not TRANSLATOR_AVAILABLE:
            return text  # Return original if translator unavailable
        
        try:
            # Check cache
            if text in self.translation_cache:
                return self.translation_cache[text]
            
            # Translate
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(str(text))
            
            # Cache result
            self.translation_cache[text] = translated
            return translated
        except Exception as e:
            return text  # Return original if translation fails
    
    def remove_html_tags(self, text):
        """Remove HTML tags"""
        if pd.isna(text):
            return ""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', str(text))
    
    def remove_urls(self, text):
        """Remove URLs"""
        if pd.isna(text):
            return ""
        return re.sub(r'http\S+|www.\S+', '', str(text))
    
    def remove_emojis(self, text):
        """Remove emojis"""
        if pd.isna(text):
            return ""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', str(text))
    
    def remove_special_chars(self, text):
        """Remove special characters except basic punctuation"""
        if pd.isna(text):
            return ""
        # Keep letters, numbers, and basic punctuation
        return re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', str(text))
    
    def normalize_whitespace(self, text):
        """Normalize multiple spaces to single space"""
        if pd.isna(text):
            return ""
        return ' '.join(str(text).split())
    
    def to_lowercase(self, text):
        """Convert to lowercase"""
        if pd.isna(text):
            return ""
        return str(text).lower()
    
    def tokenize_words(self, text):
        """Tokenize into words"""
        if pd.isna(text) or not text:
            return []
        try:
            return word_tokenize(str(text))
        except:
            # Fallback to simple split
            return str(text).split()
    
    def tokenize_sentences(self, text):
        """Tokenize into sentences"""
        if pd.isna(text) or not text:
            return []
        try:
            return sent_tokenize(str(text))
        except:
            # Fallback to split by period
            sentences = str(text).split('.')
            return [s.strip() for s in sentences if s.strip()]
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        if not tokens:
            return []
        return [word for word in tokens if word.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens (reduce to base form)"""
        if not tokens or not self.lemmatizer:
            return tokens
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def clean_text(self, text):
        """Complete cleaning pipeline"""
        if pd.isna(text):
            return ""
        
        # Step-by-step cleaning
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        text = self.to_lowercase(text)
        
        return text
    
    def preprocess_dataframe(self, df, text_column='review'):
        """
        Complete preprocessing pipeline for DataFrame
        """
        print("=" * 80)
        print("PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # Make a copy
        df_processed = df.copy()
        
        # 1. Initial info
        print(f"\n📊 Initial Data:")
        print(f"   Total reviews: {len(df_processed)}")
        print(f"   Columns: {list(df_processed.columns)}")
        
        # 2. Extract and parse rating
        if 'rating' in df_processed.columns:
            print(f"\n⭐ Processing Ratings:")
            df_processed['rating_score'] = df_processed['rating'].apply(self.extract_rating_score)
            valid_ratings = df_processed['rating_score'].notna().sum()
            print(f"   ✅ Extracted {valid_ratings} valid ratings")
            if valid_ratings > 0:
                print(f"   Average rating: {df_processed['rating_score'].mean():.2f}/5.0")
        
        # 3. Parse dates
        if 'date' in df_processed.columns:
            print(f"\n📅 Processing Dates:")
            df_processed['date_parsed'] = df_processed['date'].apply(self.parse_date)
            valid_dates = df_processed['date_parsed'].notna().sum()
            print(f"   ✅ Parsed {valid_dates} valid dates")
            if valid_dates > 0:
                earliest = df_processed['date_parsed'].min()
                latest = df_processed['date_parsed'].max()
                print(f"   Date range: {earliest.strftime('%d %b %Y')} to {latest.strftime('%d %b %Y')}")
        
        # 4. Handle missing values
        print(f"\n🔍 Handling Missing Values:")
        missing_before = df_processed[text_column].isna().sum()
        df_processed = df_processed.dropna(subset=[text_column])
        print(f"   Removed {missing_before} missing reviews")
        print(f"   Remaining: {len(df_processed)}")
        
        # 5. Remove duplicates
        print(f"\n🔍 Removing Duplicates:")
        duplicates = df_processed.duplicated(subset=[text_column]).sum()
        df_processed = df_processed.drop_duplicates(subset=[text_column])
        print(f"   Removed {duplicates} duplicate reviews")
        print(f"   Remaining: {len(df_processed)}")
        
        # 6. Language detection (if available)
        if LANGDETECT_AVAILABLE:
            print(f"\n🌐 Detecting Languages...")
            df_processed['language'] = df_processed[text_column].apply(self.detect_language)
            lang_dist = df_processed['language'].value_counts()
            print(f"   Language distribution:")
            for lang, count in lang_dist.head(5).items():
                print(f"      {lang}: {count}")
            
            # 7. Translation (if Hindi reviews exist)
            hindi_count = (df_processed['language'] == 'hi').sum()
            if hindi_count > 0 and TRANSLATOR_AVAILABLE:
                print(f"\n🔄 Translating {hindi_count} Hindi reviews to English...")
                hindi_mask = df_processed['language'] == 'hi'
                
                # Save original Hindi text
                df_processed['original_text'] = df_processed[text_column]
                
                for idx in df_processed[hindi_mask].index:
                    try:
                        original = df_processed.loc[idx, text_column]
                        translated = self.translate_text(original)
                        df_processed.loc[idx, text_column] = translated
                        print(f"   Translated review {idx}", end='\r')
                    except:
                        pass
                print(f"\n   ✅ Translation complete")
        else:
            df_processed['language'] = 'en'
        
        # 8. Text cleaning
        print(f"\n🧹 Cleaning Text:")
        df_processed['cleaned_text'] = df_processed[text_column].apply(self.clean_text)
        print(f"   ✅ Cleaned {len(df_processed)} reviews")
        
        # 9. Tokenization
        print(f"\n✂️ Tokenizing:")
        df_processed['tokens'] = df_processed['cleaned_text'].apply(self.tokenize_words)
        df_processed['token_count'] = df_processed['tokens'].apply(len)
        print(f"   ✅ Average tokens per review: {df_processed['token_count'].mean():.2f}")
        
        # 10. Sentence tokenization
        df_processed['sentences'] = df_processed['cleaned_text'].apply(self.tokenize_sentences)
        df_processed['sentence_count'] = df_processed['sentences'].apply(len)
        
        # 11. Remove stopwords
        print(f"\n🚫 Removing Stopwords:")
        df_processed['tokens_no_stopwords'] = df_processed['tokens'].apply(self.remove_stopwords)
        avg_before = df_processed['token_count'].mean()
        avg_after = df_processed['tokens_no_stopwords'].apply(len).mean()
        print(f"   Tokens before: {avg_before:.2f}")
        print(f"   Tokens after: {avg_after:.2f}")
        print(f"   Reduction: {((avg_before - avg_after) / avg_before * 100):.1f}%")
        
        # 12. Lemmatization
        print(f"\n🔤 Lemmatizing:")
        df_processed['lemmatized_tokens'] = df_processed['tokens_no_stopwords'].apply(self.lemmatize_tokens)
        df_processed['lemmatized_text'] = df_processed['lemmatized_tokens'].apply(lambda x: ' '.join(x))
        print(f"   ✅ Lemmatization complete")
        
        # 13. Final statistics
        print(f"\n📈 Final Statistics:")
        print(f"   Total processed reviews: {len(df_processed)}")
        print(f"   Average review length: {df_processed['lemmatized_text'].apply(len).mean():.2f} characters")
        print(f"   Average tokens: {df_processed['lemmatized_tokens'].apply(len).mean():.2f}")
        print(f"   Average sentences: {df_processed['sentence_count'].mean():.2f}")
        
        if 'rating_score' in df_processed.columns:
            print(f"\n⭐ Rating Statistics:")
            print(df_processed['rating_score'].value_counts().sort_index())
        
        return df_processed

def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("PHASE 1: DATA PREPROCESSING (ENHANCED)")
    print("=" * 80 + "\n")
    
    # File paths
    INPUT_FILE = "data/raw/amazon_reviews.csv"  # Your scraped data
    OUTPUT_FILE = "data/processed/reviews_preprocessed.csv"
    
    # Create output directory
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"📂 Loading data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   ✅ Loaded {len(df)} reviews\n")
    except FileNotFoundError:
        print(f"   ❌ File not found: {INPUT_FILE}")
        print(f"   Make sure your scraped CSV is in data/raw/ folder")
        return
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Initialize preprocessor
    preprocessor = ReviewPreprocessor()
    
    # Preprocess
    df_processed = preprocessor.preprocess_dataframe(df, text_column='review')
    
    # Save processed data
    print(f"\n💾 Saving processed data to: {OUTPUT_FILE}")
    df_processed.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved successfully!")
    
    # Display sample
    print("\n" + "=" * 80)
    print("SAMPLE PROCESSED REVIEWS (First 3)")
    print("=" * 80)
    
    for idx, row in df_processed.head(3).iterrows():
        print(f"\n[Review {idx + 1}]")
        if 'rating_score' in df_processed.columns:
            print(f"Rating:   {row['rating_score']}/5.0")
        if 'date_parsed' in df_processed.columns and pd.notna(row['date_parsed']):
            print(f"Date:     {row['date_parsed'].strftime('%d %B %Y')}")
        print(f"Original: {row['review'][:80]}...")
        print(f"Cleaned:  {row['cleaned_text'][:80]}...")
        print(f"Tokens:   {row['lemmatized_tokens'][:10]}")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 1 COMPLETE!")
    print("=" * 80)
    print(f"\nProcessed file saved at: {OUTPUT_FILE}")
    print(f"Total reviews processed: {len(df_processed)}")
    print("\nYou can now proceed to Phase 2: POS Tagging & NER\n")

if __name__ == "__main__":
    main()