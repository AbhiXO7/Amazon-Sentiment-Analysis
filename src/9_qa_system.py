"""
Phase 6: Question-Answering System for Reviews
Interactive QA system using TF-IDF similarity and keyword matching
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Similarity and Ranking
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re


class ReviewQASystem:
    """
    Question-Answering System for Product Reviews
    Answers user questions by finding relevant review excerpts
    """
    
    def __init__(self, df):
        """Initialize QA system with review data"""
        print("🚀 Initializing QA System...")
        
        self.df = df.copy()
        self.vectorizer = None
        self.tfidf_matrix = None
        
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            nltk.download('stopwords')
            nltk.download('wordnet')
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        
        # Feature-specific keywords for better matching
        self.feature_keywords = {
            'battery': ['battery', 'backup', 'charge', 'charging', 'power', 'drain', 'life'],
            'camera': ['camera', 'photo', 'picture', 'image', 'quality', 'lens', 'zoom'],
            'display': ['display', 'screen', 'brightness', 'touch', 'resolution', 'panel'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'smooth', 'processor', 'ram'],
            'quality': ['quality', 'build', 'design', 'material', 'sturdy', 'premium'],
            'price': ['price', 'value', 'money', 'worth', 'expensive', 'cheap', 'cost'],
            'delivery': ['delivery', 'shipping', 'package', 'arrived', 'received', 'packaging'],
            'service': ['service', 'customer', 'support', 'warranty', 'replacement']
        }
        
        # Question patterns
        self.question_patterns = {
            'quality': ['how is', 'quality', 'good or bad', 'worth'],
            'problem': ['problem', 'issue', 'defect', 'complaint', 'bad', 'wrong'],
            'positive': ['good', 'best', 'pros', 'advantage', 'like', 'love'],
            'negative': ['bad', 'worst', 'cons', 'disadvantage', 'dislike', 'hate'],
            'recommendation': ['should i buy', 'recommend', 'worth buying', 'good choice'],
            'comparison': ['better', 'compared', 'versus', 'vs', 'difference']
        }
        
        print("   ✅ QA System initialized")
        self._build_search_index()
    
    def _build_search_index(self):
        """Build TF-IDF index for semantic search"""
        print("\n📊 Building search index...")
        
        # Use cleaned text for better matching
        text_column = 'cleaned_text' if 'cleaned_text' in self.df.columns else 'review'
        documents = self.df[text_column].fillna('').astype(str).tolist()
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        print(f"   ✅ Index built: {self.tfidf_matrix.shape[0]} documents, {self.tfidf_matrix.shape[1]} features")
    
    def _preprocess_query(self, query):
        """Preprocess user query"""
        query_lower = query.lower().strip()
        
        # Remove question marks and common question words
        query_clean = query_lower.replace('?', '').strip()
        question_words = ['what', 'how', 'is', 'are', 'does', 'do', 'can', 'will', 
                         'should', 'about', 'the', 'a', 'an']
        
        tokens = word_tokenize(query_clean)
        tokens = [t for t in tokens if t not in question_words and t not in self.stop_words]
        
        return ' '.join(tokens), query_lower
    
    def _detect_question_type(self, query_lower):
        """Detect the type of question being asked"""
        for qtype, patterns in self.question_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return qtype
        return 'general'
    
    def _detect_feature(self, query_lower):
        """Detect which product feature the question is about"""
        detected_features = []
        
        for feature, keywords in self.feature_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_features.append(feature)
        
        return detected_features
    
    def _semantic_search(self, query, top_k=10):
        """Find most relevant reviews using TF-IDF similarity"""
        query_processed, _ = self._preprocess_query(query)
        
        # Transform query to TF-IDF vector
        query_vec = self.vectorizer.transform([query_processed])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top K indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def _filter_by_sentiment(self, indices, sentiment_filter):
        """Filter results by sentiment"""
        if 'vader_sentiment' not in self.df.columns:
            return indices
        
        filtered_indices = []
        for idx in indices:
            review_sentiment = self.df.iloc[idx]['vader_sentiment']
            if sentiment_filter == 'positive' and review_sentiment == 'positive':
                filtered_indices.append(idx)
            elif sentiment_filter == 'negative' and review_sentiment == 'negative':
                filtered_indices.append(idx)
            elif sentiment_filter == 'neutral' and review_sentiment == 'neutral':
                filtered_indices.append(idx)
        
        return filtered_indices if filtered_indices else indices
    
    def _extract_relevant_sentences(self, review_text, query, max_sentences=2):
        """Extract most relevant sentences from a review"""
        sentences = sent_tokenize(str(review_text))
        
        if len(sentences) <= max_sentences:
            return review_text
        
        query_processed, _ = self._preprocess_query(query)
        query_words = set(query_processed.lower().split())
        
        # Score each sentence by word overlap with query
        sentence_scores = []
        for sentence in sentences:
            sentence_words = set(word_tokenize(sentence.lower()))
            overlap = len(query_words & sentence_words)
            sentence_scores.append((overlap, sentence))
        
        # Sort by score and get top sentences
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s for _, s in sentence_scores[:max_sentences]]
        
        return ' '.join(top_sentences)
    
    def answer_question(self, query, top_k=5, show_scores=False):
        """
        Main function to answer user questions
        """
        print(f"\n❓ Question: {query}")
        print("-" * 80)
        
        # Preprocess query
        query_processed, query_lower = self._preprocess_query(query)
        
        # Detect question type and features
        question_type = self._detect_question_type(query_lower)
        features = self._detect_feature(query_lower)
        
        print(f"🔍 Detected: Type='{question_type}'", end="")
        if features:
            print(f", Features={features}")
        else:
            print()
        
        # Semantic search
        indices, scores = self._semantic_search(query, top_k=top_k*2)
        
        # Filter by question type
        if question_type == 'positive':
            indices = self._filter_by_sentiment(indices, 'positive')
        elif question_type == 'negative':
            indices = self._filter_by_sentiment(indices, 'negative')
        elif question_type == 'problem':
            indices = self._filter_by_sentiment(indices, 'negative')
        
        # Limit to top_k
        indices = indices[:top_k]
        scores = scores[:top_k]
        
        # Display answers
        print(f"\n💬 Top {len(indices)} Relevant Answers:")
        print("=" * 80)
        
        for i, (idx, score) in enumerate(zip(indices, scores), 1):
            review = self.df.iloc[idx]
            
            # Extract relevant text
            text_column = 'cleaned_text' if 'cleaned_text' in self.df.columns else 'review'
            review_text = review[text_column]
            
            # Extract most relevant sentences
            relevant_text = self._extract_relevant_sentences(review_text, query, max_sentences=2)
            
            print(f"\n[Answer {i}]")
            
            # Show metadata
            if 'rating_score' in self.df.columns:
                rating = review['rating_score']
                stars = '★' * int(rating) + '☆' * (5 - int(rating))
                print(f"Rating: {stars} ({rating:.1f}/5.0)")
            
            if 'vader_sentiment' in self.df.columns:
                sentiment = review['vader_sentiment']
                sentiment_emoji = '😊' if sentiment == 'positive' else '😐' if sentiment == 'neutral' else '😞'
                print(f"Sentiment: {sentiment_emoji} {sentiment.capitalize()}")
            
            if show_scores:
                print(f"Relevance: {score:.3f}")
            
            print(f"\n{relevant_text}")
            print("-" * 80)
        
        return indices
    
    def get_summary_answer(self, query):
        """
        Generate a concise summary answer by aggregating multiple reviews
        """
        print(f"\n❓ Question: {query}")
        print("-" * 80)
        
        # Get top relevant reviews
        indices, scores = self._semantic_search(query, top_k=10)
        
        # Detect question type
        _, query_lower = self._preprocess_query(query)
        question_type = self._detect_question_type(query_lower)
        features = self._detect_feature(query_lower)
        
        print(f"🔍 Analyzing {len(indices)} relevant reviews...")
        
        # Aggregate information
        relevant_reviews = self.df.iloc[indices]
        
        summary = {
            'total_reviews': len(indices),
            'avg_rating': None,
            'sentiment_breakdown': {},
            'key_points': []
        }
        
        if 'rating_score' in relevant_reviews.columns:
            summary['avg_rating'] = relevant_reviews['rating_score'].mean()
        
        if 'vader_sentiment' in relevant_reviews.columns:
            sentiment_counts = relevant_reviews['vader_sentiment'].value_counts()
            total = len(relevant_reviews)
            summary['sentiment_breakdown'] = {
                'positive': sentiment_counts.get('positive', 0),
                'neutral': sentiment_counts.get('neutral', 0),
                'negative': sentiment_counts.get('negative', 0)
            }
        
        # Generate summary text
        print(f"\n💡 Summary Answer:")
        print("=" * 80)
        
        if features:
            print(f"\nRegarding {', '.join(features).upper()}:")
        
        if summary['avg_rating']:
            print(f"\n⭐ Average Rating: {summary['avg_rating']:.2f}/5.0")
        
        if summary['sentiment_breakdown']:
            pos = summary['sentiment_breakdown']['positive']
            neg = summary['sentiment_breakdown']['negative']
            neu = summary['sentiment_breakdown']['neutral']
            
            print(f"\n😊 Sentiment: {pos} positive, {neu} neutral, {neg} negative")
            
            if pos > neg:
                print(f"\n✅ Most customers have POSITIVE feedback.")
            elif neg > pos:
                print(f"\n⚠️ Most customers have NEGATIVE feedback.")
            else:
                print(f"\n• Feedback is MIXED.")
        
        # Extract key phrases
        print(f"\n📝 Key Points from Reviews:")
        
        text_column = 'cleaned_text' if 'cleaned_text' in self.df.columns else 'review'
        
        # Get most common important words
        all_text = ' '.join(relevant_reviews[text_column].astype(str))
        words = word_tokenize(all_text.lower())
        words = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 3]
        
        word_freq = Counter(words).most_common(10)
        common_words = [word for word, _ in word_freq]
        
        print(f"   Frequently mentioned: {', '.join(common_words[:7])}")
        
        # Show a few example quotes
        print(f"\n💬 Example Reviews:")
        for i, idx in enumerate(indices[:3], 1):
            review_text = self.df.iloc[idx][text_column]
            excerpt = self._extract_relevant_sentences(review_text, query, max_sentences=1)
            print(f"   {i}. \"{excerpt[:100]}...\"")
        
        print("=" * 80)
        
        return summary
    
    def interactive_mode(self):
        """
        Interactive question-answering mode
        """
        print("\n" + "=" * 80)
        print("INTERACTIVE QA SYSTEM")
        print("=" * 80)
        print("\nAsk questions about the product based on customer reviews!")
        print("Type 'exit' or 'quit' to stop.\n")
        print("Example questions:")
        print("  - How is the battery life?")
        print("  - What do customers say about the camera?")
        print("  - Are there any problems with this product?")
        print("  - What are the pros and cons?")
        print("  - Should I buy this product?")
        print("=" * 80)
        
        while True:
            try:
                query = input("\n❓ Your Question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q', '']:
                    print("\n👋 Thanks for using the QA system!")
                    break
                
                # Determine if user wants summary or detailed answer
                if 'summary' in query.lower() or 'overall' in query.lower():
                    self.get_summary_answer(query)
                else:
                    self.answer_question(query, top_k=5)
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the QA system!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def get_statistics(self):
        """Display system statistics"""
        print("\n" + "=" * 80)
        print("QA SYSTEM STATISTICS")
        print("=" * 80)
        
        print(f"\n📊 Dataset Info:")
        print(f"   Total reviews: {len(self.df)}")
        
        if 'rating_score' in self.df.columns:
            print(f"   Average rating: {self.df['rating_score'].mean():.2f}/5.0")
        
        if 'vader_sentiment' in self.df.columns:
            sentiment_dist = self.df['vader_sentiment'].value_counts()
            print(f"\n😊 Sentiment Distribution:")
            for sentiment, count in sentiment_dist.items():
                pct = (count / len(self.df)) * 100
                print(f"   {sentiment.capitalize()}: {count} ({pct:.1f}%)")
        
        if 'dominant_topic' in self.df.columns:
            topic_dist = self.df['dominant_topic'].value_counts().sort_index()
            print(f"\n📋 Topics Available:")
            for topic_id, count in topic_dist.items():
                print(f"   Topic {topic_id + 1}: {count} reviews")
        
        print("\n🔍 Searchable Features:")
        for feature in self.feature_keywords.keys():
            print(f"   - {feature.capitalize()}")


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("PHASE 6: QUESTION-ANSWERING SYSTEM")
    print("=" * 80 + "\n")
    
    # File paths
    INPUT_FILE = "data/processed/reviews_topics.csv"
    
    # Load data
    print(f"📂 Loading data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   ✅ Loaded {len(df)} reviews\n")
    except FileNotFoundError:
        print(f"   ❌ File not found: {INPUT_FILE}")
        print(f"   Trying alternative files...")
        
        alternative_files = [
            "data/processed/reviews_sentiment.csv",
            "data/processed/reviews_pos_ner.csv",
            "data/processed/reviews_preprocessed.csv"
        ]
        
        for alt_file in alternative_files:
            try:
                df = pd.read_csv(alt_file)
                print(f"   ✅ Loaded {len(df)} reviews from {alt_file}\n")
                break
            except:
                continue
        else:
            print(f"   ❌ No processed files found. Please run previous phases first!")
            return
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Initialize QA system
    qa_system = ReviewQASystem(df)
    
    # Show statistics
    qa_system.get_statistics()
    
    # Demo mode: Answer predefined questions
    print("\n" + "=" * 80)
    print("DEMO MODE: Answering Common Questions")
    print("=" * 80)
    
    demo_questions = [
        "How is the battery life?",
        "What do customers say about the camera quality?",
        "Are there any problems with this product?",
        "What are the main advantages?",
        "Should I buy this product?"
    ]
    
    for question in demo_questions:
        qa_system.answer_question(question, top_k=3)
        input("\nPress ENTER to continue...")
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Switching to Interactive Mode...")
    print("=" * 80)
    
    qa_system.interactive_mode()
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ QA SYSTEM SESSION COMPLETE!")
    print("=" * 80)
    print(f"\nYou can run this script anytime to query the reviews.\n")


if __name__ == "__main__":
    main()