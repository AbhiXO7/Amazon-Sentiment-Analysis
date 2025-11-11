"""
Phase 5: Review Summarization
Extractive and Abstractive Summarization using Classical NLP
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict, Counter
import re

# Summarization
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ReviewSummarizer:
    """
    Multi-approach review summarization:
    1. Extractive (TextRank, TF-IDF)
    2. Topic-based summaries
    3. Pros/Cons extraction
    4. Key insights generation
    """
    
    def __init__(self):
        """Initialize summarizer"""
        print("🚀 Initializing Review Summarizer...")
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        # Sentiment keywords for pros/cons
        self.positive_keywords = [
            'good', 'great', 'excellent', 'amazing', 'best', 'perfect', 'love',
            'awesome', 'fantastic', 'wonderful', 'impressive', 'outstanding',
            'superb', 'brilliant', 'nice', 'beautiful', 'solid', 'smooth',
            'fast', 'quality', 'worth', 'recommend', 'satisfied', 'happy'
        ]
        
        self.negative_keywords = [
            'bad', 'poor', 'worst', 'terrible', 'horrible', 'awful', 'disappointing',
            'disappointed', 'waste', 'not', 'issue', 'problem', 'broken', 'defect',
            'useless', 'slow', 'cheap', 'overpriced', 'fail', 'failed', 'never',
            'dont', "don't", 'cant', "can't", 'wont', "won't", 'worst', 'hate'
        ]
        
        print("   ✅ Summarizer initialized\n")
    
    def textrank_summarize(self, text, num_sentences=3):
        """
        TextRank algorithm for extractive summarization
        Similar to PageRank but for sentences
        """
        if pd.isna(text) or not text:
            return ""
        
        # Split into sentences
        sentences = sent_tokenize(str(text))
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create sentence vectors using TF-IDF
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(sentences)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(sentence_vectors)
            
            # Build graph and apply PageRank
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Rank sentences
            ranked_sentences = sorted(
                [(scores[i], sentence) for i, sentence in enumerate(sentences)],
                reverse=True
            )
            
            # Get top sentences in original order
            top_sentences = sorted(
                [(sentences.index(sent), sent) for score, sent in ranked_sentences[:num_sentences]]
            )
            
            summary = ' '.join([sent for idx, sent in top_sentences])
            return summary
        
        except:
            # Fallback: return first N sentences
            return ' '.join(sentences[:num_sentences])
    
    def tfidf_summarize(self, text, num_sentences=3):
        """
        TF-IDF based extractive summarization
        Selects sentences with highest TF-IDF scores
        """
        if pd.isna(text) or not text:
            return ""
        
        sentences = sent_tokenize(str(text))
        
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate word frequencies
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum() and w not in self.stop_words]
        
        freq_dist = FreqDist(words)
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [w for w in sentence_words if w.isalnum()]
            
            score = sum(freq_dist.get(word, 0) for word in sentence_words)
            if len(sentence_words) > 0:
                sentence_scores[i] = score / len(sentence_words)
            else:
                sentence_scores[i] = 0
        
        # Get top sentences
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_indices = sorted(top_indices)  # Keep original order
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def extract_pros_cons(self, df):
        """
        Extract pros and cons from reviews
        """
        print("\n" + "=" * 80)
        print("EXTRACTING PROS & CONS")
        print("=" * 80)
        
        pros = []
        cons = []
        
        for idx, row in df.iterrows():
            text = str(row.get('cleaned_text', ''))
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                words = word_tokenize(sentence_lower)
                
                # Count positive/negative keywords
                pos_count = sum(1 for word in words if word in self.positive_keywords)
                neg_count = sum(1 for word in words if word in self.negative_keywords)
                
                # Classify sentence
                if pos_count > neg_count and pos_count > 0:
                    pros.append(sentence)
                elif neg_count > pos_count and neg_count > 0:
                    cons.append(sentence)
        
        print(f"\n✅ Extracted {len(pros)} positive statements")
        print(f"✅ Extracted {len(cons)} negative statements")
        
        return pros, cons
    
    def summarize_pros_cons(self, pros, cons, top_n=10):
        """
        Summarize pros and cons using frequency analysis
        """
        print("\n📊 Top Pros:")
        print("-" * 80)
        
        # Clean and count pros
        pros_clean = []
        for pro in pros:
            # Remove common prefixes
            pro_clean = re.sub(r'^(the |this |it |product )', '', pro.lower())
            pro_clean = pro_clean.strip()
            if len(pro_clean) > 10:  # Filter very short sentences
                pros_clean.append(pro_clean)
        
        # Get most common pros (using sentence similarity)
        pros_counter = Counter(pros_clean)
        top_pros = pros_counter.most_common(top_n)
        
        for i, (pro, count) in enumerate(top_pros, 1):
            print(f"{i}. [{count}x] {pro[:100]}")
        
        print("\n📊 Top Cons:")
        print("-" * 80)
        
        # Clean and count cons
        cons_clean = []
        for con in cons:
            con_clean = re.sub(r'^(the |this |it |product )', '', con.lower())
            con_clean = con_clean.strip()
            if len(con_clean) > 10:
                cons_clean.append(con_clean)
        
        cons_counter = Counter(cons_clean)
        top_cons = cons_counter.most_common(top_n)
        
        for i, (con, count) in enumerate(top_cons, 1):
            print(f"{i}. [{count}x] {con[:100]}")
        
        return top_pros, top_cons
    
    def summarize_by_topic(self, df):
        """
        Generate summaries for each topic
        """
        print("\n" + "=" * 80)
        print("TOPIC-BASED SUMMARIES")
        print("=" * 80)
        
        if 'dominant_topic' not in df.columns:
            print("⚠️ No topic information found. Skipping topic summaries.")
            return {}
        
        topic_summaries = {}
        
        for topic_id in sorted(df['dominant_topic'].unique()):
            print(f"\n🔹 Topic {topic_id + 1}:")
            print("-" * 80)
            
            topic_reviews = df[df['dominant_topic'] == topic_id]
            
            # Get top reviews for this topic
            top_reviews = topic_reviews.nlargest(5, 'topic_score')
            
            # Combine top reviews
            combined_text = ' '.join(top_reviews['cleaned_text'].astype(str))
            
            # Generate summary using TextRank
            summary = self.textrank_summarize(combined_text, num_sentences=5)
            
            topic_summaries[topic_id] = {
                'summary': summary,
                'num_reviews': len(topic_reviews),
                'avg_sentiment': topic_reviews.get('vader_compound', pd.Series([0])).mean(),
                'avg_rating': topic_reviews.get('rating_score', pd.Series([0])).mean()
            }
            
            print(f"Reviews in topic: {len(topic_reviews)}")
            if 'vader_compound' in topic_reviews.columns:
                print(f"Avg sentiment: {topic_summaries[topic_id]['avg_sentiment']:.3f}")
            if 'rating_score' in topic_reviews.columns:
                print(f"Avg rating: {topic_summaries[topic_id]['avg_rating']:.2f}/5.0")
            print(f"\nSummary:\n{summary}")
        
        return topic_summaries
    
    def generate_overall_summary(self, df):
        """
        Generate overall product summary
        """
        print("\n" + "=" * 80)
        print("OVERALL PRODUCT SUMMARY")
        print("=" * 80)
        
        summary = {
            'total_reviews': len(df),
            'avg_rating': df.get('rating_score', pd.Series([0])).mean(),
            'sentiment_distribution': {},
            'top_positive_aspects': [],
            'top_negative_aspects': [],
            'recommendation_score': 0
        }
        
        # Sentiment distribution
        if 'vader_sentiment' in df.columns:
            sentiment_counts = df['vader_sentiment'].value_counts()
            total = len(df)
            summary['sentiment_distribution'] = {
                'positive': sentiment_counts.get('positive', 0) / total * 100,
                'neutral': sentiment_counts.get('neutral', 0) / total * 100,
                'negative': sentiment_counts.get('negative', 0) / total * 100
            }
        
        # Calculate recommendation score (0-100)
        if 'rating_score' in df.columns:
            avg_rating = df['rating_score'].mean()
            rating_score = (avg_rating / 5.0) * 50  # 50% weight
        else:
            rating_score = 0
        
        if 'vader_compound' in df.columns:
            avg_sentiment = df['vader_compound'].mean()
            sentiment_score = ((avg_sentiment + 1) / 2) * 50  # 50% weight, normalized to 0-50
        else:
            sentiment_score = 0
        
        summary['recommendation_score'] = rating_score + sentiment_score
        
        # Extract key aspects using POS tags
        if 'pos_tags' in df.columns:
            # This would require parsing the pos_tags column
            # For now, we'll use a simplified approach
            pass
        
        # Display summary
        print(f"\n📊 Statistical Overview:")
        print(f"   Total Reviews Analyzed: {summary['total_reviews']}")
        print(f"   Average Rating: {summary['avg_rating']:.2f}/5.0")
        print(f"   Recommendation Score: {summary['recommendation_score']:.1f}/100")
        
        if summary['sentiment_distribution']:
            print(f"\n😊 Sentiment Breakdown:")
            print(f"   Positive: {summary['sentiment_distribution']['positive']:.1f}%")
            print(f"   Neutral:  {summary['sentiment_distribution']['neutral']:.1f}%")
            print(f"   Negative: {summary['sentiment_distribution']['negative']:.1f}%")
        
        # Generate textual summary
        positive_pct = summary['sentiment_distribution'].get('positive', 0)
        negative_pct = summary['sentiment_distribution'].get('negative', 0)
        
        if positive_pct > 60:
            sentiment_text = "overwhelmingly positive"
        elif positive_pct > 40:
            sentiment_text = "mostly positive"
        elif negative_pct > 40:
            sentiment_text = "mostly negative"
        else:
            sentiment_text = "mixed"
        
        recommendation = "Highly Recommended" if summary['recommendation_score'] > 75 else \
                        "Recommended" if summary['recommendation_score'] > 60 else \
                        "Consider with Caution" if summary['recommendation_score'] > 40 else \
                        "Not Recommended"
        
        print(f"\n💬 Summary:")
        print(f"   Based on {summary['total_reviews']} reviews, this product receives")
        print(f"   {sentiment_text} feedback with an average rating of {summary['avg_rating']:.2f}/5.0.")
        print(f"   Overall Assessment: {recommendation}")
        
        return summary
    
    def create_summary_report(self, df, pros, cons, topic_summaries, overall_summary, output_dir):
        """
        Create comprehensive summary report (text file)
        """
        print("\n📄 Creating summary report...")
        
        report_path = output_dir / "summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AMAZON PRODUCT REVIEW ANALYSIS - SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall Summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Reviews: {overall_summary['total_reviews']}\n")
            f.write(f"Average Rating: {overall_summary['avg_rating']:.2f}/5.0\n")
            f.write(f"Recommendation Score: {overall_summary['recommendation_score']:.1f}/100\n\n")
            
            if overall_summary['sentiment_distribution']:
                f.write("Sentiment Distribution:\n")
                f.write(f"  Positive: {overall_summary['sentiment_distribution']['positive']:.1f}%\n")
                f.write(f"  Neutral:  {overall_summary['sentiment_distribution']['neutral']:.1f}%\n")
                f.write(f"  Negative: {overall_summary['sentiment_distribution']['negative']:.1f}%\n\n")
            
            # Top Pros
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP PROS (What Customers Love)\n")
            f.write("=" * 80 + "\n")
            top_pros, top_cons = self.summarize_pros_cons(pros, cons, top_n=15)
            for i, (pro, count) in enumerate(top_pros, 1):
                f.write(f"{i}. [{count}x] {pro}\n")
            
            # Top Cons
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP CONS (What Customers Dislike)\n")
            f.write("=" * 80 + "\n")
            for i, (con, count) in enumerate(top_cons, 1):
                f.write(f"{i}. [{count}x] {con}\n")
            
            # Topic Summaries
            if topic_summaries:
                f.write("\n" + "=" * 80 + "\n")
                f.write("TOPIC-BASED SUMMARIES\n")
                f.write("=" * 80 + "\n")
                
                for topic_id, topic_data in topic_summaries.items():
                    f.write(f"\nTopic {topic_id + 1}:\n")
                    f.write(f"  Reviews: {topic_data['num_reviews']}\n")
                    f.write(f"  Avg Sentiment: {topic_data['avg_sentiment']:.3f}\n")
                    f.write(f"  Avg Rating: {topic_data['avg_rating']:.2f}/5.0\n")
                    f.write(f"  Summary: {topic_data['summary']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"   ✅ Report saved: {report_path}")
    
    def create_visualizations(self, df, pros, cons, overall_summary, output_dir):
        """
        Create summary visualizations
        """
        print("\n📊 Creating summary visualizations...")
        
        sns.set_style("whitegrid")
        
        # 1. Pros vs Cons comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        categories = ['Positive\nStatements', 'Negative\nStatements']
        counts = [len(pros), len(cons)]
        colors = ['green', 'red']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Positive vs Negative Statements in Reviews')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pros_vs_cons_count.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: pros_vs_cons_count.png")
        plt.close()
        
        # 2. Recommendation Score Gauge
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        score = overall_summary['recommendation_score']
        
        # Create gauge chart
        colors_gauge = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        bounds = [0, 20, 40, 60, 80, 100]
        
        for i in range(len(colors_gauge)):
            ax.barh(0, 20, left=bounds[i], height=0.3, color=colors_gauge[i], alpha=0.7)
        
        # Add score marker
        ax.plot([score, score], [-0.2, 0.2], 'k-', linewidth=3)
        ax.plot(score, 0, 'ko', markersize=15)
        ax.text(score, -0.5, f'{score:.1f}', ha='center', fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.7, 0.5)
        ax.set_xlabel('Recommendation Score')
        ax.set_title('Overall Product Recommendation Score', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add labels
        labels = ['Not\nRecommended', 'Poor', 'Fair', 'Good', 'Excellent']
        positions = [10, 30, 50, 70, 90]
        for label, pos in zip(labels, positions):
            ax.text(pos, 0.35, label, ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'recommendation_score.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: recommendation_score.png")
        plt.close()
        
        # 3. Key Metrics Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total reviews
        axes[0, 0].text(0.5, 0.5, f"{overall_summary['total_reviews']}", 
                       ha='center', va='center', fontsize=48, fontweight='bold', color='steelblue')
        axes[0, 0].text(0.5, 0.2, 'Total Reviews', 
                       ha='center', va='center', fontsize=14)
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')
        
        # Average rating
        avg_rating = overall_summary['avg_rating']
        axes[0, 1].text(0.5, 0.5, f"{avg_rating:.2f}★", 
                       ha='center', va='center', fontsize=48, fontweight='bold', color='gold')
        axes[0, 1].text(0.5, 0.2, 'Average Rating', 
                       ha='center', va='center', fontsize=14)
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axis('off')
        
        # Positive percentage
        if overall_summary['sentiment_distribution']:
            pos_pct = overall_summary['sentiment_distribution']['positive']
            axes[1, 0].text(0.5, 0.5, f"{pos_pct:.1f}%", 
                           ha='center', va='center', fontsize=48, fontweight='bold', color='green')
            axes[1, 0].text(0.5, 0.2, 'Positive Reviews', 
                           ha='center', va='center', fontsize=14)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        
        # Recommendation score
        axes[1, 1].text(0.5, 0.5, f"{score:.0f}/100", 
                       ha='center', va='center', fontsize=48, fontweight='bold', color='purple')
        axes[1, 1].text(0.5, 0.2, 'Recommendation Score', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.suptitle('Key Metrics Dashboard', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / 'key_metrics_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: key_metrics_dashboard.png")
        plt.close()


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("PHASE 5: REVIEW SUMMARIZATION")
    print("=" * 80 + "\n")
    
    # File paths
    INPUT_FILE = "data/processed/reviews_topics.csv"
    OUTPUT_FILE = "data/processed/reviews_summarized.csv"
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print(f"📂 Loading data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   ✅ Loaded {len(df)} reviews\n")
    except FileNotFoundError:
        print(f"   ❌ File not found: {INPUT_FILE}")
        print(f"   Trying alternative file...")
        try:
            INPUT_FILE = "data/processed/reviews_sentiment.csv"
            df = pd.read_csv(INPUT_FILE)
            print(f"   ✅ Loaded {len(df)} reviews from sentiment file\n")
        except:
            print(f"   ❌ No processed files found. Please run previous phases first!")
            return
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Initialize summarizer
    summarizer = ReviewSummarizer()
    
    # 1. Extract pros and cons
    pros, cons = summarizer.extract_pros_cons(df)
    summarizer.summarize_pros_cons(pros, cons, top_n=15)
    
    # 2. Topic-based summaries
    topic_summaries = summarizer.summarize_by_topic(df)
    
    # 3. Generate overall summary
    overall_summary = summarizer.generate_overall_summary(df)
    
    # 4. Create summary report
    summarizer.create_summary_report(df, pros, cons, topic_summaries, overall_summary, RESULTS_DIR)
    
    # 5. Create visualizations
    summarizer.create_visualizations(df, pros, cons, overall_summary, RESULTS_DIR)
    
    # 6. Generate individual review summaries (optional)
    print("\n📝 Generating individual review summaries...")
    df['review_summary_textrank'] = df['cleaned_text'].apply(
        lambda x: summarizer.textrank_summarize(x, num_sentences=2)
    )
    df['review_summary_tfidf'] = df['cleaned_text'].apply(
        lambda x: summarizer.tfidf_summarize(x, num_sentences=2)
    )
    print("   ✅ Generated summaries for all reviews")
    
    # Save processed data
    print(f"\n💾 Saving results to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved successfully!")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 5 COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved:")
    print(f"  - Data: {OUTPUT_FILE}")
    print(f"  - Report: {RESULTS_DIR}/summary_report.txt")
    print(f"  - Visualizations: {RESULTS_DIR}/")
    print(f"\nNext: Phase 6 - QA System (9_qa_system.py)\n")


if __name__ == "__main__":
    main()