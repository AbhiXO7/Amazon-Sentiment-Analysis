"""
Phase 4: Topic Modeling using Latent Semantic Analysis (LSA)
Classical NLP approach (no Transformers)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLTK
import nltk
from collections import Counter


class TopicModeler:
    """
    Topic Modeling for Amazon Reviews
    Using LSA (Latent Semantic Analysis) and LDA (Latent Dirichlet Allocation)
    """
    
    def __init__(self, n_topics=5):
        """Initialize topic modeler"""
        print("🚀 Initializing Topic Modeler...")
        
        self.n_topics = n_topics
        self.lsa_model = None
        self.lda_model = None
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.tfidf_matrix = None
        self.count_matrix = None
        
        print(f"   ✅ Number of topics: {n_topics}")
        print()
    
    def prepare_tfidf_matrix(self, documents, max_features=1000):
        """Create TF-IDF matrix for LSA"""
        print("\n📊 Creating TF-IDF matrix for LSA...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),  # unigrams and bigrams
            stop_words='english'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        print(f"   ✅ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"   ✅ Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return self.tfidf_matrix
    
    def prepare_count_matrix(self, documents, max_features=1000):
        """Create Count matrix for LDA"""
        print("\n📊 Creating Count matrix for LDA...")
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.count_matrix = self.count_vectorizer.fit_transform(documents)
        
        print(f"   ✅ Count matrix shape: {self.count_matrix.shape}")
        print(f"   ✅ Vocabulary size: {len(self.count_vectorizer.vocabulary_)}")
        
        return self.count_matrix
    
    def train_lsa_model(self, n_components=None):
        """Train LSA model using TruncatedSVD"""
        if n_components is None:
            n_components = self.n_topics
        
        print("\n" + "=" * 80)
        print("TRAINING LSA MODEL (Latent Semantic Analysis)")
        print("=" * 80)
        
        # TruncatedSVD for dimensionality reduction
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        normalizer = Normalizer(copy=False)
        
        # Pipeline: TruncatedSVD + Normalization
        self.lsa_model = make_pipeline(svd, normalizer)
        
        # Fit and transform
        print(f"\n🎯 Training LSA with {n_components} topics...")
        doc_topic_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)
        
        # Explained variance
        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"   ✅ Explained variance: {explained_variance:.2%}")
        
        return doc_topic_matrix, svd
    
    def train_lda_model(self, n_components=None, max_iter=10):
        """Train LDA model"""
        if n_components is None:
            n_components = self.n_topics
        
        print("\n" + "=" * 80)
        print("TRAINING LDA MODEL (Latent Dirichlet Allocation)")
        print("=" * 80)
        
        print(f"\n🎯 Training LDA with {n_components} topics...")
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_components,
            max_iter=max_iter,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        
        doc_topic_matrix = self.lda_model.fit_transform(self.count_matrix)
        
        # Perplexity (lower is better)
        perplexity = self.lda_model.perplexity(self.count_matrix)
        print(f"   ✅ Model perplexity: {perplexity:.2f}")
        
        return doc_topic_matrix
    
    def get_top_words_per_topic(self, model_type='lsa', n_words=10):
        """Extract top words for each topic"""
        
        if model_type == 'lsa':
            if self.lsa_model is None:
                return None
            
            # Get SVD component
            svd = self.lsa_model.named_steps['truncatedsvd']
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            components = svd.components_
            
        elif model_type == 'lda':
            if self.lda_model is None:
                return None
            
            feature_names = self.count_vectorizer.get_feature_names_out()
            components = self.lda_model.components_
        
        else:
            return None
        
        # Extract top words
        topics = {}
        for topic_idx, topic in enumerate(components):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [topic[i] for i in top_indices]
            topics[topic_idx] = list(zip(top_words, top_scores))
        
        return topics
    
    def display_topics(self, topics, model_name='LSA', n_words=10):
        """Display topics with their top words"""
        print(f"\n📋 {model_name} Topics:")
        print("=" * 80)
        
        for topic_idx, words_scores in topics.items():
            print(f"\n🔹 Topic {topic_idx + 1}:")
            words = [word for word, score in words_scores[:n_words]]
            scores = [score for word, score in words_scores[:n_words]]
            
            print(f"   Top words: {', '.join(words)}")
            print(f"   Weights: {', '.join([f'{s:.3f}' for s in scores[:5]])}")
    
    def interpret_topics(self, topics, n_words=5):
        """Suggest interpretations for topics based on keywords"""
        print(f"\n🔍 Topic Interpretations:")
        print("=" * 80)
        
        # Common themes
        theme_keywords = {
            'Quality': ['good', 'quality', 'excellent', 'great', 'best', 'perfect', 'amazing'],
            'Price/Value': ['price', 'value', 'money', 'worth', 'cheap', 'expensive', 'cost'],
            'Battery': ['battery', 'charging', 'charge', 'power', 'backup', 'drain'],
            'Display/Screen': ['display', 'screen', 'brightness', 'touch', 'resolution'],
            'Camera': ['camera', 'photo', 'picture', 'image', 'quality', 'lens'],
            'Performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'smooth', 'processor'],
            'Build/Design': ['build', 'design', 'look', 'body', 'material', 'weight'],
            'Delivery/Service': ['delivery', 'package', 'received', 'shipping', 'arrived', 'service'],
            'Problems': ['problem', 'issue', 'bad', 'poor', 'worst', 'defect', 'broken']
        }
        
        for topic_idx, words_scores in topics.items():
            top_words = [word.lower() for word, score in words_scores[:n_words]]
            
            # Match with themes
            theme_scores = {}
            for theme, keywords in theme_keywords.items():
                score = sum(1 for word in top_words if word in keywords)
                if score > 0:
                    theme_scores[theme] = score
            
            print(f"\n🔹 Topic {topic_idx + 1}:")
            print(f"   Keywords: {', '.join(top_words)}")
            
            if theme_scores:
                best_theme = max(theme_scores, key=theme_scores.get)
                print(f"   💡 Likely about: {best_theme}")
            else:
                print(f"   💡 Likely about: General feedback")
    
    def get_document_topics(self, doc_topic_matrix, df):
        """Assign dominant topic to each document"""
        print(f"\n📄 Assigning topics to documents...")
        
        # Get dominant topic for each document
        dominant_topics = doc_topic_matrix.argmax(axis=1)
        topic_scores = doc_topic_matrix.max(axis=1)
        
        df['dominant_topic'] = dominant_topics
        df['topic_score'] = topic_scores
        
        print(f"   ✅ Assigned dominant topics")
        
        # Topic distribution
        topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
        print(f"\n📊 Document Distribution Across Topics:")
        for topic_idx, count in topic_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   Topic {topic_idx + 1}: {count:>3} documents ({percentage:>5.1f}%)")
        
        return df
    
    def get_representative_documents(self, df, n_docs=3):
        """Get representative documents for each topic"""
        print(f"\n📝 Representative Reviews for Each Topic:")
        print("=" * 80)
        
        for topic_idx in sorted(df['dominant_topic'].unique()):
            topic_docs = df[df['dominant_topic'] == topic_idx].nlargest(n_docs, 'topic_score')
            
            print(f"\n🔹 Topic {topic_idx + 1} - Representative Reviews:")
            for i, (idx, row) in enumerate(topic_docs.iterrows(), 1):
                review_text = row['review'][:120] + "..." if len(row['review']) > 120 else row['review']
                print(f"   {i}. [{row['topic_score']:.3f}] {review_text}")
    
    def create_visualizations(self, df, lsa_topics, lda_topics, output_dir):
        """Create topic modeling visualizations"""
        print("\n📊 Creating visualizations...")
        
        sns.set_style("whitegrid")
        
        # 1. Topic Distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        topic_counts = df['dominant_topic'].value_counts().sort_index()
        topic_labels = [f'Topic {i+1}' for i in topic_counts.index]
        
        ax.bar(topic_labels, topic_counts.values, color='steelblue')
        ax.set_xlabel('Topic')
        ax.set_ylabel('Number of Documents')
        ax.set_title('Distribution of Documents Across LSA Topics')
        
        for i, v in enumerate(topic_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'topic_distribution.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: topic_distribution.png")
        plt.close()
        
        # 2. Topic Score Distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(df['topic_score'], bins=30, color='coral', edgecolor='black')
        ax.set_xlabel('Topic Score (Confidence)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Topic Assignment Confidence')
        plt.tight_layout()
        plt.savefig(output_dir / 'topic_score_distribution.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: topic_score_distribution.png")
        plt.close()
        
        # 3. Word Clouds for Each LSA Topic
        print(f"\n☁️ Creating word clouds for LSA topics...")
        n_topics = len(lsa_topics)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for topic_idx, words_scores in lsa_topics.items():
            if topic_idx >= len(axes):
                break
            
            # Create word frequency dict
            word_freq = {word: score for word, score in words_scores[:20]}
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                colormap='viridis',
                relative_scaling=0.5
            ).generate_from_frequencies(word_freq)
            
            axes[topic_idx].imshow(wordcloud, interpolation='bilinear')
            axes[topic_idx].set_title(f'Topic {topic_idx + 1}', fontsize=12, fontweight='bold')
            axes[topic_idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_topics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'topic_wordclouds.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: topic_wordclouds.png")
        plt.close()
        
        # 4. Top Words per Topic (Bar chart)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for topic_idx, words_scores in lsa_topics.items():
            if topic_idx >= len(axes):
                break
            
            words = [w for w, s in words_scores[:10]]
            scores = [s for w, s in words_scores[:10]]
            
            axes[topic_idx].barh(range(len(words)), scores, color='steelblue')
            axes[topic_idx].set_yticks(range(len(words)))
            axes[topic_idx].set_yticklabels(words)
            axes[topic_idx].invert_yaxis()
            axes[topic_idx].set_xlabel('Weight')
            axes[topic_idx].set_title(f'Topic {topic_idx + 1}', fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_topics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'topic_top_words.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: topic_top_words.png")
        plt.close()
        
        # 5. Sentiment by Topic (if available)
        if 'vader_sentiment' in df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            sentiment_by_topic = df.groupby(['dominant_topic', 'vader_sentiment']).size().unstack(fill_value=0)
            sentiment_by_topic.index = [f'Topic {i+1}' for i in sentiment_by_topic.index]
            
            sentiment_by_topic.plot(kind='bar', stacked=True, ax=ax, 
                                   color=['red', 'gray', 'green'])
            ax.set_xlabel('Topic')
            ax.set_ylabel('Number of Reviews')
            ax.set_title('Sentiment Distribution by Topic')
            ax.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_dir / 'sentiment_by_topic.png', dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: sentiment_by_topic.png")
            plt.close()


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("PHASE 4: TOPIC MODELING (LSA & LDA)")
    print("=" * 80 + "\n")
    
    # File paths
    INPUT_FILE = "data/processed/reviews_sentiment.csv"
    OUTPUT_FILE = "data/processed/reviews_topics.csv"
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print(f"📂 Loading data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   ✅ Loaded {len(df)} reviews\n")
    except FileNotFoundError:
        print(f"   ❌ File not found: {INPUT_FILE}")
        print(f"   Please run Phase 3 (Sentiment Analysis) first!")
        return
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Use lemmatized text for better topic modeling
    text_column = 'lemmatized_text' if 'lemmatized_text' in df.columns else 'cleaned_text'
    documents = df[text_column].fillna('').astype(str).tolist()
    
    print(f"📝 Using column: {text_column}")
    print(f"📊 Number of documents: {len(documents)}")
    
    # Determine optimal number of topics (3-7 is typical for product reviews)
    n_topics = min(5, max(3, len(df) // 15))  # Heuristic: 1 topic per 15 docs
    print(f"🎯 Number of topics: {n_topics}\n")
    
    # Initialize modeler
    modeler = TopicModeler(n_topics=n_topics)
    
    # 1. LSA (Latent Semantic Analysis)
    tfidf_matrix = modeler.prepare_tfidf_matrix(documents, max_features=1000)
    doc_topic_lsa, svd = modeler.train_lsa_model()
    lsa_topics = modeler.get_top_words_per_topic(model_type='lsa', n_words=15)
    modeler.display_topics(lsa_topics, model_name='LSA', n_words=10)
    modeler.interpret_topics(lsa_topics, n_words=10)
    
    # 2. LDA (Latent Dirichlet Allocation)
    count_matrix = modeler.prepare_count_matrix(documents, max_features=1000)
    doc_topic_lda = modeler.train_lda_model(max_iter=10)
    lda_topics = modeler.get_top_words_per_topic(model_type='lda', n_words=15)
    modeler.display_topics(lda_topics, model_name='LDA', n_words=10)
    
    # 3. Assign topics to documents (using LSA)
    df = modeler.get_document_topics(doc_topic_lsa, df)
    
    # 4. Get representative documents
    modeler.get_representative_documents(df, n_docs=3)
    
    # 5. Create visualizations
    modeler.create_visualizations(df, lsa_topics, lda_topics, RESULTS_DIR)
    
    # Save results
    print(f"\n💾 Saving results to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved successfully!")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 4 COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved:")
    print(f"  - Data: {OUTPUT_FILE}")
    print(f"  - Visualizations: {RESULTS_DIR}/")
    print(f"\nNext: Phase 5 - Word Embeddings (7_embeddings.py)\n")


if __name__ == "__main__":
    main()