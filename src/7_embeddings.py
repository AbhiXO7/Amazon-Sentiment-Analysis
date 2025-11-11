"""
Phase 5: Word Embeddings & Semantic Similarity
Word2Vec and GloVe embeddings (Classical NLP)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Word Embeddings
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import gensim.downloader as api

# Similarity & Distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# NLTK
import nltk
from collections import Counter


class EmbeddingsAnalyzer:
    """
    Word Embeddings and Semantic Similarity Analysis
    Using Word2Vec (trained) and GloVe (pre-trained)
    """
    
    def __init__(self):
        """Initialize embeddings analyzer"""
        print("🚀 Initializing Embeddings Analyzer...")
        
        self.word2vec_model = None
        self.glove_model = None
        self.phraser = None
        self.vocab = None
        
        print()
    
    def prepare_sentences(self, df, token_column='lemmatized_tokens'):
        """Prepare sentences for Word2Vec training"""
        print("\n📝 Preparing sentences for training...")
        
        import ast
        
        sentences = []
        for idx, row in df.iterrows():
            tokens = row[token_column]
            
            # Convert string representation to list if needed
            if isinstance(tokens, str):
                try:
                    tokens = ast.literal_eval(tokens)
                except:
                    tokens = tokens.split()
            
            if isinstance(tokens, list) and len(tokens) > 0:
                sentences.append(tokens)
        
        print(f"   ✅ Prepared {len(sentences)} sentences")
        print(f"   ✅ Average sentence length: {np.mean([len(s) for s in sentences]):.1f} tokens")
        
        return sentences
    
    def detect_phrases(self, sentences):
        """Detect common phrases/bigrams"""
        print("\n🔍 Detecting common phrases (bigrams)...")
        
        # Train phrase detector
        phrases = Phrases(sentences, min_count=3, threshold=10)
        self.phraser = Phraser(phrases)
        
        # Transform sentences with phrases
        sentences_with_phrases = [self.phraser[sentence] for sentence in sentences]
        
        # Show some examples
        common_bigrams = []
        for sent in sentences_with_phrases:
            common_bigrams.extend([token for token in sent if '_' in token])
        
        bigram_counts = Counter(common_bigrams).most_common(10)
        
        if bigram_counts:
            print(f"   ✅ Top 10 detected phrases:")
            for phrase, count in bigram_counts:
                print(f"      {phrase}: {count}")
        else:
            print(f"   ℹ️ No significant phrases detected")
        
        return sentences_with_phrases
    
    def train_word2vec(self, sentences, vector_size=100, window=5, min_count=2, epochs=10):
        """Train Word2Vec model"""
        print("\n" + "=" * 80)
        print("TRAINING WORD2VEC MODEL")
        print("=" * 80)
        
        print(f"\n🎯 Training parameters:")
        print(f"   Vector size: {vector_size}")
        print(f"   Window size: {window}")
        print(f"   Min count: {min_count}")
        print(f"   Epochs: {epochs}")
        
        # Train Word2Vec
        print(f"\n⚙️ Training Word2Vec model...")
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,  # Skip-gram (better for small datasets)
            epochs=epochs
        )
        
        self.vocab = list(self.word2vec_model.wv.index_to_key)
        
        print(f"   ✅ Training complete!")
        print(f"   ✅ Vocabulary size: {len(self.vocab)}")
        print(f"   ✅ Vector dimensions: {self.word2vec_model.wv.vector_size}")
        
        return self.word2vec_model
    
    def load_glove_embeddings(self, model_name='glove-wiki-gigaword-50'):
        """Load pre-trained GloVe embeddings"""
        print("\n" + "=" * 80)
        print("LOADING PRE-TRAINED GLOVE EMBEDDINGS")
        print("=" * 80)
        
        print(f"\n📥 Downloading GloVe model: {model_name}")
        print(f"   (This may take a few minutes on first run...)")
        
        try:
            self.glove_model = api.load(model_name)
            print(f"   ✅ GloVe model loaded!")
            print(f"   ✅ Vocabulary size: {len(self.glove_model.index_to_key)}")
            print(f"   ✅ Vector dimensions: {self.glove_model.vector_size}")
        except Exception as e:
            print(f"   ⚠️ Could not load GloVe: {e}")
            print(f"   ℹ️ Continuing with Word2Vec only...")
            self.glove_model = None
        
        return self.glove_model
    
    def find_similar_words(self, word, model_type='word2vec', topn=10):
        """Find most similar words to a given word"""
        
        if model_type == 'word2vec':
            model = self.word2vec_model
        elif model_type == 'glove':
            model = self.glove_model
        else:
            return None
        
        if model is None:
            return None
        
        try:
            if model_type == 'word2vec':
                similar = model.wv.most_similar(word, topn=topn)
            else:
                similar = model.most_similar(word, topn=topn)
            return similar
        except KeyError:
            return None
    
    def analyze_product_features(self, feature_words):
        """Analyze semantic similarity for product features"""
        print("\n" + "=" * 80)
        print("PRODUCT FEATURE SEMANTIC SIMILARITY ANALYSIS")
        print("=" * 80)
        
        for feature in feature_words:
            print(f"\n🔹 Feature: '{feature.upper()}'")
            print("-" * 70)
            
            # Word2Vec similarities
            print(f"\n   Word2Vec - Top 5 similar words:")
            w2v_similar = self.find_similar_words(feature, 'word2vec', topn=5)
            if w2v_similar:
                for i, (word, score) in enumerate(w2v_similar, 1):
                    print(f"      {i}. {word:<20} (similarity: {score:.4f})")
            else:
                print(f"      '{feature}' not in Word2Vec vocabulary")
            
            # GloVe similarities
            if self.glove_model:
                print(f"\n   GloVe - Top 5 similar words:")
                glove_similar = self.find_similar_words(feature, 'glove', topn=5)
                if glove_similar:
                    for i, (word, score) in enumerate(glove_similar, 1):
                        print(f"      {i}. {word:<20} (similarity: {score:.4f})")
                else:
                    print(f"      '{feature}' not in GloVe vocabulary")
    
    def word_analogy(self, positive, negative, model_type='word2vec', topn=5):
        """Perform word analogy (e.g., king - man + woman = queen)"""
        
        model = self.word2vec_model if model_type == 'word2vec' else self.glove_model
        
        if model is None:
            return None
        
        try:
            if model_type == 'word2vec':
                result = model.wv.most_similar(positive=positive, negative=negative, topn=topn)
            else:
                result = model.most_similar(positive=positive, negative=negative, topn=topn)
            return result
        except KeyError:
            return None
    
    def analyze_analogies(self):
        """Analyze product-related analogies"""
        print("\n" + "=" * 80)
        print("WORD ANALOGIES")
        print("=" * 80)
        
        # Product-related analogies
        analogies = [
            (['good', 'battery'], ['bad'], "good:battery :: bad:?"),
            (['fast', 'charging'], ['slow'], "fast:charging :: slow:?"),
            (['excellent', 'camera'], ['poor'], "excellent:camera :: poor:?"),
            (['high', 'quality'], ['low'], "high:quality :: low:?"),
        ]
        
        for positive, negative, description in analogies:
            print(f"\n🔹 Analogy: {description}")
            
            result = self.word_analogy(positive, negative, 'word2vec', topn=3)
            if result:
                print(f"   Results:")
                for i, (word, score) in enumerate(result, 1):
                    print(f"      {i}. {word} (score: {score:.4f})")
            else:
                print(f"   ⚠️ Could not compute analogy (words not in vocabulary)")
    
    def compute_document_vectors(self, sentences):
        """Compute document vectors by averaging word vectors"""
        print(f"\n📊 Computing document vectors...")
        
        doc_vectors = []
        
        for sentence in sentences:
            # Get vectors for words in sentence
            vectors = []
            for word in sentence:
                try:
                    vectors.append(self.word2vec_model.wv[word])
                except KeyError:
                    continue
            
            # Average vectors
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(self.word2vec_model.wv.vector_size)
            
            doc_vectors.append(doc_vector)
        
        doc_vectors = np.array(doc_vectors)
        print(f"   ✅ Document vectors shape: {doc_vectors.shape}")
        
        return doc_vectors
    
    def find_similar_reviews(self, doc_vectors, df, review_idx, topn=5):
        """Find most similar reviews based on document vectors"""
        
        target_vector = doc_vectors[review_idx].reshape(1, -1)
        similarities = cosine_similarity(target_vector, doc_vectors)[0]
        
        # Get top similar (excluding the review itself)
        similar_indices = similarities.argsort()[::-1][1:topn+1]
        
        return similar_indices, similarities[similar_indices]
    
    def analyze_review_similarity(self, doc_vectors, df, n_examples=3):
        """Analyze review similarity"""
        print("\n" + "=" * 80)
        print("REVIEW SIMILARITY ANALYSIS")
        print("=" * 80)
        
        # Pick random reviews to analyze
        sample_indices = np.random.choice(len(df), size=min(n_examples, len(df)), replace=False)
        
        for sample_idx in sample_indices:
            print(f"\n🔹 Original Review:")
            print(f"   {df.iloc[sample_idx]['review'][:100]}...")
            
            similar_indices, scores = self.find_similar_reviews(doc_vectors, df, sample_idx, topn=3)
            
            print(f"\n   Top 3 Similar Reviews:")
            for i, (idx, score) in enumerate(zip(similar_indices, scores), 1):
                print(f"\n   {i}. [Similarity: {score:.4f}]")
                print(f"      {df.iloc[idx]['review'][:100]}...")
    
    def create_visualizations(self, sentences, doc_vectors, df, output_dir):
        """Create embedding visualizations"""
        print("\n📊 Creating visualizations...")
        
        sns.set_style("whitegrid")
        
        # 1. Word Embeddings Visualization (t-SNE)
        print(f"\n   Creating word embedding visualization...")
        
        # Get most common words
        all_words = [word for sent in sentences for word in sent]
        word_counts = Counter(all_words)
        top_words = [word for word, count in word_counts.most_common(100)]
        
        # Filter words in vocabulary
        top_words = [w for w in top_words if w in self.word2vec_model.wv][:50]
        
        if len(top_words) > 10:
            # Get word vectors
            word_vectors = np.array([self.word2vec_model.wv[word] for word in top_words])
            
            # Reduce to 2D using t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(top_words)-1))
            word_vectors_2d = tsne.fit_transform(word_vectors)
            
            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], alpha=0.6, s=100)
            
            for i, word in enumerate(top_words):
                ax.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
                           fontsize=9, alpha=0.8)
            
            ax.set_title('Word Embeddings Visualization (t-SNE)', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            plt.tight_layout()
            plt.savefig(output_dir / 'word_embeddings_tsne.png', dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: word_embeddings_tsne.png")
            plt.close()
        
        # 2. Document Vectors Visualization (PCA)
        print(f"\n   Creating document vectors visualization...")
        
        pca = PCA(n_components=2)
        doc_vectors_2d = pca.fit_transform(doc_vectors)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Color by sentiment if available
        if 'vader_sentiment' in df.columns:
            sentiment_colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            for sentiment in ['positive', 'neutral', 'negative']:
                mask = df['vader_sentiment'] == sentiment
                ax.scatter(doc_vectors_2d[mask, 0], doc_vectors_2d[mask, 1],
                          c=sentiment_colors[sentiment], label=sentiment.capitalize(),
                          alpha=0.6, s=50)
            ax.legend()
        else:
            ax.scatter(doc_vectors_2d[:, 0], doc_vectors_2d[:, 1], alpha=0.6, s=50)
        
        ax.set_title('Document Vectors Visualization (PCA)', fontsize=14, fontweight='bold')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        plt.tight_layout()
        plt.savefig(output_dir / 'document_vectors_pca.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: document_vectors_pca.png")
        plt.close()
        
        # 3. Cosine Similarity Heatmap (sample)
        print(f"\n   Creating similarity heatmap...")
        
        # Take sample of reviews
        n_sample = min(20, len(doc_vectors))
        sample_indices = np.random.choice(len(doc_vectors), n_sample, replace=False)
        sample_vectors = doc_vectors[sample_indices]
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(sample_vectors)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.heatmap(sim_matrix, cmap='coolwarm', center=0.5, 
                   xticklabels=[f'R{i+1}' for i in range(n_sample)],
                   yticklabels=[f'R{i+1}' for i in range(n_sample)],
                   ax=ax, cbar_kws={'label': 'Cosine Similarity'})
        ax.set_title('Review Similarity Matrix (Sample)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'review_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: review_similarity_heatmap.png")
        plt.close()
        
        # 4. Word Similarity Network (for key features)
        print(f"\n   Creating word similarity network...")
        
        key_features = ['battery', 'camera', 'display', 'quality', 'price', 'performance']
        available_features = [f for f in key_features if f in self.word2vec_model.wv]
        
        if len(available_features) >= 3:
            # Compute similarity matrix
            feature_vectors = np.array([self.word2vec_model.wv[f] for f in available_features])
            feature_sim = cosine_similarity(feature_vectors)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sns.heatmap(feature_sim, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=available_features,
                       yticklabels=available_features,
                       ax=ax, cbar_kws={'label': 'Cosine Similarity'})
            ax.set_title('Product Feature Similarity Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_similarity_matrix.png', dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: feature_similarity_matrix.png")
            plt.close()


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("PHASE 5: WORD EMBEDDINGS & SEMANTIC SIMILARITY")
    print("=" * 80 + "\n")
    
    # File paths
    INPUT_FILE = "data/processed/reviews_topics.csv"
    OUTPUT_FILE = "data/processed/reviews_embeddings.csv"
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print(f"📂 Loading data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   ✅ Loaded {len(df)} reviews\n")
    except FileNotFoundError:
        print(f"   ❌ File not found: {INPUT_FILE}")
        print(f"   Please run Phase 4 (Topic Modeling) first!")
        return
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Initialize analyzer
    analyzer = EmbeddingsAnalyzer()
    
    # 1. Prepare sentences
    sentences = analyzer.prepare_sentences(df, token_column='lemmatized_tokens')
    
    # 2. Detect phrases
    sentences_with_phrases = analyzer.detect_phrases(sentences)
    
    # 3. Train Word2Vec
    analyzer.train_word2vec(sentences_with_phrases, vector_size=100, window=5, 
                           min_count=2, epochs=10)
    
    # 4. Load GloVe (optional)
    analyzer.load_glove_embeddings('glove-wiki-gigaword-50')
    
    # 5. Analyze product features
    feature_words = ['battery', 'camera', 'display', 'screen', 'quality', 'price', 
                     'charging', 'performance', 'delivery', 'product']
    
    # Filter features that exist in vocabulary
    available_features = [f for f in feature_words if f in analyzer.word2vec_model.wv]
    if available_features:
        analyzer.analyze_product_features(available_features[:5])
    else:
        print("\n⚠️ Key feature words not found in vocabulary")
    
    # 6. Word analogies
    analyzer.analyze_analogies()
    
    # 7. Document vectors
    doc_vectors = analyzer.compute_document_vectors(sentences_with_phrases)
    
    # 8. Review similarity analysis
    analyzer.analyze_review_similarity(doc_vectors, df, n_examples=3)
    
    # 9. Create visualizations
    analyzer.create_visualizations(sentences_with_phrases, doc_vectors, df, RESULTS_DIR)
    
    # Save document vectors
    print(f"\n💾 Saving results...")
    
    # Add document vectors as a column (save as string for CSV)
    df['doc_vector'] = [vec.tolist() for vec in doc_vectors]
    
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved to: {OUTPUT_FILE}")
    
    # Save Word2Vec model
    model_path = RESULTS_DIR / "word2vec_model.bin"
    analyzer.word2vec_model.save(str(model_path))
    print(f"   ✅ Saved Word2Vec model to: {model_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 5 COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved:")
    print(f"  - Data: {OUTPUT_FILE}")
    print(f"  - Word2Vec Model: {model_path}")
    print(f"  - Visualizations: {RESULTS_DIR}/")
    print(f"\nNext: Phase 6 - Review Summarization (8_summarization.py)\n")


if __name__ == "__main__":
    main()