"""
Phase 3: Sentiment Analysis
Lexicon-based (VADER) + LSTM approach
Classical NLP (no Transformers)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sentiment Analysis
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Deep Learning for LSTM
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Download VADER lexicon if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')


class SentimentAnalyzer:
    """
    Dual approach sentiment analysis:
    1. Lexicon-based (VADER)
    2. Deep Learning (LSTM)
    """
    
    def __init__(self):
        """Initialize sentiment analyzers"""
        print("🚀 Initializing Sentiment Analyzer...")
        
        # VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        print("   ✅ VADER initialized")
        
        # LSTM parameters
        self.max_words = 5000
        self.max_len = 100
        self.embedding_dim = 128
        self.tokenizer = None
        self.lstm_model = None
        
        print()
    
    def vader_sentiment(self, text):
        """Get VADER sentiment scores"""
        if pd.isna(text) or not text:
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        
        scores = self.vader.polarity_scores(str(text))
        return scores
    
    def classify_sentiment(self, compound_score):
        """Classify sentiment based on compound score"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_vader_sentiment(self, df, text_column='cleaned_text'):
        """Analyze sentiment using VADER"""
        print("=" * 80)
        print("VADER SENTIMENT ANALYSIS (Lexicon-based)")
        print("=" * 80)
        
        print(f"\n🔍 Analyzing sentiment with VADER...")
        
        # Get VADER scores for each review
        vader_scores = []
        for idx, row in df.iterrows():
            text = row[text_column]
            scores = self.vader_sentiment(text)
            vader_scores.append(scores)
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} reviews...", end='\r')
        
        print(f"\n   ✅ Completed VADER analysis for {len(df)} reviews")
        
        # Add scores to dataframe
        df['vader_neg'] = [s['neg'] for s in vader_scores]
        df['vader_neu'] = [s['neu'] for s in vader_scores]
        df['vader_pos'] = [s['pos'] for s in vader_scores]
        df['vader_compound'] = [s['compound'] for s in vader_scores]
        df['vader_sentiment'] = df['vader_compound'].apply(self.classify_sentiment)
        
        # Statistics
        print(f"\n📊 VADER Sentiment Distribution:")
        sentiment_counts = df['vader_sentiment'].value_counts()
        total = len(df)
        
        for sentiment in ['positive', 'neutral', 'negative']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / total) * 100
            print(f"   {sentiment.capitalize():<10}: {count:>4} ({percentage:>5.1f}%)")
        
        print(f"\n📊 VADER Score Statistics:")
        print(f"   Average compound score: {df['vader_compound'].mean():.3f}")
        print(f"   Score range: {df['vader_compound'].min():.3f} to {df['vader_compound'].max():.3f}")
        
        return df
    
    def prepare_lstm_data(self, df, text_column='cleaned_text', rating_column='rating_score'):
        """Prepare data for LSTM training"""
        print("\n" + "=" * 80)
        print("PREPARING DATA FOR LSTM MODEL")
        print("=" * 80)
        
        # Convert ratings to sentiment labels
        print(f"\n🔄 Converting ratings to sentiment labels...")
        
        def rating_to_sentiment(rating):
            """Convert 5-star rating to sentiment label"""
            if pd.isna(rating):
                return None
            if rating >= 4.0:
                return 2  # positive
            elif rating <= 2.0:
                return 0  # negative
            else:
                return 1  # neutral
        
        df['sentiment_label'] = df[rating_column].apply(rating_to_sentiment)
        
        # Remove rows without labels
        df_labeled = df.dropna(subset=['sentiment_label']).copy()
        df_labeled['sentiment_label'] = df_labeled['sentiment_label'].astype(int)
        
        print(f"   ✅ Labeled reviews: {len(df_labeled)}")
        print(f"\n📊 Label Distribution:")
        label_counts = df_labeled['sentiment_label'].value_counts().sort_index()
        label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        for label, count in label_counts.items():
            print(f"   {label_names[label]:<10}: {count:>4}")
        
        # Tokenize text
        print(f"\n✂️ Tokenizing text for LSTM...")
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df_labeled[text_column])
        
        sequences = self.tokenizer.texts_to_sequences(df_labeled[text_column])
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        y = keras.utils.to_categorical(df_labeled['sentiment_label'], num_classes=3)
        
        print(f"   ✅ Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"   ✅ Sequence shape: {X.shape}")
        
        return X, y, df_labeled
    
    def build_lstm_model(self):
        """Build LSTM model architecture"""
        print(f"\n🏗️ Building LSTM model...")
        
        model = Sequential([
            Embedding(input_dim=self.max_words, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_len),
            
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   ✅ Model built successfully")
        print(f"\n📋 Model Architecture:")
        model.summary()
        
        return model
    
    def train_lstm_model(self, X, y, epochs=10, batch_size=16):
        """Train LSTM model"""
        print("\n" + "=" * 80)
        print("TRAINING LSTM MODEL")
        print("=" * 80)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Build model
        self.lstm_model = self.build_lstm_model()
        
        # Train
        print(f"\n🎯 Training LSTM model...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print()
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate
        print(f"\n📊 Evaluating model...")
        test_loss, test_acc = self.lstm_model.evaluate(X_test, y_test, verbose=0)
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
        
        # Predictions
        y_pred = self.lstm_model.predict(X_test, verbose=0)
        y_pred_classes = y_pred.argmax(axis=1)
        y_test_classes = y_test.argmax(axis=1)
        
        # Classification report
        print(f"\n📋 Classification Report:")
        label_names = ['Negative', 'Neutral', 'Positive']
        
        # Get unique classes present in the data
        unique_classes = np.unique(np.concatenate([y_test_classes, y_pred_classes]))
        present_labels = [label_names[i] for i in unique_classes]
        
        print(classification_report(y_test_classes, y_pred_classes, 
                                   labels=unique_classes,
                                   target_names=present_labels,
                                   zero_division=0))
        
        return history, (X_test, y_test, y_pred_classes, y_test_classes)
    
    def predict_lstm_sentiment(self, text):
        """Predict sentiment using trained LSTM model"""
        if not self.lstm_model or not self.tokenizer:
            return None
        
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')
        prediction = self.lstm_model.predict(padded, verbose=0)[0]
        
        label_names = ['negative', 'neutral', 'positive']
        predicted_class = prediction.argmax()
        
        return {
            'label': label_names[predicted_class],
            'confidence': float(prediction[predicted_class]),
            'scores': {label_names[i]: float(prediction[i]) for i in range(3)}
        }
    
    def extract_sentiment_phrases(self, df, sentiment='positive', n=10):
        """Extract key phrases for specific sentiment"""
        print(f"\n🔍 Extracting key {sentiment} phrases...")
        
        # Filter by sentiment
        sentiment_df = df[df['vader_sentiment'] == sentiment].copy()
        
        if len(sentiment_df) == 0:
            print(f"   No {sentiment} reviews found")
            return []
        
        # Sort by compound score
        if sentiment == 'positive':
            sentiment_df = sentiment_df.nlargest(n, 'vader_compound')
        elif sentiment == 'negative':
            sentiment_df = sentiment_df.nsmallest(n, 'vader_compound')
        else:
            sentiment_df = sentiment_df.head(n)
        
        print(f"   ✅ Found {len(sentiment_df)} {sentiment} reviews")
        
        return sentiment_df
    
    def create_visualizations(self, df, history, test_results, output_dir):
        """Create sentiment analysis visualizations"""
        print("\n📊 Creating visualizations...")
        
        sns.set_style("whitegrid")
        
        # 1. VADER Sentiment Distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sentiment_counts = df['vader_sentiment'].value_counts()
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        ax.bar(sentiment_counts.index, sentiment_counts.values, 
               color=[colors[s] for s in sentiment_counts.index])
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.set_title('VADER Sentiment Distribution')
        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'vader_sentiment_distribution.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: vader_sentiment_distribution.png")
        plt.close()
        
        # 2. VADER Compound Score Distribution
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.hist(df['vader_compound'], bins=30, color='steelblue', edgecolor='black')
        ax.axvline(x=0.05, color='green', linestyle='--', label='Positive threshold')
        ax.axvline(x=-0.05, color='red', linestyle='--', label='Negative threshold')
        ax.set_xlabel('VADER Compound Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of VADER Compound Scores')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'vader_compound_distribution.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: vader_compound_distribution.png")
        plt.close()
        
        # 3. Rating vs VADER Sentiment
        if 'rating_score' in df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            df_plot = df.dropna(subset=['rating_score'])
            sentiment_order = ['negative', 'neutral', 'positive']
            ax.boxplot([df_plot[df_plot['vader_sentiment'] == s]['rating_score'].values 
                       for s in sentiment_order],
                      labels=sentiment_order)
            ax.set_xlabel('VADER Sentiment')
            ax.set_ylabel('Rating Score')
            ax.set_title('Rating Distribution by VADER Sentiment')
            plt.tight_layout()
            plt.savefig(output_dir / 'rating_vs_sentiment.png', dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: rating_vs_sentiment.png")
            plt.close()
        
        # 4. LSTM Training History
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('LSTM Model Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('LSTM Model Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'lstm_training_history.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: lstm_training_history.png")
        plt.close()
        
        # 5. LSTM Confusion Matrix
        _, _, y_pred, y_test = test_results
        
        # Get unique classes
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        label_names = ['Negative', 'Neutral', 'Positive']
        present_labels = [label_names[i] for i in unique_classes]
        
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=present_labels,
                   yticklabels=present_labels,
                   ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('LSTM Model - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: lstm_confusion_matrix.png")
        plt.close()


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("PHASE 3: SENTIMENT ANALYSIS")
    print("=" * 80 + "\n")
    
    # File paths
    INPUT_FILE = "data/processed/reviews_pos_ner.csv"
    OUTPUT_FILE = "data/processed/reviews_sentiment.csv"
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print(f"📂 Loading data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   ✅ Loaded {len(df)} reviews\n")
    except FileNotFoundError:
        print(f"   ❌ File not found: {INPUT_FILE}")
        print(f"   Please run Phase 2 (POS & NER) first!")
        return
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # 1. VADER Sentiment Analysis
    df = analyzer.analyze_vader_sentiment(df, text_column='cleaned_text')
    
    # 2. Extract sentiment phrases
    print("\n" + "=" * 80)
    print("KEY SENTIMENT PHRASES")
    print("=" * 80)
    
    for sentiment in ['positive', 'negative']:
        phrases_df = analyzer.extract_sentiment_phrases(df, sentiment=sentiment, n=5)
        print(f"\n🔹 Top {sentiment.upper()} reviews:")
        print("-" * 80)
        for idx, (i, row) in enumerate(phrases_df.iterrows(), 1):
            print(f"{idx}. [{row['vader_compound']:.3f}] {row['review'][:100]}...")
    
    # 3. LSTM Model Training
    if 'rating_score' in df.columns:
        X, y, df_labeled = analyzer.prepare_lstm_data(df)
        
        if len(df_labeled) >= 20:  # Minimum samples for training
            history, test_results = analyzer.train_lstm_model(X, y, epochs=10, batch_size=16)
            
            # 4. Create visualizations
            analyzer.create_visualizations(df, history, test_results, RESULTS_DIR)
        else:
            print("\n⚠️ Not enough labeled data for LSTM training (need at least 20 samples)")
            history = None
            test_results = None
    else:
        print("\n⚠️ Rating column not found. Skipping LSTM training.")
        history = None
        test_results = None
    
    # Save results
    print(f"\n💾 Saving results to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved successfully!")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 3 COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved:")
    print(f"  - Data: {OUTPUT_FILE}")
    print(f"  - Visualizations: {RESULTS_DIR}/")
    print(f"\nNext: Phase 4 - Topic Modeling (6_topic_modeling.py)\n")


if __name__ == "__main__":
    main()