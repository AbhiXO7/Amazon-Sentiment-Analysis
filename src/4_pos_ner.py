"""
Phase 2: Part-of-Speech Tagging & Named Entity Recognition
Classical NLP approach using NLTK (no Transformers)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# NLTK imports
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.chunk import tree2conlltags
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure required NLTK data is downloaded
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading POS tagger...")
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    print("Downloading NE chunker...")
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading words corpus...")
    nltk.download('words')


class POSAndNERAnalyzer:
    """
    POS Tagging and Named Entity Recognition for Amazon Reviews
    Using classical NLTK approaches
    """
    
    def __init__(self):
        """Initialize analyzer"""
        print("🚀 Initializing POS & NER Analyzer...")
        
        # POS tag descriptions
        self.pos_descriptions = {
            'NN': 'Noun (singular)',
            'NNS': 'Noun (plural)',
            'NNP': 'Proper noun (singular)',
            'NNPS': 'Proper noun (plural)',
            'JJ': 'Adjective',
            'JJR': 'Adjective (comparative)',
            'JJS': 'Adjective (superlative)',
            'VB': 'Verb (base form)',
            'VBD': 'Verb (past tense)',
            'VBG': 'Verb (gerund)',
            'VBN': 'Verb (past participle)',
            'VBP': 'Verb (present)',
            'VBZ': 'Verb (3rd person singular)',
            'RB': 'Adverb',
            'RBR': 'Adverb (comparative)',
            'RBS': 'Adverb (superlative)',
        }
        
        # Product-related keywords (for domain-specific NER)
        self.product_keywords = [
            'battery', 'screen', 'camera', 'display', 'charging', 'performance',
            'quality', 'price', 'design', 'build', 'speaker', 'sound', 'processor',
            'storage', 'ram', 'memory', 'color', 'colour', 'size', 'weight',
            'charger', 'cable', 'box', 'package', 'delivery', 'shipping'
        ]
        
        print("   ✅ Analyzer initialized\n")
    
    def pos_tag_text(self, tokens):
        """Apply POS tagging to tokens"""
        if not tokens or len(tokens) == 0:
            return []
        try:
            return pos_tag(tokens)
        except:
            return []
    
    def extract_entities(self, text):
        """Extract named entities using NLTK's NE chunker"""
        if pd.isna(text) or not text:
            return []
        
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(str(text))
            pos_tags = pos_tag(tokens)
            
            # Named Entity Recognition
            ne_tree = ne_chunk(pos_tags)
            
            # Convert tree to IOB tags
            iob_tagged = tree2conlltags(ne_tree)
            
            # Extract entities
            entities = []
            current_entity = []
            current_label = None
            
            for word, pos, ne_tag in iob_tagged:
                if ne_tag.startswith('B-'):  # Beginning of entity
                    if current_entity:
                        entities.append((' '.join(current_entity), current_label))
                    current_entity = [word]
                    current_label = ne_tag[2:]
                elif ne_tag.startswith('I-'):  # Inside entity
                    current_entity.append(word)
                else:  # Outside entity
                    if current_entity:
                        entities.append((' '.join(current_entity), current_label))
                        current_entity = []
                        current_label = None
            
            # Add last entity if exists
            if current_entity:
                entities.append((' '.join(current_entity), current_label))
            
            return entities
        except:
            return []
    
    def extract_product_features(self, tokens):
        """Extract product-specific features (domain-specific NER)"""
        if not tokens:
            return []
        
        features = []
        for i, token in enumerate(tokens):
            # Check if token is a product keyword
            if token.lower() in self.product_keywords:
                # Get context (word before and after)
                context_before = tokens[i-1] if i > 0 else ''
                context_after = tokens[i+1] if i < len(tokens)-1 else ''
                
                features.append({
                    'feature': token,
                    'context_before': context_before,
                    'context_after': context_after,
                    'position': i
                })
        
        return features
    
    def analyze_pos_distribution(self, df):
        """Analyze POS tag distribution across all reviews"""
        print("\n" + "=" * 80)
        print("POS TAG ANALYSIS")
        print("=" * 80)
        
        all_pos_tags = []
        pos_tags_list = []
        
        print("\n🔤 Performing POS tagging...")
        for idx, row in df.iterrows():
            tokens = row['tokens'] if isinstance(row['tokens'], list) else []
            pos_tags = self.pos_tag_text(tokens)
            pos_tags_list.append(pos_tags)
            all_pos_tags.extend([tag for _, tag in pos_tags])
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} reviews...", end='\r')
        
        # Add as new column
        df['pos_tags'] = pos_tags_list
        
        print(f"\n   ✅ Completed POS tagging for {len(df)} reviews")
        
        # Count POS tags
        pos_counter = Counter(all_pos_tags)
        
        print(f"\n📊 Top 15 Most Common POS Tags:")
        print(f"{'Tag':<8} {'Description':<30} {'Count':<10} {'%':<8}")
        print("-" * 60)
        
        total_tags = sum(pos_counter.values())
        for tag, count in pos_counter.most_common(15):
            desc = self.pos_descriptions.get(tag, 'Other')
            percentage = (count / total_tags) * 100
            print(f"{tag:<8} {desc:<30} {count:<10} {percentage:>6.2f}%")
        
        return df, pos_counter
    
    def extract_adjectives_analysis(self, df):
        """Extract and analyze adjectives describing the product"""
        print("\n" + "=" * 80)
        print("ADJECTIVE ANALYSIS (Product Descriptors)")
        print("=" * 80)
        
        adjectives = []
        
        for idx, row in df.iterrows():
            pos_tags = row.get('pos_tags', [])
            if pos_tags:
                for word, tag in pos_tags:
                    if tag.startswith('JJ'):  # JJ, JJR, JJS
                        adjectives.append((word.lower(), tag))
        
        # Count adjectives
        adj_counter = Counter([adj for adj, _ in adjectives])
        
        print(f"\n📊 Top 20 Most Common Adjectives:")
        print(f"{'Adjective':<20} {'Count':<10}")
        print("-" * 35)
        
        for adj, count in adj_counter.most_common(20):
            print(f"{adj:<20} {count:<10}")
        
        return adj_counter
    
    def extract_verbs_analysis(self, df):
        """Extract and analyze verbs (actions/experiences)"""
        print("\n" + "=" * 80)
        print("VERB ANALYSIS (Customer Actions/Experiences)")
        print("=" * 80)
        
        verbs = []
        
        for idx, row in df.iterrows():
            pos_tags = row.get('pos_tags', [])
            if pos_tags:
                for word, tag in pos_tags:
                    if tag.startswith('VB'):  # VB, VBD, VBG, VBN, VBP, VBZ
                        verbs.append((word.lower(), tag))
        
        # Count verbs
        verb_counter = Counter([verb for verb, _ in verbs])
        
        print(f"\n📊 Top 20 Most Common Verbs:")
        print(f"{'Verb':<20} {'Count':<10}")
        print("-" * 35)
        
        for verb, count in verb_counter.most_common(20):
            print(f"{verb:<20} {count:<10}")
        
        return verb_counter
    
    def perform_ner(self, df):
        """Perform Named Entity Recognition"""
        print("\n" + "=" * 80)
        print("NAMED ENTITY RECOGNITION")
        print("=" * 80)
        
        all_entities = []
        entities_list = []
        
        print("\n🔍 Extracting named entities...")
        for idx, row in df.iterrows():
            # Use cleaned_text for NER (better results)
            text = row.get('cleaned_text', '')
            entities = self.extract_entities(text)
            entities_list.append(entities)
            all_entities.extend(entities)
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} reviews...", end='\r')
        
        # Add as new column
        df['entities'] = entities_list
        
        print(f"\n   ✅ Completed NER for {len(df)} reviews")
        
        # Count entities by type
        entity_by_type = defaultdict(list)
        for entity, label in all_entities:
            entity_by_type[label].append(entity)
        
        print(f"\n📊 Named Entities Found:")
        print(f"{'Type':<20} {'Count':<10} {'Examples'}")
        print("-" * 70)
        
        for entity_type, entities in entity_by_type.items():
            entity_counter = Counter(entities)
            top_entities = [e for e, _ in entity_counter.most_common(3)]
            print(f"{entity_type:<20} {len(entities):<10} {', '.join(top_entities[:3])}")
        
        # Show most common entities
        print(f"\n📊 Top 10 Most Mentioned Entities:")
        all_entity_names = [entity for entity, _ in all_entities]
        entity_counter = Counter(all_entity_names)
        
        print(f"{'Entity':<30} {'Count':<10}")
        print("-" * 45)
        for entity, count in entity_counter.most_common(10):
            print(f"{entity:<30} {count:<10}")
        
        return df, entity_by_type
    
    def extract_product_features_analysis(self, df):
        """Extract product-specific features"""
        print("\n" + "=" * 80)
        print("PRODUCT FEATURE EXTRACTION (Domain-Specific NER)")
        print("=" * 80)
        
        all_features = []
        
        print("\n🔍 Extracting product features...")
        for idx, row in df.iterrows():
            tokens = row['tokens'] if isinstance(row['tokens'], list) else []
            features = self.extract_product_features(tokens)
            df.at[idx, 'product_features'] = features
            all_features.extend(features)
        
        print(f"   ✅ Found {len(all_features)} product feature mentions")
        
        # Count features
        feature_counter = Counter([f['feature'] for f in all_features])
        
        print(f"\n📊 Top 15 Most Discussed Product Features:")
        print(f"{'Feature':<20} {'Count':<10}")
        print("-" * 35)
        
        for feature, count in feature_counter.most_common(15):
            print(f"{feature:<20} {count:<10}")
        
        # Show feature with context
        print(f"\n📝 Sample Feature Contexts (Top 3 features):")
        print("-" * 70)
        
        for feature, count in feature_counter.most_common(3):
            print(f"\n🔹 Feature: {feature.upper()} (mentioned {count} times)")
            # Get 3 example contexts
            examples = [f for f in all_features if f['feature'] == feature][:3]
            for i, ex in enumerate(examples, 1):
                context = f"{ex['context_before']} [{feature}] {ex['context_after']}"
                print(f"   {i}. ...{context}...")
        
        return df, feature_counter
    
    def create_visualizations(self, pos_counter, adj_counter, verb_counter, feature_counter, output_dir):
        """Create visualization plots"""
        print("\n📊 Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # 1. POS Tag Distribution
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        top_pos = dict(pos_counter.most_common(15))
        ax.bar(range(len(top_pos)), list(top_pos.values()), color='steelblue')
        ax.set_xticks(range(len(top_pos)))
        ax.set_xticklabels(list(top_pos.keys()), rotation=45)
        ax.set_xlabel('POS Tag')
        ax.set_ylabel('Frequency')
        ax.set_title('Top 15 Part-of-Speech Tags Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / 'pos_distribution.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: pos_distribution.png")
        plt.close()
        
        # 2. Top Adjectives
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        top_adj = dict(adj_counter.most_common(20))
        ax.barh(range(len(top_adj)), list(top_adj.values()), color='coral')
        ax.set_yticks(range(len(top_adj)))
        ax.set_yticklabels(list(top_adj.keys()))
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_title('Top 20 Adjectives (Product Descriptors)')
        plt.tight_layout()
        plt.savefig(output_dir / 'top_adjectives.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: top_adjectives.png")
        plt.close()
        
        # 3. Top Verbs
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        top_verbs = dict(verb_counter.most_common(20))
        ax.barh(range(len(top_verbs)), list(top_verbs.values()), color='lightgreen')
        ax.set_yticks(range(len(top_verbs)))
        ax.set_yticklabels(list(top_verbs.keys()))
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_title('Top 20 Verbs (Customer Actions)')
        plt.tight_layout()
        plt.savefig(output_dir / 'top_verbs.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: top_verbs.png")
        plt.close()
        
        # 4. Product Features
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        top_features = dict(feature_counter.most_common(15))
        ax.barh(range(len(top_features)), list(top_features.values()), color='mediumpurple')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(list(top_features.keys()))
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_title('Top 15 Product Features Mentioned')
        plt.tight_layout()
        plt.savefig(output_dir / 'product_features.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: product_features.png")
        plt.close()


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("PHASE 2: POS TAGGING & NAMED ENTITY RECOGNITION")
    print("=" * 80 + "\n")
    
    # File paths
    INPUT_FILE = "data/processed/reviews_preprocessed.csv"
    OUTPUT_FILE = "data/processed/reviews_pos_ner.csv"
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load preprocessed data
    print(f"📂 Loading preprocessed data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        
        # Convert string representation of lists back to actual lists
        import ast
        df['tokens'] = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        print(f"   ✅ Loaded {len(df)} reviews\n")
    except FileNotFoundError:
        print(f"   ❌ File not found: {INPUT_FILE}")
        print(f"   Please run Phase 1 (preprocessing) first!")
        return
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Initialize analyzer
    analyzer = POSAndNERAnalyzer()
    
    # 1. POS Tagging Analysis
    df, pos_counter = analyzer.analyze_pos_distribution(df)
    
    # 2. Adjective Analysis
    adj_counter = analyzer.extract_adjectives_analysis(df)
    
    # 3. Verb Analysis
    verb_counter = analyzer.extract_verbs_analysis(df)
    
    # 4. Named Entity Recognition
    df, entity_by_type = analyzer.perform_ner(df)
    
    # 5. Product Feature Extraction
    df, feature_counter = analyzer.extract_product_features_analysis(df)
    
    # 6. Create visualizations
    analyzer.create_visualizations(pos_counter, adj_counter, verb_counter, 
                                   feature_counter, RESULTS_DIR)
    
    # Save processed data
    print(f"\n💾 Saving results to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved successfully!")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 2 COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved:")
    print(f"  - Data: {OUTPUT_FILE}")
    print(f"  - Visualizations: {RESULTS_DIR}/")
    print(f"\nNext: Phase 3 - Sentiment Analysis (5_sentiment.py)\n")


if __name__ == "__main__":
    main()