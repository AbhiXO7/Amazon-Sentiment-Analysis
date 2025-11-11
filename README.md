# Amazon Reviews NLP Analysis Pipeline

A comprehensive Natural Language Processing pipeline for analyzing Amazon product reviews with multiple phases of analysis including preprocessing, sentiment analysis, topic modeling, embeddings, summarization, and question-answering.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Prerequisites & Installation](#prerequisites--installation)
- [Project Structure](#project-structure)
- [Phase-by-Phase Guide](#phase-by-phase-guide)
- [Running the Pipeline](#running-the-pipeline)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

--------------------------------------

## 🎯 Project Overview

This project provides a complete end-to-end NLP pipeline for Amazon product review analysis. It combines classical NLP techniques with modern deep learning approaches to extract meaningful insights from customer reviews.

### Key Features:

✅ **Web Scraping**: Automated review collection from Amazon.in
✅ **Data Preprocessing**: Text cleaning, tokenization, lemmatization
✅ **POS Tagging & NER**: Grammatical analysis and entity extraction
✅ **Sentiment Analysis**: VADER lexicon-based + LSTM deep learning
✅ **Topic Modeling**: LSA and LDA for discovering review themes
✅ **Word Embeddings**: Word2Vec training and semantic similarity
✅ **Summarization**: Extractive and abstractive review summaries
✅ **QA System**: Interactive question-answering over reviews

---

## 🏗️ System Architecture

--------------------------------------

## 📦 Prerequisites & Installation

### System Requirements:

- Python 3.8+
- 4GB RAM minimum (8GB+ recommended)
- Internet connection (for downloads and scraping)
- Chrome browser (for Selenium-based scraping)

### Installation Steps:

```bash
# 1. Clone or download the project
cd d:\Coding\NLP

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r [requirements.txt](http://_vscodecontentref_/0)

# 4. Download NLTK data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('vader_lexicon')"

--------------------------------------

Required Libraries:

pandas >= 1.3.0
numpy >= 1.21.0
nltk >= 3.6.0
scikit-learn >= 0.24.0
gensim >= 4.0.0
tensorflow >= 2.6.0
keras >= 2.4.0
selenium >= 3.141.0
beautifulsoup4 >= 4.9.0
wordcloud >= 1.8.0
matplotlib >= 3.4.0
seaborn >= 0.11.0

--------------------------------------

📁 Project Structure

NLP/
├── [README.md](http://_vscodecontentref_/1)                 # This file
├── [requirements.txt](http://_vscodecontentref_/2)          # Python dependencies
├── data/
│   ├── raw/                 # Raw scraped data
│   ├── processed/           # Processed data (phases 1-6)
│   └── translated/          # Translated reviews
├── src/
│   ├── [1_scraper.py](http://_vscodecontentref_/3)        # Phase 0: Data collection
│   ├── [2_preprocessing.py](http://_vscodecontentref_/4)   # Phase 1: Text cleaning
│   ├── 3_translation.py     # Optional: Language translation
│   ├── [4_pos_ner.py](http://_vscodecontentref_/5)         # Phase 2: POS tagging & NER
│   ├── [5_sentiment.py](http://_vscodecontentref_/6)       # Phase 3: Sentiment analysis
│   ├── [6_topic_modeling.py](http://_vscodecontentref_/7)  # Phase 4: Topic modeling
│   ├── [7_embeddings.py](http://_vscodecontentref_/8)      # Phase 5: Word embeddings
│   ├── 8_summarization.py   # Phase 6: Summarization
│   └── [9_qa_system.py](http://_vscodecontentref_/9)       # Phase 7: QA system
├── results/
│   ├── [summary_report.txt](http://_vscodecontentref_/10)   # Text summary report
│   └── visualizations/      # Generated plots
└── notebooks/
    └── analysis.ipynb       # Jupyter notebook analysis


🚀 Phase-by-Phase Guide

Phase 0: Web Scraping (Optional)
File: src/1_scraper.py

Collects Amazon product reviews using Selenium and BeautifulSoup.

Features:
Automated pagination handling
Duplicate detection and removal
Persistent session management with cookies
Extracts: reviewer name, rating, review text, date, product varian

#Running

python src/1_scraper.py

#Configuration

PRODUCT_ASIN = "B0DGGSNM3B"  # Change to your product ASIN
START_PAGE = 1
END_PAGE = 38
FORCE_NEW_LOGIN = False

#Output

CSV file: amazon_reviews.csv
Fields: name, rating, review, colour, date, title, review_id