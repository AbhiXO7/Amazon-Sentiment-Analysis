"""
Fixed NLTK Data Downloader
Handles SSL and network issues
"""

import nltk
import ssl
import os

# Fix SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download essential NLTK packages with error handling"""
    
    # Essential packages for this project
    packages = [
        'punkt',              # Tokenization
        'stopwords',          # Stop words
        'wordnet',            # Lemmatization
        'averaged_perceptron_tagger',  # POS tagging
        'maxent_ne_chunker',  # NER
        'words',              # Word lists
        'omw-1.4',            # Open Multilingual Wordnet
        'vader_lexicon',      # Sentiment analysis
        'brown',              # Brown corpus
        'movie_reviews',      # Movie reviews corpus
    ]
    
    print("=" * 80)
    print("DOWNLOADING NLTK DATA")
    print("=" * 80)
    print(f"Total packages to download: {len(packages)}\n")
    
    successful = []
    failed = []
    
    for i, package in enumerate(packages, 1):
        try:
            print(f"[{i}/{len(packages)}] Downloading '{package}'...", end=" ")
            nltk.download(package, quiet=True)
            successful.append(package)
            print("✅")
        except Exception as e:
            print(f"❌ Failed: {str(e)[:50]}")
            failed.append(package)
    
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(successful)}/{len(packages)}")
    print(f"❌ Failed: {len(failed)}/{len(packages)}")
    
    if failed:
        print(f"\nFailed packages: {', '.join(failed)}")
        print("\n💡 Alternative: Download manually from NLTK GUI")
    
    return len(failed) == 0

def open_nltk_gui():
    """Open NLTK download GUI as fallback"""
    print("\n" + "=" * 80)
    print("OPENING NLTK DOWNLOAD GUI")
    print("=" * 80)
    print("Instructions:")
    print("1. A window will open")
    print("2. Go to 'All Packages' tab")
    print("3. Click 'Download' button at the bottom")
    print("4. Wait for all downloads to complete")
    print("5. Close the window when done")
    print("=" * 80 + "\n")
    
    try:
        nltk.download()  # Opens GUI
    except Exception as e:
        print(f"❌ Could not open GUI: {e}")

if __name__ == "__main__":
    print("\n🚀 Starting NLTK setup...\n")
    
    # Try automatic download first
    success = download_nltk_data()
    
    if not success:
        print("\n⚠️ Some downloads failed. Trying alternative method...")
        user_input = input("\nOpen NLTK GUI for manual download? (y/n): ")
        
        if user_input.lower() == 'y':
            open_nltk_gui()
    
    # Verify installation
    print("\n" + "=" * 80)
    print("VERIFYING INSTALLATION")
    print("=" * 80)
    
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        # Test
        test_text = "Testing NLTK installation successfully"
        tokens = word_tokenize(test_text)
        print(f"✅ Tokenization works: {tokens}")
        
        stop_words = set(stopwords.words('english'))
        print(f"✅ Stop words loaded: {len(stop_words)} words")
        
        lemmatizer = WordNetLemmatizer()
        print(f"✅ Lemmatizer works: running -> {lemmatizer.lemmatize('running', pos='v')}")
        
        print("\n🎉 NLTK setup completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        print("\n💡 Try running: nltk.download() in Python shell")