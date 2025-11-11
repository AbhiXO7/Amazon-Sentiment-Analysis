"""
Amazon Reviews Scraper - Fixed pagination issue
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import pickle
import os

# ------------------------------
# Extract Functions
# ------------------------------

def get_name(review):
    """Extract reviewer name"""
    try:
        name = review.find("span", class_="a-profile-name")
        if name:
            return name.text.strip()
        profile = review.find("a", class_="a-profile")
        if profile:
            name = profile.find("span", class_="a-profile-name")
            if name:
                return name.text.strip()
        return ""
    except:
        return ""

def get_rating(review):
    """Extract star rating"""
    try:
        rating = review.find("i", {"data-hook": "review-star-rating"})
        if rating:
            rating_text = rating.find("span", class_="a-icon-alt")
            if rating_text:
                return rating_text.text.strip()
        return ""
    except:
        return ""

def get_review_text(review):
    """Extract review body text"""
    try:
        text = review.find("span", {"data-hook": "review-body"})
        if text:
            inner_span = text.find("span")
            if inner_span:
                return inner_span.text.strip()
            return text.text.strip()
        return ""
    except:
        return ""

def get_colour(review):
    """Extract product colour/variant"""
    try:
        colour = review.find("a", {"data-hook": "format-strip"})
        if colour:
            text = colour.text.strip()
            if "Colour:" in text:
                return text.split("Colour:")[-1].strip()
            return text
        return ""
    except:
        return ""

def get_date(review):
    """Extract review date"""
    try:
        date = review.find("span", {"data-hook": "review-date"})
        if date:
            date_text = date.text.strip()
            if " on " in date_text:
                return date_text.split(" on ")[-1].strip()
            return date_text
        return ""
    except:
        return ""

def get_review_title(review):
    """Extract review title"""
    try:
        title_link = review.find("a", {"data-hook": "review-title"})
        if title_link:
            title_span = title_link.find_all("span")
            if title_span:
                return title_span[-1].text.strip()
        return ""
    except:
        return ""

def get_review_id(review):
    """Extract unique review ID for deduplication"""
    try:
        # Get the review ID from the data-hook attribute or id
        review_id = review.get('id', '')
        if review_id:
            return review_id
        # Alternative: look for review link
        review_link = review.find("a", {"data-hook": "review-title"})
        if review_link and review_link.get('href'):
            return review_link.get('href')
        return ""
    except:
        return ""

# ------------------------------
# Driver Setup
# ------------------------------

def setup_driver_with_profile():
    """Setup driver with persistent profile"""
    options = Options()
    
    user_data_dir = os.path.join(os.getcwd(), "chrome_profile")
    options.add_argument(f"user-data-dir={user_data_dir}")
    
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--start-maximized')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def manual_login_and_save(driver):
    """Open browser for manual login"""
    print("\n🔐 MANUAL LOGIN REQUIRED")
    print("=" * 80)
    print("1. Browser will open Amazon.in")
    print("2. Please LOG IN manually")
    print("3. After login, press ENTER here to continue...")
    print("=" * 80)
    
    driver.get("https://www.amazon.in")
    input("\nPress ENTER after you've logged in: ")
    
    pickle.dump(driver.get_cookies(), open("amazon_cookies.pkl", "wb"))
    print("✅ Cookies saved!\n")

def load_cookies(driver):
    """Load saved cookies"""
    if os.path.exists("amazon_cookies.pkl"):
        driver.get("https://www.amazon.in")
        cookies = pickle.load(open("amazon_cookies.pkl", "rb"))
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except:
                pass
        print("✅ Loaded saved cookies")
        return True
    return False

# ------------------------------
# Main Scraping Function - FIXED
# ------------------------------

def scrape_amazon_reviews(asin, start_page=1, end_page=38, force_login=False):
    """Scrape Amazon reviews with proper pagination"""
    
    data = {
        "name": [],
        "rating": [],
        "review": [],
        "colour": [],
        "date": [],
        "title": [],
        "review_id": []
    }
    
    seen_reviews = set()  # Track unique reviews
    
    print("🚀 Setting up Chrome driver...")
    driver = setup_driver_with_profile()
    
    if force_login or not load_cookies(driver):
        manual_login_and_save(driver)
    
    print(f"\nStarting to scrape reviews for ASIN: {asin}")
    print(f"Pages: {start_page} to {end_page}\n")
    
    try:
        # Start with the first page to get initial reviews
        initial_url = f"https://www.amazon.in/product-reviews/{asin}/?ie=UTF8&reviewerType=all_reviews"
        driver.get(initial_url)
        time.sleep(3)
        
        for page_num in range(start_page, end_page + 1):
            print(f"📄 Page {page_num}...", end=" ", flush=True)
            
            try:
                # Method 1: Use "Next page" button (more reliable)
                if page_num > start_page:
                    try:
                        # Find and click next page button
                        next_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "li.a-last a"))
                        )
                        driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                        time.sleep(1)
                        next_button.click()
                        time.sleep(3)  # Wait for page to load
                    except:
                        # Fallback: Use direct URL navigation
                        print("(using URL nav)", end=" ", flush=True)
                        url = f"https://www.amazon.in/product-reviews/{asin}/?ie=UTF8&reviewerType=all_reviews&pageNumber={page_num}"
                        driver.get(url)
                        time.sleep(3)
                
                # Check for login redirect
                if "ap/signin" in driver.current_url or "ap/cvf" in driver.current_url:
                    print("\n⚠️ Login required. Please login in the browser...")
                    input("Press ENTER after logging in: ")
                    driver.get(driver.current_url)
                
                # Wait for reviews to load
                wait = WebDriverWait(driver, 15)
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-hook='review']")))
                except:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li[data-hook='review']")))
                
                # Scroll to load content
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                time.sleep(2)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                # Parse page
                soup = BeautifulSoup(driver.page_source, "html.parser")
                
                reviews = soup.find_all("div", {"data-hook": "review"})
                if not reviews:
                    reviews = soup.find_all("li", {"data-hook": "review"})
                
                if not reviews:
                    print("⚠️ No reviews found")
                    continue
                
                # Track new reviews on this page
                new_reviews = 0
                duplicate_reviews = 0
                
                for review in reviews:
                    # Get unique identifier
                    review_id = get_review_id(review)
                    review_text = get_review_text(review)
                    
                    # Create unique key (use both ID and text to be safe)
                    unique_key = f"{review_id}_{review_text[:50]}"
                    
                    # Skip if we've seen this review
                    if unique_key in seen_reviews:
                        duplicate_reviews += 1
                        continue
                    
                    seen_reviews.add(unique_key)
                    new_reviews += 1
                    
                    # Add to data
                    data["name"].append(get_name(review))
                    data["rating"].append(get_rating(review))
                    data["review"].append(review_text)
                    data["colour"].append(get_colour(review))
                    data["date"].append(get_date(review))
                    data["title"].append(get_review_title(review))
                    data["review_id"].append(review_id)
                
                print(f"✅ {new_reviews} new reviews", end="")
                if duplicate_reviews > 0:
                    print(f" ({duplicate_reviews} duplicates)", end="")
                print(flush=True)
                
                # Stop if no new reviews found
                if new_reviews == 0:
                    print("\n⚠️ No new reviews found. Stopping.")
                    break
                
                # Delay between pages
                if page_num < end_page:
                    delay = random.uniform(3, 6)
                    time.sleep(delay)
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:80]}")
                continue
    
    finally:
        driver.quit()
        print("\n🔒 Browser closed")
    
    return data

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    
    # Configuration
    PRODUCT_ASIN = "B0DGGSNM3B"
    START_PAGE = 1
    END_PAGE = 38
    FORCE_NEW_LOGIN = False
    
    print("=" * 80)
    print("AMAZON REVIEWS SCRAPER - Fixed Pagination")
    print("=" * 80 + "\n")
    
    # Scrape reviews
    reviews_data = scrape_amazon_reviews(PRODUCT_ASIN, START_PAGE, END_PAGE, FORCE_NEW_LOGIN)
    
    # Create DataFrame
    df = pd.DataFrame(reviews_data)
    print(f"\n📊 Total reviews collected: {len(df)}")
    
    # Clean data
    if len(df) > 0:
        df = df[df['review'].astype(str).str.strip() != '']
    
    print(f"📊 Reviews after cleaning: {len(df)}")
    
    if len(df) > 0:
        # Remove review_id column before saving (internal use only)
        df_export = df.drop('review_id', axis=1)
        
        # Save to CSV
        output_file = "amazon_reviews_ff.csv"
        df_export.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ Data saved to '{output_file}'")
        
        # Display sample
        print("\n" + "=" * 80)
        print("SAMPLE REVIEWS:")
        print("=" * 80)
        pd.set_option('display.max_colwidth', 40)
        pd.set_option('display.width', 120)
        print(df_export[['name', 'rating', 'review', 'date']].head(10))
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        print(f"Total Reviews: {len(df)}")
        print(f"Unique Reviewers: {df['name'].nunique()}")
        if df['rating'].astype(str).str.strip().any():
            print(f"\nRating Distribution:")
            print(df['rating'].value_counts().sort_index())
    else:
        print("\n⚠️ No reviews were scraped")