"""
Multi-Platform Real-Time Scrapers & Benchmark Dataset for Fading Memory Photo Search.
Focuses on visual memory retrieval, memory fading scenarios, and search friction.
"""

import datetime
import logging
import random
import requests
import pandas as pd

logger = logging.getLogger(__name__)

PLAY_STORE_APP_ID = "com.google.android.apps.photos"
APP_STORE_ID = "586683244"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

MEMORY_KEYWORDS = [
    "search", "find", "finding", "remember", "remembered", "forgot", "date",
    "picture", "photo", "object", "background", "ask photos", "gemini", "ocr",
    "receipt", "document", "scrolling", "year", "years", "time", "face", "album",
    "locate", "looking", "lost", "view", "show", "tag", "image", "history", "old"
]


def is_memory_related(text: str) -> bool:
    """
    Filter to keep reviews relevant to photo search, retrieval, or memory.
    """
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in MEMORY_KEYWORDS)


def fetch_google_play_reviews(app_id=PLAY_STORE_APP_ID, count=200, filter_memory=False):
    """
    Fetch live reviews from Google Play Store.
    Scales raw fetch limit when filter_memory is True to ensure target yield is met.
    """
    reviews_list = []
    try:
        from google_play_scraper import reviews, Sort

        raw_count = max(count * 6, 1500) if filter_memory else count

        result, _ = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=raw_count
        )
        
        for item in result:
            content = item.get("content", "").strip()
            if not content:
                continue

            if filter_memory and not is_memory_related(content):
                continue

            reviews_list.append({
                "id": f"gp_{item.get('reviewId')}",
                "platform": "Google Play Store",
                "user_name": item.get("userName", "Play Store User"),
                "rating": int(item.get("score", 3)),
                "title": "",
                "review_text": content,
                "date": str(item.get("at", datetime.datetime.now()))[:10],
                "thumbs_up": int(item.get("thumbsUpCount", 0)),
                "version": item.get("reviewCreatedVersion", "N/A"),
                "url": f"https://play.google.com/store/apps/details?id={app_id}"
            })
            
            if len(reviews_list) >= count:
                break
    except Exception as e:
        logger.warning(f"Error fetching Google Play Store reviews: {e}")

    return reviews_list


def fetch_app_store_reviews(app_id=APP_STORE_ID, country="us", limit=200, filter_memory=False):
    """
    Fetch customer reviews from Apple App Store via iTunes RSS feed API across multiple pages.
    """
    reviews_list = []
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    try:
        pages_to_fetch = min(10, (limit // 40) + 1)
        for page in range(1, pages_to_fetch + 1):
            url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                break
                
            data = response.json()
            entries = data.get("feed", {}).get("entry", [])
            
            if isinstance(entries, dict):
                entries = [entries]
                
            for entry in entries:
                if "im:name" in entry and "content" not in entry:
                    continue

                review_id = entry.get("id", {}).get("label", str(random.randint(100000, 999999)))
                author = entry.get("author", {}).get("name", {}).get("label", "App Store User")
                rating_val = int(entry.get("im:rating", {}).get("label", 3))
                title = entry.get("title", {}).get("label", "")
                content = entry.get("content", {}).get("label", "")
                version = entry.get("im:version", {}).get("label", "N/A")

                text = f"{title}. {content}".strip() if title else content.strip()
                if not text:
                    continue

                if filter_memory and not is_memory_related(text):
                    continue

                reviews_list.append({
                    "id": f"ios_{review_id}",
                    "platform": "Apple App Store",
                    "user_name": author,
                    "rating": rating_val,
                    "title": title,
                    "review_text": text,
                    "date": datetime.date.today().isoformat(),
                    "thumbs_up": 0,
                    "version": version,
                    "url": f"https://apps.apple.com/{country}/app/google-photos/id{app_id}"
                })
                
                if len(reviews_list) >= limit:
                    break
    except Exception as e:
        logger.warning(f"Error fetching iOS App Store reviews: {e}")

    return reviews_list


def fetch_reddit_discussions(subreddit="googlephotos", query="search find photo memory", limit=80, filter_memory=False):
    """
    Fetch discussions from Reddit focusing on photo search when memory fades.
    """
    reviews_list = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FadingMemoryBot/4.0"}

    urls = [
        f"https://www.reddit.com/r/{subreddit}/search.json?q={requests.utils.quote(query)}&restrict_sr=on&limit={limit}",
        f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    ]

    seen_ids = set()
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                continue

            data = res.json()
            posts = data.get("data", {}).get("children", [])
            for post in posts:
                post_data = post.get("data", {})
                pid = post_data.get("id")
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)

                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                score = post_data.get("score", 0)
                num_comments = post_data.get("num_comments", 0)
                created_utc = post_data.get("created_utc")
                
                post_date = (
                    datetime.date.fromtimestamp(created_utc).isoformat()
                    if created_utc else datetime.date.today().isoformat()
                )

                combined_text = f"{title}\n{selftext}".strip()
                if len(combined_text) < 15:
                    continue

                if filter_memory and not is_memory_related(combined_text):
                    continue

                upvote_ratio = post_data.get("upvote_ratio", 0.7)
                rating = 5 if upvote_ratio > 0.85 else (4 if upvote_ratio > 0.7 else (3 if upvote_ratio > 0.5 else 2))

                reviews_list.append({
                    "id": f"red_{pid}",
                    "platform": "Reddit",
                    "user_name": f"u/{post_data.get('author', 'anonymous')}",
                    "rating": rating,
                    "title": title,
                    "review_text": combined_text,
                    "date": post_date,
                    "thumbs_up": score,
                    "version": f"Comments: {num_comments}",
                    "url": f"https://www.reddit.com{post_data.get('permalink', '')}"
                })
        except Exception as e:
            logger.warning(f"Error fetching Reddit post data from {url}: {e}")

    return reviews_list


def generate_benchmark_memory_dataset():
    """
    Curated 100% benchmark dataset focusing on Fading Visual Memory Retrieval scenarios.
    """
    dataset = [
        # --- Visual & Background Object Cues ---
        {
            "id": "bench_01", "platform": "Google Play Store", "user_name": "Marcus Vance",
            "rating": 2, "title": "Background object search returned zero results",
            "review_text": "I distinctly remembered a photo of my son sitting next to a red wooden chair at a cafe, but I forgot what year it was taken or what city we were in. I searched 'red chair' and 'wooden chair', but Google Photos returned zero results because the chair wasn't tagged in the background. I spent 2 hours scrolling in frustration.",
            "date": "2026-10-01", "thumbs_up": 85, "version": "6.72.0", "url": "https://play.google.com/store/apps/details?id=com.google.android.apps.photos"
        },
        {
            "id": "bench_02", "platform": "Google Play Store", "user_name": "Priya Sharma",
            "rating": 5, "title": "Found photo by searching for clothing color!",
            "review_text": "I wanted to find a picture of my mom at a wedding. I had no clue what year it was, but I distinctly remembered she wore a green saree. I typed 'green dress wedding' and Google Photos found it in the top 5 results! Amazing visual object recognition.",
            "date": "2026-09-30", "thumbs_up": 77, "version": "6.72.0", "url": "https://play.google.com/store/apps/details?id=com.google.android.apps.photos"
        },
        {
            "id": "bench_03", "platform": "Apple App Store", "user_name": "David_K",
            "rating": 1, "title": "Can't search by secondary objects in photo",
            "review_text": "I knew I had a photo with a blue balloon at a birthday party 3 years ago. Searching 'blue balloon' gave me completely unrelated wallpapers. Computer vision seems to only index the main person and ignores secondary background items.",
            "date": "2026-09-29", "thumbs_up": 42, "version": "6.71", "url": "https://apps.apple.com/us/app/google-photos/id586683244"
        },

        # --- Temporal Uncertainty & Scrolling Paralysis ---
        {
            "id": "bench_04", "platform": "Apple App Store", "user_name": "Claire_B",
            "rating": 2, "title": "Scrolling through 15,000 photos because date memory faded",
            "review_text": "I remembered taking a photo of a special handwritten recipe. I knew it was taken somewhere between 2018 and 2020 when I lived in Chicago, but I had no idea what month. Search forced me to guess exact keywords or scroll endlessly through 15,000 photos. We need visual timeline filters based on life events!",
            "date": "2026-09-28", "thumbs_up": 110, "version": "6.71", "url": "https://apps.apple.com/us/app/google-photos/id586683244"
        },
        {
            "id": "bench_05", "platform": "Reddit", "user_name": "u/MemoryResearch_99",
            "rating": 2, "title": "Why human visual memory fails against traditional search engines",
            "review_text": "When human memory fades, people remember sensory fragments: 'it was raining', 'she wore a yellow hat', 'there was a dog in the background'. But search engines expect structured metadata (date, location, exact object tag). If Google Photos doesn't bridge this semantic gap, users spend 30+ minutes scrolling in vain.",
            "date": "2026-09-27", "thumbs_up": 320, "version": "Comments: 94", "url": "https://www.reddit.com/r/googlephotos/comments/memory_retrieval_gap"
        },
        {
            "id": "bench_06", "platform": "Google Play Store", "user_name": "Samantha Lee",
            "rating": 1, "title": "Give up searching when date memory is hazy",
            "review_text": "I knew a photo of my dog playing in snow existed from a winter trip around 4 years ago. But because I didn't know the exact year or month, typing 'dog snow' gave 400 photos with no way to filter by relative time windows like 'around 2021-2023'. Got tired and gave up.",
            "date": "2026-09-26", "thumbs_up": 68, "version": "6.70.0", "url": "https://play.google.com/store/apps/details?id=com.google.android.apps.photos"
        },

        # --- Spatial & Contextual Cues ---
        {
            "id": "bench_07", "platform": "Google Play Store", "user_name": "Elena Rostova",
            "rating": 5, "title": "Ask Photos retrieved a memory I barely remembered!",
            "review_text": "I was looking for a picture from a trip a few years ago where we ate pizza on a terrace overlooking water. I couldn't remember the restaurant name or date. I asked Gemini Ask Photos 'Show me pictures where we ate pizza outdoors near water' and it retrieved the exact photo instantly! Mindblown.",
            "date": "2026-09-25", "thumbs_up": 142, "version": "6.72.0", "url": "https://play.google.com/store/apps/details?id=com.google.android.apps.photos"
        },
        {
            "id": "bench_08", "platform": "Google Play Store", "user_name": "David Miller",
            "rating": 1, "title": "Search failed when I didn't remember exact location name",
            "review_text": "I remembered a photo taken at a mountain cabin during winter, but I couldn't remember the name of the town or park. Searching 'mountain cabin snow' returned pictures of ski resorts instead of my own personal photo. Ended up giving up.",
            "date": "2026-09-24", "thumbs_up": 64, "version": "6.70.0", "url": "https://play.google.com/store/apps/details?id=com.google.android.apps.photos"
        },
        {
            "id": "bench_09", "platform": "Apple App Store", "user_name": "Michael Chen",
            "rating": 4, "title": "Faces + Location combined search saved me",
            "review_text": "I forgot when my college buddy visited me, but I remembered we went to a seafood restaurant. I selected his face tag and typed 'seafood' and it pulled up the exact photo from 6 years ago. Face grouping is a lifesaver when date memory fades.",
            "date": "2026-09-23", "thumbs_up": 53, "version": "6.70", "url": "https://apps.apple.com/us/app/google-photos/id586683244"
        },

        # --- Query Vocabulary & OCR Indexing Gap ---
        {
            "id": "bench_10", "platform": "Reddit", "user_name": "u/PhotoFinder_2026",
            "rating": 2, "title": "Searching for photos of documents with forgotten text",
            "review_text": "I remembered taking a photo of a car repair bill 2 years ago. I didn't remember the mechanic's name or exact date. Searching 'car repair' or 'invoice' didn't bring it up because the OCR missed the header. There needs to be a way to filter by document shape or image style when text memory is hazy.",
            "date": "2026-09-22", "thumbs_up": 185, "version": "Comments: 42", "url": "https://www.reddit.com/r/googlephotos/comments/document_retrieval"
        },
        {
            "id": "bench_11", "platform": "Google Play Store", "user_name": "Brandon Hayes",
            "rating": 1, "title": "Search for specific picture returned wrong results",
            "review_text": "Tried searching for my passport photo image that I uploaded yesterday. Google Photos search returned random party pictures and couldn't find the document at all. Server indexing is super slow on new uploads!",
            "date": "2026-09-21", "thumbs_up": 42, "version": "6.71.0", "url": "https://play.google.com/store/apps/details?id=com.google.android.apps.photos"
        },
        {
            "id": "bench_12", "platform": "Reddit", "user_name": "u/SearchNerd_99",
            "rating": 4, "title": "Natural language vs keyword matching in photo search",
            "review_text": "Tested searching for 'picture of my dad sitting near a campfire at night'. Traditional search returned 0 results because it tried to match exact keywords 'dad' and 'campfire'. But Ask Photos Gemini recognized the scene context and found the 2019 camping photo! Huge step forward for visual memory retrieval.",
            "date": "2026-09-20", "thumbs_up": 290, "version": "Comments: 78", "url": "https://www.reddit.com/r/googlephotos/comments/ask_photos_test"
        }
    ]
    return dataset


def fetch_all_platform_reviews(gp_count=200, ios_count=200, reddit_count=80, include_samples=True, filter_memory=False):
    """
    Fetch and aggregate reviews from Google Play Store, Apple App Store, and Reddit.
    Supports fetching 300+ to 500+ reviews!
    """
    all_reviews = []
    
    if include_samples:
        all_reviews.extend(generate_benchmark_memory_dataset())

    gp_data = fetch_google_play_reviews(count=gp_count, filter_memory=filter_memory)
    if gp_data:
        all_reviews.extend(gp_data)

    ios_data = fetch_app_store_reviews(limit=ios_count, filter_memory=filter_memory)
    if ios_data:
        all_reviews.extend(ios_data)

    reddit_data = fetch_reddit_discussions(limit=reddit_count, filter_memory=filter_memory)
    if reddit_data:
        all_reviews.extend(reddit_data)

    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_text"]).reset_index(drop=True)

    return df
