Copy# CLAUDE.md — MarketPulse: Data-Centric Sentiment Intelligence for Financial Social Media

## PROJECT OVERVIEW

MarketPulse is a production-grade, data-centric ML pipeline that ingests real financial social media data, applies programmatic labeling through weak supervision, trains classical ML models, extracts ticker entities, and surfaces everything through an interactive Streamlit dashboard.

This is not a proof of concept. This is a working product that processes live data, learns from it, and improves through systematic data quality iteration.

### Core Thesis
> "A logistic regression trained on high-quality, programmatically labeled data is a production-ready system. The data is the model."

### What This Product Does
1. **Ingests real financial social media data** from Reddit (WallStreetBets), Stocktwits, and financial news APIs with configurable date ranges
2. **Falls back to synthetic data** when no API keys are configured, so the full pipeline runs regardless
3. **Applies programmatic labeling** through 15+ labeling functions encoding financial domain heuristics
4. **Assesses label quality** with coverage, conflict, confidence, and agreement metrics
5. **Trains production ML models** (TF-IDF + Logistic Regression) on programmatically labeled data
6. **Extracts ticker entities** with rule-based extraction and normalization
7. **Evaluates everything** with comprehensive metrics, error analysis, and data quality diagnostics
8. **Surfaces the full pipeline** through an interactive Streamlit dashboard with real-time inference

---

## ARCHITECTURE
marketpulse/
│
├── CLAUDE.md
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── Makefile                            # CLI commands for the full pipeline
│
├── config/
│   ├── default.yaml                    # Default configuration
│   └── sources.yaml                    # Data source configuration
│
├── data/
│   ├── raw/                            # Raw ingested data (gitignored)
│   ├── labeled/                        # Programmatically labeled data
│   ├── gold/                           # Hand-labeled evaluation set
│   │   └── gold_standard.csv           # 100 labeled posts for eval
│   ├── synthetic/                      # Synthetic fallback dataset
│   │   └── synthetic_posts.csv         # Pre-generated synthetic data
│   └── models/                         # Trained model artifacts
│       ├── tfidf_vectorizer.pkl
│       ├── sentiment_model.pkl
│       └── model_metadata.json
│
├── src/
│   ├── init.py
│   │
│   ├── ingestion/
│   │   ├── init.py
│   │   ├── base.py                     # Abstract base ingestion class
│   │   ├── reddit.py                   # Reddit/WSB ingestion via PRAW
│   │   ├── stocktwits.py              # Stocktwits API ingestion
│   │   ├── news.py                     # Financial news API ingestion
│   │   ├── synthetic.py                # Synthetic data generator
│   │   └── manager.py                  # Orchestrates all sources with fallback
│   │
│   ├── labeling/
│   │   ├── init.py
│   │   ├── functions.py                # All labeling functions
│   │   ├── aggregator.py              # Vote aggregation strategies
│   │   └── quality.py                 # Label quality assessment
│   │
│   ├── models/
│   │   ├── init.py
│   │   ├── pipeline.py                 # Training and inference pipeline
│   │   └── versioning.py              # Model saving, loading, versioning
│   │
│   ├── extraction/
│   │   ├── init.py
│   │   ├── ticker_extractor.py         # Rule-based ticker/company extraction
│   │   └── normalizer.py              # Entity normalization
│   │
│   ├── evaluation/
│   │   ├── init.py
│   │   ├── classification.py           # Sentiment classification eval
│   │   ├── extraction.py              # Entity extraction eval
│   │   └── label_quality.py           # Programmatic vs gold comparison
│   │
│   └── utils/
│       ├── init.py
│       ├── config.py                   # Configuration loader
│       ├── logger.py                   # Structured logging
│       └── cache.py                    # Caching layer for API responses
│
├── app/
│   ├── streamlit_app.py                # Main dashboard entry point
│   ├── pages/
│   │   ├── 1_data_ingestion.py         # Live data ingestion + exploration
│   │   ├── 2_labeling_studio.py        # Labeling function visualization
│   │   ├── 3_label_quality.py          # Label quality assessment
│   │   ├── 4_model_training.py         # Model training + feature analysis
│   │   ├── 5_entity_extraction.py      # Ticker extraction + eval
│   │   ├── 6_evaluation.py             # Comprehensive evaluation
│   │   └── 7_live_inference.py         # Real-time classification of new posts
│   └── components/
│       ├── charts.py                   # Reusable Plotly components
│       ├── metrics.py                  # Metric display cards
│       └── styles.py                   # Theme and CSS
│
├── scripts/
│   ├── ingest.py                       # CLI: python scripts/ingest.py --source reddit --days 7
│   ├── label.py                        # CLI: python scripts/label.py
│   ├── train.py                        # CLI: python scripts/train.py
│   ├── evaluate.py                     # CLI: python scripts/evaluate.py
│   └── run_pipeline.py                 # CLI: runs full pipeline end to end
│
└── tests/
├── test_ingestion.py
├── test_labeling_functions.py
├── test_aggregator.py
├── test_pipeline.py
├── test_extraction.py
├── test_normalizer.py
└── test_evaluation.py
Copy
---

## CONFIGURATION

### File: `config/default.yaml`

```yaml
project:
  name: "MarketPulse"
  version: "1.0.0"

data:
  mode: "auto"  # "live" | "synthetic" | "auto"
  # auto = try live sources first, fall back to synthetic if no API keys
  
  storage:
    raw_dir: "data/raw"
    labeled_dir: "data/labeled"
    gold_dir: "data/gold"
    model_dir: "data/models"
  
  synthetic:
    num_posts: 500
    seed: 42

ingestion:
  reddit:
    subreddits: ["wallstreetbets", "stocks", "investing"]
    post_limit_per_sub: 200
    include_comments: false
    min_score: 5  # filter low-quality posts
  
  stocktwits:
    symbols: ["AAPL", "TSLA", "NVDA", "GME", "AMC", "SPY", "MSFT", "AMZN"]
    limit_per_symbol: 50
  
  news:
    query_terms: ["stock market", "earnings", "IPO", "SEC", "Fed"]
    language: "en"
    page_size: 100
  
  date_range:
    start: null  # null = 7 days ago
    end: null    # null = today
    default_lookback_days: 7

labeling:
  aggregation_strategy: "confidence_weighted"  # "majority" | "weighted" | "confidence_weighted"
  confidence_threshold: 0.6
  min_votes: 2  # post needs at least 2 function votes to get a label

model:
  max_features: 500
  ngram_range: [1, 2]
  min_df: 3
  C: 1.0
  class_weight: "balanced"
  test_size: 0.2
  random_state: 42

evaluation:
  gold_set_size: 100

dashboard:
  theme: "dark"
  port: 8501
File: .env.example
bashCopy# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=MarketPulse/1.0

# Stocktwits (https://api.stocktwits.com/developers)
STOCKTWITS_ACCESS_TOKEN=your_token

# News API (https://newsapi.org/)
NEWS_API_KEY=your_key

# Optional: Alpha Vantage for price data context
ALPHA_VANTAGE_KEY=your_key

DATA INGESTION SYSTEM
File: src/ingestion/base.py
pythonCopyfrom abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

class BaseIngester(ABC):
    """
    Abstract base class for all data sources.
    Every ingester must return data in the same standardized schema.
    """
    
    REQUIRED_COLUMNS = [
        'post_id',        # str: unique identifier (source_prefix + original_id)
        'text',           # str: raw post text
        'source',         # str: 'reddit' | 'stocktwits' | 'news' | 'synthetic'
        'timestamp',      # datetime: when the post was created
        'author',         # str: username or 'unknown'
        'score',          # int: upvotes/likes/engagement metric
        'url',            # str: link to original post or empty string
        'metadata',       # dict: source-specific metadata (subreddit, symbol, etc.)
    ]
    
    @abstractmethod
    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Ingest data for the given date range.
        Must return DataFrame matching REQUIRED_COLUMNS schema.
        Must handle API rate limits gracefully.
        Must log ingestion progress.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this source is configured and reachable.
        Returns False if API keys are missing or invalid.
        """
        pass
    
    def validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that output matches expected schema.
        Drop rows with null text. Deduplicate by post_id.
        Log any data quality issues found.
        """
        pass
File: src/ingestion/reddit.py
pythonCopyimport praw
from datetime import datetime
import pandas as pd
from .base import BaseIngester

class RedditIngester(BaseIngester):
    """
    Ingest posts from Reddit using PRAW (Python Reddit API Wrapper).
    
    Targets subreddits: wallstreetbets, stocks, investing
    
    Implementation details:
    - Use praw.Reddit() with credentials from .env
    - Fetch posts using subreddit.new() and subreddit.hot() within date range
    - Extract: title + selftext as the text field
    - Filter posts with score < min_score to remove low-quality content
    - Handle rate limits with exponential backoff
    - Cache responses to avoid redundant API calls
    - post_id format: "reddit_{subreddit}_{original_id}"
    
    Metadata to capture:
    - subreddit: str
    - num_comments: int
    - flair: str (posts often have flairs like "DD", "YOLO", "Discussion")
    - is_self: bool (text post vs link post)
    - link_flair_text: str
    """
    
    def __init__(self, config):
        pass
    
    def is_available(self) -> bool:
        """Check if Reddit credentials exist and are valid."""
        pass
    
    def ingest(self, start_date, end_date) -> pd.DataFrame:
        pass
File: src/ingestion/stocktwits.py
pythonCopyimport requests
from .base import BaseIngester

class StocktwitsIngester(BaseIngester):
    """
    Ingest messages from Stocktwits API.
    
    Stocktwits is valuable because:
    - Posts are already associated with ticker symbols
    - Some posts have user-submitted sentiment tags (bullish/bearish)
    - These user tags can be used as additional weak supervision signal
    
    API endpoint: https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json
    
    Implementation details:
    - Fetch streams for configured symbols
    - Extract message body as text
    - Capture user-submitted sentiment tag if present (in metadata)
    - Handle pagination for date range filtering
    - post_id format: "stocktwits_{message_id}"
    
    Metadata to capture:
    - symbols: list of associated ticker symbols
    - user_sentiment: str or null (user-tagged bullish/bearish)
    - reshares: int
    - likes: int
    
    Note: Stocktwits API may not require authentication for basic access.
    Check current API docs for rate limits.
    """
    
    def __init__(self, config):
        pass
    
    def is_available(self) -> bool:
        pass
    
    def ingest(self, start_date, end_date) -> pd.DataFrame:
        pass
File: src/ingestion/news.py
pythonCopyimport requests
from .base import BaseIngester

class NewsIngester(BaseIngester):
    """
    Ingest financial news headlines and summaries.
    
    Primary source: NewsAPI (https://newsapi.org/)
    Fallback: Finnhub news API if NewsAPI unavailable
    
    Why news matters:
    - News posts are almost always NEUTRAL (factual reporting)
    - They provide good training signal for the neutral class
    - News about earnings, mergers, regulation gives entity extraction data
    - Mixing news with social media posts creates a realistic class distribution
    
    API endpoint: https://newsapi.org/v2/everything
    Parameters: q=query, from=date, to=date, language=en, sortBy=publishedAt
    
    Implementation details:
    - Query financial keywords from config
    - Use article title + description as text (not full article body)
    - Filter to English only
    - Deduplicate by headline similarity (news gets syndicated)
    - post_id format: "news_{source}_{hash_of_url}"
    
    Metadata to capture:
    - news_source: str (CNBC, Reuters, Bloomberg, etc.)
    - article_url: str
    - image_url: str or null
    - published_at: datetime
    """
    
    def __init__(self, config):
        pass
    
    def is_available(self) -> bool:
        pass
    
    def ingest(self, start_date, end_date) -> pd.DataFrame:
        pass
File: src/ingestion/synthetic.py
pythonCopyimport pandas as pd
import numpy as np
from .base import BaseIngester

class SyntheticIngester(BaseIngester):
    """
    Generate realistic synthetic financial social media data.
    Used as fallback when no API keys are configured.
    
    THIS IS THE FALLBACK, NOT THE PRIMARY SOURCE.
    The synthetic data should be pre-generated and committed 
    to the repo so the project works out of the box.
    
    Also used for:
    - Unit testing
    - Demo mode for the dashboard
    - Consistent evaluation baseline
    
    Generation requirements:
    - 500 posts minimum
    - Distribution: 30% bullish, 20% bearish, 25% neutral, 25% meme
    - Must include all text patterns listed in DATA SPECIFICATION section
    - Must include 50+ intentionally ambiguous edge cases
    - Must include realistic timestamps (spread across configured date range)
    - Must include realistic score distributions (most posts low engagement, 
      few viral posts)
    - Source field set to 'synthetic'
    
    The synthetic dataset file should be committed to git at:
    data/synthetic/synthetic_posts.csv
    
    Also generate and commit:
    data/gold/gold_standard.csv (100 hand-labeled posts from synthetic set)
    """
    
    def __init__(self, config):
        pass
    
    def is_available(self) -> bool:
        """Always returns True — synthetic data is always available."""
        return True
    
    def ingest(self, start_date, end_date) -> pd.DataFrame:
        """Load pre-generated synthetic data."""
        pass
    
    def generate(self, num_posts=500, seed=42):
        """
        Generate fresh synthetic data and save to disk.
        Called during setup or when user wants to regenerate.
        """
        pass
File: src/ingestion/manager.py
pythonCopyfrom datetime import datetime, timedelta
import pandas as pd
from .reddit import RedditIngester
from .stocktwits import StocktwitsIngester
from .news import NewsIngester
from .synthetic import SyntheticIngester

class IngestionManager:
    """
    Orchestrates data ingestion across all sources.
    
    Behavior by mode:
    
    mode="live":
      - Attempts all configured live sources
      - Raises error if no API keys configured
    
    mode="synthetic":
      - Only uses synthetic data
      - No API keys needed
    
    mode="auto" (default):
      - Checks which sources are available
      - Ingests from all available live sources
      - Falls back to synthetic if no live sources available
      - Logs which sources were used
      - Dashboard shows data source composition
    
    The manager also handles:
    - Deduplication across sources (same post might appear on Reddit and be 
      quoted on Stocktwits)
    - Schema validation for all ingested data
    - Caching: don't re-fetch data for the same date range
    - Combining multiple sources into a single unified DataFrame
    - Logging source-level statistics (how many posts per source)
    """
    
    def __init__(self, config):
        self.config = config
        self.sources = []
        self.synthetic = SyntheticIngester(config)
        
        # Register live sources
        self.reddit = RedditIngester(config)
        self.stocktwits = StocktwitsIngester(config)
        self.news = NewsIngester(config)
    
    def ingest(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Main ingestion entry point.
        
        Args:
            start_date: datetime or None (defaults to N days ago from config)
            end_date: datetime or None (defaults to now)
        
        Returns:
            Unified DataFrame with data from all available sources.
            Includes 'source' column indicating where each post came from.
        
        Behavior:
            1. Resolve date range from args or config defaults
            2. Check which sources are available
            3. Ingest from each available source
            4. Combine, deduplicate, validate
            5. Cache results
            6. Log summary statistics
            7. Fall back to synthetic if no live data retrieved
        """
        pass
    
    def get_source_summary(self) -> dict:
        """
        Return dict with ingestion stats:
        {
            'total_posts': int,
            'sources_used': ['reddit', 'stocktwits'],
            'sources_unavailable': ['news'],
            'date_range': {'start': datetime, 'end': datetime},
            'posts_per_source': {'reddit': 150, 'stocktwits': 80},
            'mode': 'auto',
            'used_fallback': False
        }
        """
        pass

SYNTHETIC DATA SPECIFICATION
Generate posts that feel authentic to each source and sentiment class.
Sentiment Categories
CopyBULLISH (30%):
  Posts expressing genuine positive market sentiment.
  The author believes a stock/market will go up and may be 
  taking or recommending a long position.
  
  Subcategories to include:
  - Conviction buys: "Just loaded 500 shares of NVDA. This is going to $200."
  - Technical analysis bulls: "AAPL breaking out of the wedge pattern. Bullish."
  - Earnings optimism: "MSFT earnings going to crush. Azure growth is insane."
  - Dip buyers: "Everyone panicking about TSLA but I'm buying every dip."
  - Options bulls: "Loaded up on SPY 500c for December. Free money."
  - Subtle bulls: "Added to my GOOG position. Long term this is a no brainer."

BEARISH (20%):
  Posts expressing genuine negative market sentiment.
  The author believes a stock/market will go down.
  
  Subcategories:
  - Short sellers: "Shorting TSLA at these levels. P/E is insane."
  - Put buyers: "Loaded puts on SPY. This rally is fake."
  - Warning posts: "Get out of AMC now. The squeeze is done."
  - Fundamental bears: "NVDA at 65x P/E. This ends badly for everyone."
  - Macro bears: "Fed isn't cutting rates. Markets will tank."
  - Subtle bears: "Taking profits on everything. Something feels off."

NEUTRAL (25%):
  Factual, informational, analytical without clear directional bias.
  
  Subcategories:
  - Questions: "What's everyone thinking about AAPL earnings Thursday?"
  - News sharing: "TSLA Q3 deliveries came in at 435K units."
  - Analysis: "Looking at MSFT's balance sheet. Here's what I found."
  - Educational: "For those asking about IV crush, here's how it works."
  - Discussion starters: "Anyone else watching the Fed meeting tomorrow?"
  - Comparative: "NVDA vs AMD for AI exposure. Thoughts?"

MEME (25%):
  Hype-driven, joke-heavy, ironic, or culturally referential content 
  that does NOT carry reliable directional signal.
  
  Subcategories:
  - Loss porn: "Down 90% on GME weeklies. See you behind Wendy's 🍔"
  - Self-deprecating: "My wife's boyfriend picks better stocks than me"
  - Hype without substance: "APES TOGETHER STRONG 🦍💎🙌 GME TO THE MOON"
  - Ironic: "Buy high sell low, this is the way"
  - Cultural: "Sir this is a casino"
  - Sarcastic: "oh yeah WISH is definitely going to $100 🤡🤡🤡"
Required Edge Cases (Generate at Least 50)
Copy1. SARCASTIC BULLISH → Actually BEARISH or MEME:
   "sure this time the short squeeze will definitely happen 🙄"
   "WISH going to $100 EOY trust the process 🤡"

2. MEME WITH REAL CONVICTION:
   "I know this sounds like a meme but I genuinely believe GME 
    is transforming into a tech company. Not just apes. Real DD."

3. BEARISH WORDS, BULLISH POSITION:
   "Everyone says PLTR is garbage but I just doubled my position. 
    Contrarian play."

4. QUESTION WITH EMBEDDED SENTIMENT:
   "Is anyone else worried NVDA is in a bubble at these prices?"
   "Why would anyone sell AAPL before earnings? Seems obvious."

5. NEWS MIXED WITH OPINION:
   "TSLA beat deliveries by 10%. This stock is going to $400 🚀"

6. LOSS MENTION — MEME OR BEARISH?:
   "Down 80% on my calls. At least I still have my cardboard box 📦"
   "Down 80% on my calls. Getting out of this position ASAP."

7. CONTRADICTORY:
   "I think TSLA is overvalued but I keep buying every dip lol"

8. EXTREMELY SHORT AND AMBIGUOUS:
   "TSLA 🚀"    — bullish or meme?
   "GUH"        — meme or bearish?
   "RIP"        — bearish or meme?
   "moon soon"  — bullish or meme?

9. NEUTRAL ANALYSIS THAT SOUNDS BEARISH:
   "NVDA P/E is 65x. Historical semi average is 20x. 
    Make your own conclusions."

10. MULTI-TICKER WITH MIXED SENTIMENT:
    "Selling AAPL to buy NVDA. One's dead money, other's the future."
Text Patterns Every Post Category Must Contain
CopyACROSS ALL CATEGORIES:
  □ Ticker symbols ($TSLA, $AAPL, etc.) — at least 70% of posts
  □ Emojis — at least 40% of posts
  □ ALL CAPS words — at least 30% of posts
  □ Informal language (ngl, imo, tbh, lol) — at least 25% of posts
  □ Multiple tickers in one post — at least 15% of posts
  □ Question marks — concentrated in neutral but present elsewhere
  □ Numbers and prices ($180, 100 shares, 65x P/E) — at least 30%
  □ Options language (calls, puts, IV, theta) — at least 20%
  □ Reddit/WSB slang (DD, YOLO, tendies, apes) — concentrated in meme
  □ Varied post lengths (10 chars to 280 chars)
  □ Some posts with ONLY emojis and tickers ("$TSLA 🚀🚀🚀")
  □ Hashtags and cashtags (#YOLO, $NVDA)
  □ Misspellings and typos ("gunna", "tmrw", "stonks")
Gold Standard: data/gold/gold_standard.csv
Select 100 posts from synthetic data. Must include:

25 per category (balanced for fair eval)
At least 20 ambiguous/edge case posts
Each labeled with:

csvCopypost_id,text,sentiment_gold,tickers_gold,ambiguity_score,notes
1,"...",bullish,"['TSLA','AAPL']",1,"Clear bullish conviction buy"
42,"...",meme,"['GME']",4,"Uses bullish language but is clearly ironic"

ambiguity_score: 1 (crystal clear) to 5 (genuinely ambiguous even for humans)
notes: Brief explanation of labeling decision


PROGRAMMATIC LABELING PIPELINE
File: src/labeling/functions.py
pythonCopy"""
MarketPulse Labeling Functions

Each function encodes a single domain heuristic for financial sentiment.
Functions are intentionally imperfect — that's the point. Individual 
functions are noisy but their combination produces high-quality labels.

Every function must:
  - Accept a single string (post text)
  - Return one of: BULLISH, BEARISH, NEUTRAL, MEME, or ABSTAIN
  - Have a docstring explaining the heuristic and its known limitations
  - Be testable in isolation

Function naming convention: lf_{signal_type}_{what_it_detects}
"""

BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"
MEME = "meme"
ABSTAIN = -1


# ============================================
# KEYWORD-BASED FUNCTIONS
# ============================================

def lf_keyword_bullish(text):
    """
    Detect bullish sentiment via buying/positive keywords.
    
    Heuristic: If post contains words associated with buying or 
    positive price expectations, vote bullish.
    
    Known limitations:
    - Triggers on negated phrases: "NOT buying" still matches "buying"
    - Triggers on sarcastic usage: "definitely going to moon 🤡"
    - Cannot distinguish genuine conviction from wishful thinking
    
    Expected metrics:
    - Coverage: ~30%
    - Precision on gold set: ~70%
    """
    bullish_words = [
        'buy', 'buying', 'bought', 'long', 'calls', 'bullish',
        'loading up', 'accumulating', 'undervalued', 'breakout',
        'buy the dip', 'btd', 'price target', 'upside',
        'going up', 'all in', 'added to my position',
        'free money', 'easy money', 'no brainer'
    ]
    text_lower = text.lower()
    if any(word in text_lower for word in bullish_words):
        return BULLISH
    return ABSTAIN


def lf_keyword_bearish(text):
    """
    Detect bearish sentiment via selling/negative keywords.
    
    Known limitations:
    - "Taking profits" could be neutral portfolio management
    - "Overvalued" in a question context might be neutral
    """
    bearish_words = [
        'sell', 'selling', 'sold', 'short', 'puts', 'bearish',
        'crash', 'dump', 'overvalued', 'bubble', 'top is in',
        'rug pull', 'dead cat', 'exit', 'taking profits',
        'get out', 'going down', 'bagholder', 'bag holder'
    ]
    text_lower = text.lower()
    if any(word in text_lower for word in bearish_words):
        return BEARISH
    return ABSTAIN


def lf_keyword_neutral(text):
    """
    Detect neutral/informational content via question and analysis patterns.
    
    Known limitations:
    - Rhetorical questions expressing sentiment get caught here
    - "Anyone else worried about..." is tagged neutral but has bearish sentiment
    """
    text_lower = text.lower()
    neutral_patterns = [
        'what do you think', 'thoughts on', 'anyone know',
        'how does', 'when is', 'eli5', 'explain',
        'thoughts?', 'opinions?', 'what are',
        'announces', 'reports', 'according to',
        'earnings report', 'quarterly results'
    ]
    if any(pattern in text_lower for pattern in neutral_patterns):
        return NEUTRAL
    return ABSTAIN


def lf_keyword_meme(text):
    """
    Detect WSB meme culture language.
    
    Known limitations:
    - Posts can use meme language AND have genuine sentiment
    - "Diamond hands on AAPL" might be genuine long-term conviction
    """
    meme_words = [
        'apes', 'tendies', "wife's boyfriend", 'wifes boyfriend',
        "wendy's", 'wendys', 'diamond hands', 'paper hands',
        'yolo', 'guh', 'stonks', 'smooth brain',
        'degen', 'casino', 'gambling', 'loss porn',
        'gain porn', 'i just like the stock', 'this is the way',
        'sir this is', 'behind the dumpster', 'food stamps or lambo',
        'to the moon', 'ape together strong', 'hodl'
    ]
    text_lower = text.lower()
    if any(word in text_lower for word in meme_words):
        return MEME
    return ABSTAIN


# ============================================
# EMOJI-BASED FUNCTIONS
# ============================================

def lf_emoji_bullish(text):
    """
    Rocket and moon emojis often indicate bullish hype.
    
    Known limitations:
    - 🚀 is heavily used in meme posts — precision is lower than expected
    - Need other signals to distinguish genuine bullish from meme bullish
    """
    bullish_emojis = ['🚀', '🌙', '📈', '🐂', '💰', '🤑', '⬆️']
    if any(e in text for e in bullish_emojis):
        return BULLISH
    return ABSTAIN


def lf_emoji_bearish(text):
    """Bear and down emojis indicate bearish sentiment."""
    bearish_emojis = ['📉', '🐻', '💀', '🔻', '⬇️', '😱', '🩸']
    if any(e in text for e in bearish_emojis):
        return BEARISH
    return ABSTAIN


def lf_emoji_meme(text):
    """
    Diamond hands, ape, and clown emojis are meme culture markers.
    
    Note: 🤡 is often used to mock others — could indicate bearish 
    sentiment toward the target. But as a labeling function for the 
    POST's overall sentiment, meme is the safer label.
    """
    meme_emojis = ['💎', '🙌', '🦍', '🤡', '🎰', '🍗', '🫠']
    if any(e in text for e in meme_emojis):
        return MEME
    return ABSTAIN


# ============================================
# STRUCTURAL FUNCTIONS
# ============================================

def lf_question_structure(text):
    """
    Posts ending with question marks or containing question patterns.
    
    Known limitations:
    - Rhetorical questions: "Why would anyone sell AAPL?" (actually bullish)
    - Frustrated questions: "When will support actually respond??" (complaint)
    """
    stripped = text.strip()
    if stripped.endswith('?'):
        return NEUTRAL
    if text.count('?') >= 2:
        return NEUTRAL
    return ABSTAIN


def lf_short_post(text):
    """
    Very short posts (under 15 chars) are usually memes or reactions.
    Examples: "GUH", "🚀🚀🚀", "RIP", "moon soon", "bruh"
    
    Known limitations:
    - Short news headlines get caught: "AAPL beats earnings"
    - Short questions: "calls or puts?"
    """
    if len(text.strip()) < 15:
        return MEME
    return ABSTAIN


def lf_all_caps_ratio(text):
    """
    High ratio of ALL CAPS words indicates strong emotion.
    Combined with keyword analysis to determine direction.
    
    Posts like "TSLA IS GOING TO MARS" vs "EVERYONE IS GOING TO LOSE MONEY"
    """
    words = text.split()
    if len(words) < 3:
        return ABSTAIN
    caps_words = [w for w in words if w.isupper() and len(w) > 1 
                  and not w.startswith('$')]  # exclude tickers
    caps_ratio = len(caps_words) / len(words)
    
    if caps_ratio < 0.4:
        return ABSTAIN
    
    text_lower = text.lower()
    bear_signal = any(w in text_lower for w in ['crash', 'sell', 'dump', 'dead', 'worst'])
    bull_signal = any(w in text_lower for w in ['moon', 'buy', 'best', 'love', 'amazing'])
    
    if bear_signal and not bull_signal:
        return BEARISH
    if bull_signal and not bear_signal:
        return BULLISH
    return ABSTAIN


# ============================================
# FINANCIAL PATTERN FUNCTIONS
# ============================================

def lf_options_directional(text):
    """
    Options language is inherently directional.
    Buying calls = bullish. Buying puts = bearish.
    
    This is one of the highest-precision labeling functions because 
    options positions directly reveal market direction expectations.
    """
    text_lower = text.lower()
    
    bullish_options = [
        'bought calls', 'buying calls', 'loaded calls', 'long calls',
        'call options', 'selling puts', 'sold puts', 'bull spread',
        'call spread'
    ]
    bearish_options = [
        'bought puts', 'buying puts', 'loaded puts', 'long puts',
        'put options', 'selling calls', 'sold calls', 'bear spread',
        'put spread'
    ]
    
    if any(phrase in text_lower for phrase in bullish_options):
        return BULLISH
    if any(phrase in text_lower for phrase in bearish_options):
        return BEARISH
    return ABSTAIN


def lf_price_target_mention(text):
    """
    Price targets are almost always bullish (projecting upside).
    Pattern: "PT $XXX", "price target $XXX", "see this at $XXX"
    """
    import re
    if re.search(r'(PT|price target|see this at|heading to)\s*\$?\d+', 
                 text, re.IGNORECASE):
        return BULLISH
    return ABSTAIN


def lf_loss_reporting(text):
    """
    Posts reporting personal losses.
    "Down XX%", "lost $XXX"
    
    Could be bearish OR meme (loss porn culture).
    Vote MEME because loss sharing is a WSB cultural behavior.
    If we want bearish, keyword functions will also vote.
    The aggregator resolves the conflict.
    """
    import re
    text_lower = text.lower()
    if re.search(r'down \d+%', text_lower) or \
       re.search(r'lost \$[\d,]+', text_lower) or \
       'bag holding' in text_lower:
        return MEME
    return ABSTAIN


def lf_news_language(text):
    """
    Detect news/reporting language patterns.
    News articles have distinctive structure: formal tone, 
    third-person references, business language.
    
    Known limitations:
    - Someone quoting news in their opinion post
    - Formal-sounding analysis posts
    """
    news_patterns = [
        'announces', 'announced', 'according to', 'reports',
        'reported', 'filing', 'SEC', 'IPO', 'acquisition',
        'acquires', 'partnership', 'revenue', 'quarterly',
        'year-over-year', 'market cap', 'analysts',
        'upgrade', 'downgrade', 'price target set',
        'breaking:', 'just in:', 'source:'
    ]
    text_lower = text.lower()
    matches = sum(1 for p in news_patterns if p.lower() in text_lower)
    if matches >= 2:  # require multiple news signals
        return NEUTRAL
    return ABSTAIN


# ============================================
# SARCASM/IRONY DETECTION FUNCTIONS
# ============================================

def lf_sarcasm_indicators(text):
    """
    Detect sarcastic constructions that invert sentiment.
    "definitely going to moon" with 🤡 = actually bearish/mocking.
    
    Known limitations:
    - Sarcasm detection from text alone is fundamentally hard
    - Some genuine posts use "definitely" without sarcasm
    - Low coverage but interesting precision dynamics
    """
    text_lower = text.lower()
    
    sarcasm_markers = ['definitely', 'surely', 'of course', 'totally',
                       'oh yeah', 'sure thing', 'trust me bro', 
                       'what could go wrong', 'literally can\'t go tits up']
    bullish_words = ['moon', 'rocket', '$100', '$1000', 'going up',
                     'squeeze', 'to the moon', 'million', 'guaranteed']
    
    has_sarcasm = any(m in text_lower for m in sarcasm_markers)
    has_bullish = any(w in text_lower for w in bullish_words)
    has_clown = '🤡' in text or '🙄' in text
    
    if has_sarcasm and has_bullish:
        return BEARISH
    if has_clown and has_bullish:
        return BEARISH
    return ABSTAIN


def lf_self_deprecating(text):
    """
    WSB self-deprecating humor about losses = MEME culture.
    "My wife's boyfriend", "behind Wendy's", "smooth brain"
    """
    text_lower = text.lower()
    self_deprecating = [
        "wife's boyfriend", 'wifes boyfriend', 'smooth brain',
        'my portfolio', 'behind wendy', 'food stamps',
        'cardboard box', 'dumpster', 'financially ruined',
        'eating ramen', 'i deserve this'
    ]
    if any(phrase in text_lower for phrase in self_deprecating):
        return MEME
    return ABSTAIN


# ============================================
# SOURCE-AWARE FUNCTIONS (use metadata)
# ============================================

def lf_stocktwits_user_sentiment(text, metadata=None):
    """
    Stocktwits posts sometimes include user-submitted sentiment tags.
    If available, use them as a labeling signal.
    
    This is a unique weak supervision source — the original poster 
    self-labeled their sentiment. It's noisy (people sometimes tag 
    bullish on bearish posts for trolling) but valuable.
    """
    if metadata and metadata.get('user_sentiment'):
        tag = metadata['user_sentiment'].lower()
        if tag == 'bullish':
            return BULLISH
        if tag == 'bearish':
            return BEARISH
    return ABSTAIN


def lf_reddit_flair(text, metadata=None):
    """
    Reddit post flairs provide weak signal about content type.
    - "DD" (Due Diligence) → usually neutral/bullish analysis
    - "YOLO" → meme
    - "Loss" → meme
    - "Gain" → could be bullish or meme
    - "Discussion" → neutral
    - "News" → neutral
    """
    if metadata and metadata.get('flair'):
        flair = metadata['flair'].lower()
        if flair in ['yolo', 'loss', 'meme']:
            return MEME
        if flair in ['discussion', 'news', 'dd']:
            return NEUTRAL
    return ABSTAIN


# ============================================
# REGISTRY
# ============================================

# All labeling functions for easy iteration
LABELING_FUNCTIONS = [
    lf_keyword_bullish,
    lf_keyword_bearish,
    lf_keyword_neutral,
    lf_keyword_meme,
    lf_emoji_bullish,
    lf_emoji_bearish,
    lf_emoji_meme,
    lf_question_structure,
    lf_short_post,
    lf_all_caps_ratio,
    lf_options_directional,
    lf_price_target_mention,
    lf_loss_reporting,
    lf_news_language,
    lf_sarcasm_indicators,
    lf_self_deprecating,
]

# Functions that use metadata (only applied when metadata available)
METADATA_FUNCTIONS = [
    lf_stocktwits_user_sentiment,
    lf_reddit_flair,
]
File: src/labeling/aggregator.py
pythonCopy"""
Label Aggregation Strategies

Combines votes from multiple labeling functions into a single 
label per post. Implements three strategies of increasing sophistication.

The aggregator also computes per-post metadata:
- confidence: how certain we are in the final label
- coverage: whether any function voted
- conflict: whether functions disagreed
- vote_breakdown: which functions voted what
"""

class LabelAggregator:
    
    def __init__(self, strategy="confidence_weighted", config=None):
        """
        strategy options:
        - "majority": Simple majority vote
        - "weighted": Functions have manually assigned weights
        - "confidence_weighted": Only assign label if confidence > threshold
        """
        pass
    
    def aggregate_single(self, text, metadata=None):
        """
        Run all labeling functions on a single post.
        
        Returns dict:
        {
            'final_label': str or None,
            'confidence': float (0.0 to 1.0),
            'votes': {
                'lf_keyword_bullish': 'bullish',
                'lf_emoji_meme': 'meme',
                ...
            },
            'num_votes': int,
            'num_abstains': int,
            'has_conflict': bool,
            'competing_labels': {'bullish': 3, 'meme': 2},
        }
        """
        pass
    
    def aggregate_batch(self, df):
        """
        Label entire DataFrame. Add columns:
        - 'programmatic_label': final label
        - 'label_confidence': float
        - 'label_coverage': bool
        - 'label_conflict': bool
        - 'vote_breakdown': dict
        """
        pass
    
    def _majority_vote(self, votes):
        """Most common non-abstain label wins. Ties broken by predefined priority."""
        pass
    
    def _weighted_vote(self, votes):
        """
        Each function has a weight. Sum weights per label. Highest wins.
        Weights reflect estimated precision:
        - lf_options_directional: weight=3.0 (very reliable)
        - lf_emoji_bullish: weight=1.0 (noisy — rockets used in memes too)
        - lf_keyword_bullish: weight=2.0 (moderately reliable)
        """
        pass
    
    def _confidence_weighted(self, votes):
        """
        Like weighted vote, but only assign label if confidence > threshold.
        Posts below threshold get label=None (uncertain).
        These uncertain posts are the most valuable for human review.
        """
        pass
File: src/labeling/quality.py
pythonCopy"""
Label Quality Assessment

Measures how good our programmatic labels are before we train any model.
This is the most Snorkel-relevant part of the project.
"""

class LabelQualityAnalyzer:
    
    def __init__(self, labeling_functions, aggregator):
        pass
    
    def per_function_report(self, df):
        """
        For each labeling function, compute:
        - coverage: % of posts it voted on (vs abstained)
        - label_distribution: what labels it assigns and in what proportion
        - conflict_rate: how often it disagrees with the final aggregated label
        - accuracy_on_gold: if gold labels exist, precision and recall
        - overlap: which other functions it agrees/disagrees with most
        
        Returns DataFrame: one row per function, columns are metrics.
        """
        pass
    
    def aggregate_quality_report(self, df):
        """
        Overall labeling pipeline quality:
        - total_coverage: % of posts that received at least one vote
        - avg_votes_per_post: mean number of functions that voted per post
        - conflict_rate: % of posts where functions disagreed
        - confidence_distribution: histogram of confidence scores
        - uncertain_count: posts that couldn't be confidently labeled
        - label_distribution: final label distribution after aggregation
        - expected_vs_actual: does output distribution match expected?
        """
        pass
    
    def compare_to_gold(self, df, gold_df):
        """
        THE KEY METRIC: How do programmatic labels compare to gold standard?
        
        Returns:
        - Per-class precision, recall, F1
        - Confusion matrix
        - List of disagreements with analysis:
          For each disagreement:
            - post text
            - programmatic label + confidence
            - gold label + ambiguity score
            - which functions voted and how
            - likely reason for disagreement
        - Agreement rate overall
        - Agreement rate by ambiguity score (should be higher for easy posts)
        """
        pass
    
    def label_quality_experiment(self, df, gold_df, X_test, y_test_gold):
        """
        THE THESIS EXPERIMENT.
        
        Train the SAME model (TF-IDF + LogReg, same hyperparameters) on:
        1. Gold labels (best possible data)
        2. Programmatic labels (our pipeline output)
        3. Noisy labels (random noise injected into gold — simulate bad labeling)
        4. Random labels (completely random — baseline)
        
        Evaluate all four on the gold test set.
        
        Expected results:
        - Gold: F1 ~0.85+
        - Programmatic: F1 ~0.75-0.82 (close to gold!)
        - Noisy: F1 ~0.55-0.65 (significantly worse)
        - Random: F1 ~0.25 (chance level for 4 classes)
        
        THIS PROVES: data quality > model complexity.
        The same model with better labels dramatically outperforms.
        
        Returns comparison table and Plotly bar chart.
        """
        pass

ML MODEL PIPELINE
File: src/models/pipeline.py
pythonCopy"""
Production ML Pipeline for Sentiment Classification

This is a real, deployable pipeline — not a notebook exercise.
Supports training, evaluation, inference, and model persistence.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from datetime import datetime


class SentimentPipeline:
    
    def __init__(self, config=None):
        self.config = config or {}
        self.vectorizer = None
        self.model = None
        self.metadata = {}
        self.is_trained = False
    
    def preprocess(self, text):
        """
        MINIMAL preprocessing. Lowercase + whitespace normalization only.
        
        DO NOT strip emojis — they carry sentiment signal.
        DO NOT strip ticker symbols — they're features.
        DO NOT strip punctuation — !! and ?? indicate emotion.
        
        The vectorizer handles feature selection. Our job is just 
        to normalize obvious noise (casing, extra spaces).
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train(self, texts, labels, validation_split=True):
        """
        Train the full pipeline.
        
        Steps:
        1. Preprocess all texts
        2. Split into train/validation if validation_split=True
        3. Fit TfidfVectorizer on training data
        4. Train LogisticRegression on training data
        5. Evaluate on validation set
        6. Store metadata (training date, dataset size, metrics)
        7. Run cross-validation and store results
        
        Returns training report dict.
        """
        pass
    
    def predict(self, texts):
        """
        Predict sentiment for one or more texts.
        
        Returns list of dicts:
        [
            {
                'text': str,
                'label': str,
                'confidence': float,
                'probabilities': {'bullish': 0.7, 'bearish': 0.1, ...}
            }
        ]
        """
        pass
    
    def predict_single(self, text):
        """Convenience method for single text prediction."""
        pass
    
    def get_feature_importance(self, top_n=15):
        """
        Return top predictive features per class.
        
        Returns dict:
        {
            'bullish': [('calls', 1.23), ('buying', 1.1), ...],
            'bearish': [('puts', 1.45), ('crash', 1.3), ...],
            'neutral': [('thoughts', 0.9), ('anyone', 0.8), ...],
            'meme': [('apes', 1.5), ('tendies', 1.4), ...],
        }
        
        This validates the model learned meaningful patterns.
        """
        pass
    
    def error_analysis(self, texts, true_labels, predicted_labels):
        """
        Detailed analysis of every misclassification.
        
        For each error:
        - The text
        - True label vs predicted label
        - Confidence score (was model uncertain?)
        - Top features that drove the prediction
        - Error category: 'labeling_ambiguity' | 'model_limitation' | 'data_quality'
        
        Aggregate analysis:
        - Most confused class pairs
        - Average confidence on correct vs incorrect predictions
        - Errors by ambiguity score (if gold set)
        """
        pass
    
    def save(self, path="data/models"):
        """
        Save trained pipeline to disk.
        
        Saves:
        - tfidf_vectorizer.pkl (joblib)
        - sentiment_model.pkl (joblib)
        - model_metadata.json (training date, metrics, config, feature names)
        """
        pass
    
    def load(self, path="data/models"):
        """Load trained pipeline from disk."""
        pass
File: src/models/versioning.py
pythonCopy"""
Model Versioning

Track model versions with metadata so we can compare 
different training runs (different label sources, different configs).
"""

class ModelVersion:
    
    def __init__(self, model_dir="data/models"):
        pass
    
    def save_version(self, pipeline, label_source, metrics, notes=""):
        """
        Save a model version with full metadata.
        
        Creates directory: data/models/v{N}_{label_source}_{timestamp}/
        Contains: vectorizer, model, metadata
        
        Metadata includes:
        - version number
        - label_source: 'gold' | 'programmatic' | 'noisy' | 'random'
        - training_date
        - dataset_size
        - metrics (full classification report)
        - config used
        - notes
        """
        pass
    
    def list_versions(self):
        """List all saved model versions with summary metrics."""
        pass
    
    def compare_versions(self, version_ids):
        """
        Compare metrics across model versions.
        Returns comparison table.
        """
        pass
    
    def load_version(self, version_id):
        """Load a specific model version."""
        pass

ENTITY EXTRACTION
File: src/extraction/ticker_extractor.py
pythonCopy"""
Ticker Symbol and Company Name Extraction

Extracts company/ticker mentions from financial social media posts.
Returns canonical company names with extraction evidence.

Challenges in this domain:
- $AAPL is a ticker but $5 is a price
- Some tickers are common words: $F (Ford), $T (AT&T), $AI, $ALL
- Informal references: "Papa Musk" → Tesla
- Emoji references: 🍎 → Apple
- Abbreviated names: "GOOG" without $ prefix
- Product names: "iPhone" → Apple (do we extract this?)
- Multiple tickers in one post
- Tickers in hashtags: #NVDA
"""

class TickerExtractor:
    
    def __init__(self):
        self.ticker_map = {
            'AAPL': 'Apple', 'TSLA': 'Tesla', 'MSFT': 'Microsoft',
            'GOOG': 'Google', 'GOOGL': 'Google', 'AMZN': 'Amazon',
            'NVDA': 'NVIDIA', 'META': 'Meta', 'GME': 'GameStop',
            'AMC': 'AMC', 'SPY': 'S&P 500 ETF', 'QQQ': 'Nasdaq ETF',
            'PLTR': 'Palantir', 'BB': 'BlackBerry', 'SOFI': 'SoFi',
            'WISH': 'ContextLogic', 'NFLX': 'Netflix', 'DIS': 'Disney',
            'AMD': 'AMD', 'INTC': 'Intel', 'JPM': 'JPMorgan',
            'BAC': 'Bank of America', 'F': 'Ford', 'GE': 'GE',
            'T': 'AT&T', 'COIN': 'Coinbase', 'HOOD': 'Robinhood',
            'RIVN': 'Rivian', 'LCID': 'Lucid', 'NIO': 'NIO',
            'BABA': 'Alibaba', 'TSM': 'TSMC', 'ORCL': 'Oracle',
            'CRM': 'Salesforce', 'ABNB': 'Airbnb', 'SNAP': 'Snap',
            'UBER': 'Uber', 'LYFT': 'Lyft', 'SQ': 'Block',
            'PYPL': 'PayPal', 'V': 'Visa', 'MA': 'Mastercard',
        }
        
        # Tickers that are also common words — require $ prefix to match
        self.ambiguous_tickers = {'F', 'T', 'V', 'AI', 'ALL', 'IT', 'NOW'}
        
        self.company_aliases = {
            'apple': 'Apple', 'tesla': 'Tesla', 'microsoft': 'Microsoft',
            'google': 'Google', 'alphabet': 'Google', 'amazon': 'Amazon',
            'nvidia': 'NVIDIA', 'meta': 'Meta', 'facebook': 'Meta',
            'gamestop': 'GameStop', 'palantir': 'Palantir',
            'netflix': 'Netflix', 'disney': 'Disney',
        }
        
        self.informal_aliases = {
            'papa musk': 'Tesla', 'elon': 'Tesla',
            'zuck': 'Meta', 'zuckerberg': 'Meta',
            'tim cook': 'Apple', 'cook': 'Apple',
            'bezos': 'Amazon', 'jensen': 'NVIDIA',
            'satya': 'Microsoft', 'nadella': 'Microsoft',
            'lisa su': 'AMD',
        }
        
        self.emoji_map = {
            '🍎': 'Apple',
        }
    
    def extract(self, text):
        """
        Extract all company entities from text.
        Returns list of canonical company names (deduplicated).
        """
        pass
    
    def extract_with_evidence(self, text):
        """
        Extract entities with provenance tracking.
        
        Returns list of dicts:
        [
            {
                'canonical': 'Tesla',
                'surface_form': '$TSLA',
                'method': 'cashtag',
                'position': (15, 20)  # character span
            },
            {
                'canonical': 'Tesla',
                'surface_form': 'Elon',
                'method': 'informal_alias',
                'position': (45, 49)
            }
        ]
        """
        pass
    
    def _extract_cashtags(self, text):
        """Extract $TICKER patterns. Handle $5 vs $AAPL."""
        pass
    
    def _extract_bare_tickers(self, text):
        """
        Extract ALL-CAPS ticker-like words without $ prefix.
        Only match tickers NOT in the ambiguous set.
        Must be standalone words (not part of a sentence in caps).
        """
        pass
    
    def _extract_company_names(self, text):
        """Extract full company names via alias lookup."""
        pass
    
    def _extract_informal(self, text):
        """Extract informal/people references."""
        pass
    
    def _extract_emoji(self, text):
        """Extract emoji-to-company mappings."""
        pass
File: src/extraction/normalizer.py
pythonCopy"""
Entity Normalization

Maps all entity variations to a single canonical form 
for fair evaluation comparison.

This module demonstrates a key Snorkel concept:
evaluation quality depends on normalization quality.
Metrics computed WITHOUT normalization are artificially low.
"""

class EntityNormalizer:
    
    def __init__(self):
        self.canonical_map = {}
        self._build_map()
    
    def _build_map(self):
        """
        Build comprehensive normalization map.
        All known variations → single lowercase canonical form.
        """
        mappings = {
            'apple': ['apple', 'apple inc', 'apple inc.', 'aapl', 
                      '$aapl', '#aapl'],
            'tesla': ['tesla', 'tesla inc', 'tesla motors', 'tsla', 
                      '$tsla', '#tsla'],
            'microsoft': ['microsoft', 'msft', '$msft', 'microsoft corp'],
            'google': ['google', 'alphabet', 'goog', 'googl', '$goog', 
                       '$googl'],
            'amazon': ['amazon', 'amzn', '$amzn', 'amazon.com'],
            'nvidia': ['nvidia', 'nvda', '$nvda', '#nvda', 'nvidia corp'],
            'meta': ['meta', 'meta platforms', 'facebook', 'fb', '$meta'],
            'gamestop': ['gamestop', 'gme', '$gme', '#gme', 'game stop'],
            'amc': ['amc', 'amc entertainment', '$amc', '#amc'],
            'palantir': ['palantir', 'pltr', '$pltr', 'palantir technologies'],
            's&p 500 etf': ['spy', '$spy', 's&p 500', 's&p500', 'sp500',
                            's&p 500 etf'],
            'nasdaq etf': ['qqq', '$qqq', 'nasdaq'],
            # ... extend for all tracked companies
        }
        
        for canonical, variations in mappings.items():
            for variation in variations:
                self.canonical_map[variation.lower()] = canonical
    
    def normalize(self, entity):
        """Normalize single entity to canonical form."""
        entity_lower = entity.strip().lower()
        entity_lower = re.sub(r'^[\$#]', '', entity_lower)  # strip $ and #
        return self.canonical_map.get(entity_lower, entity_lower)
    
    def normalize_set(self, entities):
        """Normalize list of entities, deduplicate after normalization."""
        normalized = set()
        for entity in entities:
            normalized.add(self.normalize(entity))
        return sorted(normalized)
    
    def entities_match(self, entity_a, entity_b):
        """Check if two entity strings refer to the same company."""
        return self.normalize(entity_a) == self.normalize(entity_b)

EVALUATION FRAMEWORK
File: src/evaluation/classification.py
pythonCopy"""
Sentiment Classification Evaluation

Goes beyond sklearn's classification_report to provide 
production-relevant analysis.
"""

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def evaluate_classification(y_true, y_pred, texts=None, 
                            confidence_scores=None, gold_metadata=None):
    """
    Comprehensive classification evaluation.
    
    Returns dict with:
    
    'report': sklearn classification_report as dict
    'confusion_matrix': as DataFrame with labeled rows/columns
    'accuracy': float
    'weighted_f1': float
    'per_class': {
        'bullish': {'precision': x, 'recall': x, 'f1': x, 'support': n},
        ...
    }
    'errors': [   # only if texts provided
        {
            'text': str,
            'true_label': str,
            'predicted_label': str,
            'confidence': float or None,
            'error_category': str,  # 'labeling_ambiguity' | 'model_limitation'
            'ambiguity_score': int or None  # from gold metadata
        }
    ]
    'summary': {
        'most_confused_pair': ('meme', 'bullish'),
        'avg_confidence_correct': float,
        'avg_confidence_incorrect': float,
        'error_rate_by_ambiguity': {1: 0.05, 2: 0.12, 3: 0.25, 4: 0.45, 5: 0.60}
    }
    """
    pass
File: src/evaluation/extraction.py
pythonCopy"""
Entity Extraction Evaluation

Entity-level precision, recall, F1 with normalization.
Demonstrates that normalization is critical for fair evaluation.
"""

def evaluate_extraction(predictions, ground_truths, normalizer):
    """
    Entity-level evaluation with normalization.
    
    Returns dict with:
    
    'metrics': {
        'precision': float,
        'recall': float,
        'f1': float
    }
    'metrics_without_normalization': {
        # Same metrics but with strict string matching
        # SHOW THIS to demonstrate normalization impact
    }
    'normalization_lift': {
        # Difference between normalized and unnormalized metrics
        # This proves why normalization matters
    }
    'per_entity_performance': {
        'apple': {'tp': n, 'fp': n, 'fn': n, 'f1': x},
        ...
    }
    'errors': [
        {
            'post_idx': int,
            'text': str,
            'predicted': set,
            'ground_truth': set,
            'false_positives': set,  # extracted but shouldn't have
            'false_negatives': set,  # missed
            'error_type': str  # 'over_extraction' | 'missed' | 'normalization_gap'
        }
    ]
    """
    pass
File: src/evaluation/label_quality.py
pythonCopy"""
Label Quality Evaluation

Compares programmatic labels against gold standard.
Runs the thesis experiment showing data quality > model complexity.
"""

def run_thesis_experiment(df_programmatic, df_gold, config):
    """
    THE CENTRAL EXPERIMENT OF THE PROJECT.
    
    Steps:
    1. Split gold data into train/test
    2. Create four training sets:
       a. gold_train: Gold labels on training split
       b. programmatic_train: Programmatic labels on training split
       c. noisy_train: Gold labels with 30% random noise injected
       d. random_train: Completely random labels
    3. Train identical SentimentPipeline on each
    4. Evaluate all four on gold_test
    5. Return comparison table
    
    Returns dict with:
    'results_table': DataFrame with columns [label_source, f1, precision, recall]
    'per_class_comparison': nested dict with per-class metrics per source
    'thesis_validated': bool (is programmatic > noisy?)
    'programmatic_vs_gold_gap': float (how close is programmatic to gold?)
    'visualization_data': dict ready for Plotly bar chart
    """
    pass

STREAMLIT DASHBOARD
File: app/streamlit_app.py
pythonCopy"""
MarketPulse Dashboard — Main Entry Point

Run with: streamlit run app/streamlit_app.py

Multi-page layout with sidebar navigation.
Dark financial theme.
"""

import streamlit as st

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("📊 MarketPulse")
st.sidebar.markdown("""
**Data-Centric Sentiment Intelligence**

Demonstrating that label quality drives 
model performance more than model complexity.
""")

# Show data source status
# Show model training status
# Show last refresh time

# Main page: project overview with key metrics
st.title("MarketPulse: Financial Sentiment Intelligence")
st.markdown("### Data-centric AI applied to financial social media")

# Key metric cards:
# - Total posts ingested
# - Labeling coverage %
# - Model F1 score
# - Thesis experiment result (programmatic vs noisy)

# Quick summary visualization
# - Label distribution pie chart
# - Data source composition
# - The thesis bar chart (gold vs programmatic vs noisy vs random)
Page 1: Data Ingestion (pages/1_data_ingestion.py)
CopyFEATURES:
- Data source toggle: Live / Synthetic / Auto
- Date range picker (start date, end date)
- "Ingest Data" button that triggers the pipeline
- Source status indicators (green = connected, red = no API key)
- After ingestion:
  - Total posts count, posts per source
  - Sample posts table (filterable, searchable)
  - Text length distribution
  - Posts over time (line chart by day)
  - Source composition pie chart
  - Data quality audit: nulls, duplicates, empty posts, avg post length
Page 2: Labeling Studio (pages/2_labeling_studio.py)
CopyFEATURES:
- Labeling function explorer:
  - List of all 18 functions with descriptions
  - Per-function: coverage %, label distribution, precision on gold
  - Expandable: example posts each function labeled
  
- Interactive labeler:
  - Text input box: "Enter a post to see how it gets labeled"
  - Shows which functions fire, their votes, and the aggregated result
  - Visualization of vote breakdown (horizontal stacked bar)
  
- Conflict explorer:
  - Table of posts where functions disagreed
  - For each conflict: show all votes and the resolution
  - Filter by conflict type (bullish vs meme, bearish vs neutral, etc.)

- "Run Labeling" button:
  - Labels all ingested posts
  - Shows progress bar
  - Displays summary statistics when complete
Page 3: Label Quality (pages/3_label_quality.py)
CopyFEATURES:
- THE THESIS CHART:
  - Bar chart: F1 score for gold vs programmatic vs noisy vs random labels
  - THIS IS THE HERO VISUALIZATION OF THE ENTIRE PROJECT
  - Should be prominent, well-designed, and clearly labeled
  - Include annotation: "Same model. Different data quality."

- Label quality metrics:
  - Overall coverage rate
  - Conflict rate
  - Confidence distribution histogram
  - Per-function accuracy on gold set (heatmap or bar chart)

- Programmatic vs Gold comparison:
  - Confusion matrix
  - Per-class agreement rates
  - Disagreement explorer: click a cell to see specific posts

- Uncertain posts:
  - Table of posts that couldn't be confidently labeled
  - These are candidates for human review
  - Sort by confidence (lowest first)

- Coverage heatmap:
  - Posts vs functions matrix
  - Shows which functions voted on which posts
  - Reveals coverage gaps
Page 4: Model Training (pages/4_model_training.py)
CopyFEATURES:
- Training configuration:
  - Label source selector: gold / programmatic / noisy
  - Hyperparameter controls: max_features, ngram_range, C
  - "Train Model" button

- Training results:
  - Classification report table
  - Confusion matrix heatmap
  - Cross-validation scores

- Feature importance:
  - Top 15 features per class (bar chart)
  - Validate model learned meaningful patterns
  - Flag any suspicious features (noise that got high weight)

- Model versioning:
  - List of all trained model versions
  - Compare button: select 2+ versions to compare metrics
  - Best model indicator
Page 5: Entity Extraction (pages/5_entity_extraction.py)
CopyFEATURES:
- Extraction results:
  - Entity-level P/R/F1 metrics
  - Most mentioned companies bar chart (across all posts)

- Normalization impact:
  - SIDE BY SIDE: metrics WITH normalization vs WITHOUT
  - This proves why normalization matters
  - Highlight the lift from normalization

- Extraction evidence viewer:
  - Table: post text | extracted entities | method | confidence
  - Click a post to see detailed extraction breakdown
  - Which surface forms were found, which method extracted them

- Error analysis:
  - False positives: what was incorrectly extracted
  - False negatives: what was missed
  - Categorize: normalization gap vs genuine miss vs ambiguous ticker

- Interactive extractor:
  - Text input: "Enter a post to see extracted tickers"
  - Shows extraction results with evidence
Page 6: Evaluation Deep Dive (pages/6_evaluation.py)
CopyFEATURES:
- Comprehensive metrics dashboard:
  - Classification metrics (per-class P/R/F1)
  - Extraction metrics (entity-level P/R/F1)
  - Label quality metrics (coverage, conflict, gold agreement)

- Error explorer:
  - Filterable table of all misclassifications
  - Filter by: true class, predicted class, confidence range
  - Each error tagged with category
  - Sort by confidence (show uncertain predictions first)

- Performance by difficulty:
  - If gold set has ambiguity scores:
    - Chart: F1 vs ambiguity score
    - Model should do well on easy posts, struggle on ambiguous ones
    - This is expected and shows the model knows what it knows

- Data improvement recommendations:
  - Based on error analysis, suggest specific improvements:
    - "Add more training examples at the meme/bullish boundary"
    - "Clarify labeling guideline for sarcastic posts"
    - "Consider adding a sarcasm-detection labeling function"
  - These should be auto-generated from error patterns
Page 7: Live Inference (pages/7_live_inference.py)
CopyFEATURES:
- Text input area:
  - Large text box: "Paste a financial social media post"
  - "Classify" button

- Results display:
  - Predicted sentiment with confidence bar
  - Probability distribution across all 4 classes (bar chart)
  - Extracted ticker entities with evidence
  - Which labeling functions would fire on this text
  - Top features that drove the prediction

- Batch inference:
  - Upload CSV with texts
  - Download CSV with predictions + confidence + entities
  
- Recent predictions log:
  - History of predictions made in this session
  - Useful for demoing the system

VISUAL DESIGN
pythonCopy# Color palette — dark financial terminal aesthetic
COLORS = {
    'bullish': '#00C853',       # Green
    'bearish': '#FF1744',       # Red  
    'neutral': '#78909C',       # Blue-gray
    'meme': '#FFD600',          # Gold
    
    'primary': '#58A6FF',       # Blue accent
    'secondary': '#8B949E',     # Muted gray
    'success': '#3FB950',       # Green
    'warning': '#D29922',       # Amber
    'danger': '#F85149',        # Red
    
    'bg_primary': '#0D1117',    # Dark background
    'bg_secondary': '#161B22',  # Card background
    'bg_tertiary': '#21262D',   # Hover/active
    'text_primary': '#E6EDF3',  # Primary text
    'text_secondary': '#8B949E', # Secondary text
    'border': '#30363D',        # Borders
}
Use Plotly with template='plotly_dark' for all charts. Custom CSS in Streamlit for card-based metric displays.

CLI AND MAKEFILE
File: Makefile
makefileCopy.PHONY: setup ingest label train evaluate run all clean

setup:
	pip install -r requirements.txt
	python scripts/setup.py  # generate synthetic data if needed

ingest:
	python scripts/ingest.py --days 7

label:
	python scripts/label.py

train:
	python scripts/train.py --source programmatic

evaluate:
	python scripts/evaluate.py

run:
	streamlit run app/streamlit_app.py

all: ingest label train evaluate run

pipeline:
	python scripts/run_pipeline.py  # runs ingest → label → train → evaluate

clean:
	rm -rf data/raw/*
	rm -rf data/labeled/*
	rm -rf data/models/*

test:
	pytest tests/ -v
File: scripts/run_pipeline.py
pythonCopy"""
Run the complete MarketPulse pipeline end-to-end.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --source reddit --days 14
  python scripts/run_pipeline.py --synthetic
"""

import argparse
# 1. Ingest data (live or synthetic)
# 2. Run labeling functions
# 3. Aggregate labels
# 4. Assess label quality
# 5. Train model on programmatic labels
# 6. Run thesis experiment (gold vs programmatic vs noisy)
# 7. Extract entities
# 8. Evaluate everything
# 9. Print summary report
# 10. Save all artifacts

REQUIREMENTS
Copy# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
streamlit>=1.29.0
praw>=7.7.0
requests>=2.31.0
python-dotenv>=1.0.0
pyyaml>=6.0
joblib>=1.3.0
wordcloud>=
