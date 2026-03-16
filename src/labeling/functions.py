"""
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

import re

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
    - Triggers on sarcastic usage: "definitely going to moon"
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
    - Rockets heavily used in meme posts — precision is lower than expected
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

    Note: Clown is often used to mock others — could indicate bearish
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
    Examples: "GUH", "rocket rocket rocket", "RIP", "moon soon", "bruh"

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
    """
    words = text.split()
    if len(words) < 3:
        return ABSTAIN
    caps_words = [w for w in words if w.isupper() and len(w) > 1
                  and not w.startswith('$')]
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
    """
    text_lower = text.lower()
    if re.search(r'down \d+%', text_lower) or \
       re.search(r'lost \$[\d,]+', text_lower) or \
       'bag holding' in text_lower:
        return MEME
    return ABSTAIN


def lf_news_language(text):
    """
    Detect news/reporting language patterns.

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
    if matches >= 2:
        return NEUTRAL
    return ABSTAIN


# ============================================
# SARCASM/IRONY DETECTION FUNCTIONS
# ============================================

def lf_sarcasm_indicators(text):
    """
    Detect sarcastic constructions that invert sentiment.
    "definitely going to moon" with clown emoji = actually bearish/mocking.

    Known limitations:
    - Sarcasm detection from text alone is fundamentally hard
    - Some genuine posts use "definitely" without sarcasm
    - Low coverage but interesting precision dynamics
    """
    text_lower = text.lower()

    sarcasm_markers = ['definitely', 'surely', 'of course', 'totally',
                       'oh yeah', 'sure thing', 'trust me bro',
                       'what could go wrong', "literally can't go tits up"]
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
# REGISTRY
# ============================================

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

METADATA_FUNCTIONS = []
