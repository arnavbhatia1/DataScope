"""
Synthetic Data Generator for MarketPulse

Generates 550+ realistic financial social media posts using template-based
generation with random slot-filling for tickers, prices, emojis, and slang.

Also produces a 100-post gold standard evaluation set.

Used as fallback when no API keys are configured, for unit testing,
demo mode, and as a consistent evaluation baseline.
"""

import os
import re
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.ingestion.base import BaseIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS = [
    "AAPL", "TSLA", "MSFT", "GOOG", "AMZN", "NVDA", "META", "GME",
    "AMC", "SPY", "QQQ", "PLTR", "BB", "SOFI", "WISH", "NFLX",
    "DIS", "AMD", "INTC", "JPM", "BAC", "COIN", "HOOD", "RIVN",
    "LCID", "NIO", "BABA", "CRM", "ABNB", "SNAP", "UBER", "PYPL",
]

MEME_TICKERS = ["GME", "AMC", "BB", "WISH", "HOOD", "PLTR", "SOFI"]
TECH_TICKERS = ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN", "NVDA", "META", "AMD", "NFLX", "CRM"]
ETF_TICKERS = ["SPY", "QQQ"]

BULLISH_EMOJIS = ["🚀", "🌙", "📈", "🐂", "💰", "🤑", "⬆️"]
BEARISH_EMOJIS = ["📉", "🐻", "💀", "🔻", "⬇️", "😱", "🩸"]
MEME_EMOJIS = ["💎", "🙌", "🦍", "🤡", "🎰", "🍗", "🫠", "🍔", "📦"]
GENERAL_EMOJIS = BULLISH_EMOJIS + BEARISH_EMOJIS + MEME_EMOJIS + ["🙄", "😂", "🔥", "👀", "😤"]

INFORMAL_WORDS = ["ngl", "imo", "tbh", "lol", "fr", "smh", "bruh", "lmao", "fwiw", "idk"]
MISSPELLINGS = {
    "going to": "gunna",
    "tomorrow": "tmrw",
    "stocks": "stonks",
    "probably": "prolly",
    "because": "cuz",
    "definitely": "def",
}

AUTHOR_TEMPLATES = [
    "WSB_trader_{n}", "diamond_hands_{name}", "tendies_lover",
    "theta_gang_{n}", "bull_run_{n}", "bear_cave_{n}", "smooth_brain_{n}",
    "yolo_king_{n}", "puts_printer_{n}", "ape_strong_{n}",
    "degen_trader_{n}", "moon_boy_{n}", "stonks_guy_{n}",
    "value_investor_{n}", "swing_trader_{n}", "day_trader_{n}",
    "option_wizard_{n}", "chart_reader_{n}", "dd_master_{n}",
    "FD_buyer_{n}", "bagholder_{n}", "paper_hands_{n}",
]

AUTHOR_NAMES = ["joe", "mike", "alex", "sam", "chris", "dan", "tom", "nick", "jake", "ben"]


def _random_author(rng):
    """Generate a random author username."""
    tmpl = rng.choice(AUTHOR_TEMPLATES)
    return tmpl.format(n=rng.randint(1, 999), name=rng.choice(AUTHOR_NAMES))


def _random_price(rng, low=10, high=500):
    return rng.randint(low, high)


def _random_pct(rng, low=5, high=95):
    return rng.randint(low, high)


def _random_num(rng, low=10, high=999):
    return rng.randint(low, high)


def _ticker_fmt(ticker, rng, force_dollar=False):
    """Format ticker with optional $ prefix."""
    if force_dollar:
        return f"${ticker}"
    return f"${ticker}" if rng.random() < 0.55 else ticker


def _maybe_add_emoji(text, rng, emoji_pool, prob=0.45):
    """Optionally append 1-3 emojis from a pool."""
    if rng.random() < prob:
        n = rng.choice([1, 1, 2, 2, 3])
        emojis = " ".join(rng.choice(emoji_pool) for _ in range(n))
        text = f"{text} {emojis}"
    return text


def _maybe_add_informal(text, rng, prob=0.25):
    """Optionally prepend or append an informal word."""
    if rng.random() < prob:
        word = rng.choice(INFORMAL_WORDS)
        if rng.random() < 0.5:
            text = f"{word} {text}"
        else:
            text = f"{text} {word}"
    return text


def _maybe_misspell(text, rng, prob=0.10):
    """Randomly apply one misspelling."""
    if rng.random() < prob:
        original, replacement = rng.choice(list(MISSPELLINGS.items()))
        text = text.replace(original, replacement, 1)
    return text


def _maybe_hashtag(text, rng, prob=0.08):
    """Optionally append a hashtag."""
    if rng.random() < prob:
        tags = ["#YOLO", "#bullish", "#bearish", "#stonks", "#diamondhands", "#tothemoon"]
        text = f"{text} {rng.choice(tags)}"
    return text


def _power_law_score(rng):
    """Generate a score following a power-law distribution."""
    u = rng.random()
    if u < 0.65:
        return rng.randint(1, 20)
    elif u < 0.88:
        return rng.randint(21, 100)
    elif u < 0.96:
        return rng.randint(101, 500)
    else:
        return rng.randint(501, 5000)


# ---------------------------------------------------------------------------
# Template Definitions
# ---------------------------------------------------------------------------

BULLISH_TEMPLATES = {
    "conviction_buy": [
        "Just loaded {qty} shares of {ticker}. This is going to ${price}.",
        "Went all in on {ticker} today. {qty} shares at ${price}. Not selling till ${price2}.",
        "Added another {qty} shares of {ticker} to my position. Long term hold.",
        "Loading up on {ticker} while everyone is sleeping on it. Undervalued af.",
        "Just bought {ticker} at ${price}. This is free money at these levels.",
        "{ticker} is the most undervalued stock in the market right now. Buying everything I can.",
        "Accumulating {ticker} aggressively. My average is ${price} and I keep adding.",
        "Dumped my savings into {ticker}. Not even worried. This thing is going to ${price2}.",
        "ALL IN on {ticker}. {qty} shares. Lets ride.",
        "Best entry point for {ticker} in months. Just loaded up.",
        "Bought {ticker} at the open. PT ${price2}. Easy 2x from here.",
        "If you're not buying {ticker} at these prices you hate money.",
        "My portfolio is now 80% {ticker} and I'm sleeping like a baby.",
        "Spent my entire bonus on {ticker}. No regrets. This goes to ${price2}.",
        "Buying {ticker} hand over fist. The fundamentals are rock solid.",
        "Just added more {ticker}. Conviction is at an all time high.",
        "Scaled into {ticker} all week. Average cost ${price}. Feeling great about it.",
        "Backed up the truck on {ticker} today. This is a generational buy.",
        "{ticker} at ${price} is a steal. I'm buying as much as I can afford.",
        "Holding {qty} shares of {ticker}. Not selling a single share under ${price2}.",
        "Started a massive position in {ticker}. This is my highest conviction play.",
        "Just pressed buy on {ticker} without even flinching. ${price} is a gift.",
        "Picked up another {qty} shares of {ticker} on this pullback.",
        "Call me crazy but I think {ticker} hits ${price2} by end of year.",
        "Did my DD on {ticker}. The numbers dont lie. Going long hard.",
    ],
    "technical_bull": [
        "{ticker} breaking out of the wedge pattern. Bullish.",
        "Golden cross forming on {ticker} daily chart. I'm buying.",
        "{ticker} just broke above the 200 day moving average. Extremely bullish signal.",
        "Cup and handle forming on {ticker}. Textbook bullish setup.",
        "{ticker} bounced perfectly off support at ${price}. Adding here.",
        "MACD crossover on {ticker}. Momentum is shifting. Long.",
        "{ticker} volume just spiked 3x average. Something is brewing. Bullish.",
        "RSI on {ticker} was oversold at 28. Now reversing. Perfect entry.",
        "Bull flag on {ticker} 4hr chart. Targeting ${price2}.",
        "{ticker} just filled the gap at ${price}. Next leg up starts now.",
        "Three white soldiers on {ticker} weekly candles. Strong buy signal.",
        "The {ticker} chart looks incredible. Breakout imminent.",
        "Fib retracement on {ticker} held perfectly at 0.618. Going up from here.",
        "{ticker} is building a beautiful base at ${price}. Accumulation phase.",
        "Relative strength on {ticker} vs the index is surging. Very bullish.",
        "VWAP reclaim on {ticker}. This is headed to ${price2}.",
        "{ticker} squeezed through resistance at ${price}. Next stop ${price2}.",
        "Ascending triangle on {ticker}. Easy long here.",
        "The weekly on {ticker} looks pristine. Higher lows, about to break out.",
        "Double bottom confirmed on {ticker} at ${price}. Loading.",
        "{ticker} above all major moving averages. Path of least resistance is up.",
        "Bollinger bands tightening on {ticker}. Big move coming. Im positioned long.",
        "Accumulation/distribution line on {ticker} is screaming buy.",
        "{ticker} just broke its 52 week high. Momentum play. Long.",
        "On balance volume on {ticker} is diverging bullish. Smart money is buying.",
    ],
    "earnings_optimism": [
        "{ticker} earnings going to crush it. Cloud growth is insane.",
        "Expecting a massive beat from {ticker} this quarter. All the channel checks are positive.",
        "{ticker} about to report and whisper numbers are way above consensus. Bullish.",
        "Loaded {ticker} before earnings. The guidance is going to be the real catalyst.",
        "Every data point I see says {ticker} is going to smash earnings. Long into the print.",
        "{ticker} Q3 is going to blow minds. Revenue acceleration incoming.",
        "My sources in the industry say {ticker} had an incredible quarter. Buying before earnings.",
        "Sell side is way too low on {ticker}. Expect a massive beat and raise.",
        "{ticker} EPS estimates keep getting revised higher. Smart money knows whats coming.",
        "AWS numbers alone will carry {ticker} earnings. Also buying calls.",
        "The {ticker} earnings whisper number is 20% above street consensus. Loading.",
        "{ticker} same store sales data looks incredible. Earnings going to be huge.",
        "Guidance raise from {ticker} is going to send this to ${price2}.",
        "Bought {ticker} ahead of earnings. The setup is perfect. Under-owned, under-estimated.",
        "This is going to be the best quarter {ticker} has ever had. Mark my words.",
        "{ticker} ad revenue has been surging all quarter. Earnings will show it. Long.",
        "Short interest on {ticker} heading into earnings is insane. This is going to squeeze.",
        "Prepping for a big {ticker} earnings beat. Street estimates are a joke.",
        "Channel checks for {ticker} are the strongest I've seen in years. Buying more before the print.",
        "{ticker} margin expansion this quarter is going to surprise everyone. Very bullish.",
        "Holding {ticker} through earnings. High conviction in a beat and raise.",
        "Supplier data says {ticker} orders are up 40% YoY. Earnings going to be massive.",
        "{ticker} guided conservatively last quarter. Actual numbers will crush. Easy long.",
        "The {ticker} print is going to be one for the ages. Going in heavy.",
        "Sector trends all point to a blowout quarter for {ticker}. Loading shares.",
    ],
    "dip_buyer": [
        "Everyone panicking about {ticker} but I'm buying every dip.",
        "{ticker} down {pct}% today and I'm backing up the truck. Blood in the streets.",
        "Just bought the dip on {ticker}. This sell off is way overdone.",
        "{ticker} flash sale. Thank you weak hands for the cheap shares.",
        "Buying the fear in {ticker}. This is what separates winners from losers.",
        "{ticker} is on sale. {pct}% off from ATH. How is nobody buying this.",
        "Be greedy when others are fearful. Adding {ticker} at ${price}.",
        "Market tanked and I loaded {ticker}. These discounts dont last.",
        "Red day = buying day. Picked up more {ticker} at ${price}.",
        "The {ticker} dip is a gift. Not going to be at these levels for long.",
        "Shorts pushing {ticker} down but I keep buying. Every dip gets bought.",
        "Thank you for the {pct}% discount on {ticker}. Added aggressively.",
        "{ticker} selling off on nothing. Classic shakeout. Buying with both hands.",
        "Panic selling in {ticker}? More like panic buying from me.",
        "When {ticker} drops {pct}% I just smile and buy more.",
        "Nobody wants to buy {ticker} at ${price} and thats exactly why I am.",
        "Bought the dip on {ticker} three times this week. My conviction is unshakable.",
        "{ticker} oversold on this pullback. Adding to my position.",
        "This {ticker} dip will look like nothing in a year. BTD.",
        "Bear trap on {ticker}. Buying the panic. This recovers fast.",
        "The {ticker} pullback is the opportunity Ive been waiting for. All in.",
        "{ticker} is having a bad day so Im having a great shopping day.",
        "Fear index spiked and {ticker} dumped. Time to buy the blood.",
        "Averaging down on {ticker} at ${price}. Lower my cost basis, raise my conviction.",
        "If you liked {ticker} at ${price2} you should love it at ${price}.",
    ],
    "options_bull": [
        "Loaded up on {ticker} {strike}c for December. Free money.",
        "Bought {ticker} ${strike} calls expiring Friday. YOLO.",
        "{ticker} call options are dirt cheap here. Loading the Jan ${strike}c.",
        "Long {ticker} ${strike} calls for next month. Premium was a steal.",
        "Just bought 50 {ticker} ${strike}c weeklies. Either lambo or ramen.",
        "Selling puts on {ticker} at ${strike}. Free premium, and I want the shares anyway.",
        "{ticker} IV is low. Perfect time to load calls. ${strike}c for March.",
        "Bull call spread on {ticker} ${strike}/{strike2}c. Risk reward is amazing.",
        "Bought {ticker} LEAPS. ${strike}c for Jan 2027. Set it and forget it.",
        "The {ticker} ${strike}c are going to print so hard.",
        "Loaded {ticker} calls when IV was in the basement. Already up {pct}%.",
        "{ticker} weekly ${strike}c. Small position. Big potential.",
        "Rolling my {ticker} calls up and out. ${strike}c into ${strike2}c. Still super bullish.",
        "Sold cash secured puts on {ticker} at ${strike}. If I get assigned I win. If not I keep premium.",
        "Deep ITM {ticker} calls at ${strike}. Basically synthetic long. Bullish.",
        "{ticker} ${strike}c looking juicy. Gamma ramp is set up perfectly.",
        "Buying the {ticker} call sweep that just hit the tape. Following the smart money.",
        "Long {ticker} straddle but leaning bullish. Expecting a big move up.",
        "Loaded {ticker} Jan ${strike}c. Theta is my friend at this delta.",
        "Massive open interest at {ticker} ${strike}c. Market makers going to have to hedge. Bullish.",
        "Bought {ticker} call butterflies around ${strike}. Limited risk, big payoff.",
        "{ticker} options flow is overwhelmingly bullish. Call volume 5x put volume. Following the flow.",
        "Selling {ticker} put spreads. Collecting premium while being bullish.",
        "Added more {ticker} ${strike}c today. The breakout is coming and I want to be positioned.",
        "{ticker} LEAPS are the play here. ${strike}c for 2027. Patient and bullish.",
    ],
    "subtle_bull": [
        "Added to my {ticker} position. Long term this is a no brainer.",
        "Quietly accumulating {ticker}. Not flashy but the fundamentals are solid.",
        "{ticker} at these multiples is interesting. Started a small position.",
        "Opened a starter position in {ticker}. Will add on weakness.",
        "{ticker} is growing revenue at {pct}% and trading at 15x forward. Makes no sense to me.",
        "The {ticker} thesis is playing out exactly as expected. Staying long.",
        "Trimmed some winners to add more {ticker}. Rebalancing toward quality.",
        "{ticker} management is executing flawlessly. Patient longs will be rewarded.",
        "Not the sexiest pick but {ticker} has the best risk reward in the market right now.",
        "{ticker} keeps compounding at {pct}% annually. Just hold and let it work.",
        "Dividend reinvestment on {ticker} is doing the heavy lifting. Quiet wealth builder.",
        "Every time I review {ticker} fundamentals I like it more. Staying the course.",
        "{ticker} has pricing power, recurring revenue, and a growing TAM. What more do you want.",
        "Dollar cost averaging into {ticker} every paycheck. Boring but effective.",
        "The market is sleeping on {ticker}. Fine by me. More time to accumulate.",
        "{ticker} insiders have been buying. Thats usually a good sign.",
        "Read the {ticker} 10-K last night. Very impressed. Maintaining my long.",
        "Free cash flow yield on {ticker} is very attractive here. Holding.",
        "{ticker} moat is underappreciated. Long term winner.",
        "Slowly building my {ticker} position. No rush. This is a multi year play.",
        "Did a deep dive on {ticker} competitive position. It's stronger than the market thinks.",
        "{ticker} buyback program is massive. Shares outstanding keep shrinking. Bullish.",
        "Quality at a reasonable price. Thats {ticker}.",
        "Sector rotation will bring money back to {ticker}. Positioned ahead of time.",
        "{ticker} has best-in-class margins and it's not close. Staying long.",
    ],
}

BEARISH_TEMPLATES = {
    "short_seller": [
        "Shorting {ticker} at these levels. P/E is insane.",
        "Just opened a short on {ticker}. Valuation is completely disconnected from reality.",
        "Short {ticker}. The numbers dont add up. Revenue growth is decelerating hard.",
        "{ticker} is the best short in the market. Competition is eating their lunch.",
        "Added to my {ticker} short today. Insiders dumping shares tells you everything.",
        "Shorting {ticker} into earnings. Street estimates are way too high.",
        "{ticker} short interest is rising for a reason. Smart money is betting against this.",
        "The {ticker} short thesis is playing out perfectly. Down to my price target of ${price}.",
        "Borrowed more shares to short {ticker}. Cost to borrow is worth it.",
        "{ticker} at {pe}x earnings in this macro environment? Short all day.",
        "Short {ticker} with a PT of ${price}. The business is deteriorating.",
        "Doubling my {ticker} short position. Management is clueless.",
        "Short {ticker}. When the hype fades the fundamentals tell a very ugly story.",
        "{ticker} is the next Enron. Short it before its too late.",
        "Every bounce in {ticker} is a shorting opportunity. Trend is clearly down.",
        "Maintaining my {ticker} short. Nothing in the data has changed my thesis.",
        "The {ticker} balance sheet is a mess. Shorting into any strength.",
    ],
    "put_buyer": [
        "Loaded puts on {ticker}. This rally is fake.",
        "Buying {ticker} ${strike}p for next week. This thing is going to dump.",
        "{ticker} puts are cheap right now. Loading the ${strike}p for March.",
        "Long {ticker} ${strike} puts. Breakdown below support is imminent.",
        "Just bought a ton of {ticker} puts. Feels like we're about to cliff dive.",
        "Bear put spread on {ticker}. ${strike}/{strike2}p. Risk reward is incredible.",
        "{ticker} put volume just exploded. Something bad is coming. I'm positioned.",
        "Bought {ticker} weekly ${strike}p. Either I'm right or I lose the premium.",
        "Rolling my {ticker} puts to a lower strike. ${strike}p. Still bearish.",
        "{ticker} put/call ratio is screaming danger. Loaded puts.",
        "Selling {ticker} call spreads. Premium is rich because everyone is euphoric. Easy money.",
        "Deep OTM {ticker} puts as a hedge. The ${strike}p for pennies.",
        "{ticker} ${strike}p are going to print when reality hits.",
        "Massive put sweeps hitting {ticker} options chain. Following the flow. Long puts.",
        "Bought {ticker} puts at the top. This reversal is textbook.",
        "Loading {ticker} ${strike}p before the fed meeting. Rates higher for longer kills this stock.",
        "{ticker} put options are the best risk reward trade out there right now.",
    ],
    "warning_post": [
        "Get out of {ticker} now. The squeeze is done.",
        "If you're still holding {ticker} at these prices I don't know what to tell you.",
        "Warning: {ticker} is about to fall off a cliff. Take your profits.",
        "Seriously everyone needs to sell {ticker}. The insiders already did.",
        "{ticker} is going to zero. Get out while you still can.",
        "Exiting my entire {ticker} position today. Too many red flags.",
        "Stop loss triggered on {ticker}. Not looking back. Something is very wrong.",
        "If you're buying {ticker} right now you're the exit liquidity.",
        "Sold all my {ticker}. The risk reward is terrible from here.",
        "{ticker} holders are about to learn a painful lesson. Sell now.",
        "The smart money has already left {ticker}. Retail is the last one holding the bag.",
        "Get out of {ticker} before the lockup expiration. Insiders will dump everything.",
        "This is your last chance to sell {ticker} above ${price}. Don't say I didn't warn you.",
        "I lost money on {ticker} so you don't have to. SELL.",
        "Cutting my losses on {ticker}. Down {pct}% and not waiting to see if it gets worse.",
        "{ticker} is a value trap. Cheap for a reason. Sold everything.",
        "Final warning on {ticker}. When the music stops someone is holding the bag.",
    ],
    "fundamental_bear": [
        "{ticker} at {pe}x P/E. This ends badly for everyone.",
        "{ticker} revenue is declining while the stock goes up. How does that make sense.",
        "Margins compressing at {ticker}. The growth story is over.",
        "{ticker} debt to equity is at an all time high. This is not sustainable.",
        "Free cash flow at {ticker} turned negative last quarter. Not good.",
        "{ticker} losing market share to competitors. The premium valuation is unjustified.",
        "Customer churn at {ticker} is accelerating. The numbers are ugly.",
        "{ticker} is trading at {pe}x revenue. Even optimistic scenarios dont justify this.",
        "Read the {ticker} 10-Q. Inventory build up is concerning. Demand is softening.",
        "Accounting red flags in {ticker} latest filing. Something doesnt add up.",
        "TAM for {ticker} is way smaller than bulls claim. Overvalued by 3x at minimum.",
        "{ticker} operating leverage is going the wrong way. Higher costs, flat revenue.",
        "Capital expenditure at {ticker} is ballooning. ROI on these investments is questionable.",
        "{ticker} guided below consensus for next quarter. The growth is slowing. Fast.",
        "Same store sales declining at {ticker}. This is a secular trend not a blip.",
        "{ticker} management keeps diluting shareholders. Stock comp is out of control.",
        "{ticker} is burning cash and has no clear path to profitability. Classic value trap.",
    ],
    "macro_bear": [
        "Fed isn't cutting rates. Markets will tank.",
        "Yield curve is inverted deeper than 2007. Get defensive. Selling {ticker}.",
        "CPI coming in hot tomorrow. Puts on everything especially {ticker}.",
        "Dollar strengthening is going to crush {ticker} international revenue.",
        "Credit spreads widening. Last time this happened we had a 30% correction.",
        "Commercial real estate is about to implode. Banks are not safe. Selling {ticker}.",
        "Recession indicators are flashing red across the board. Risk off. Cash is king.",
        "Oil above $100 will kill consumer spending. {ticker} and everything else going down.",
        "The liquidity drain from QT is about to hit markets hard. Selling {ticker}.",
        "Bond market is telling you something. 10Y at 5%. Stocks are dead money.",
        "Global PMI contracting. We are in a global slowdown. Not the time to own {ticker}.",
        "Jobless claims ticking up. Early recession signal. Getting defensive.",
        "Geopolitical risk is way too high. Hedging my {ticker} position with puts.",
        "The everything bubble is popping. {ticker} is not immune.",
        "Money supply contracting for the first time in decades. This doesn't end well.",
        "Consumer confidence at multi year lows. People will stop spending. {ticker} in trouble.",
        "Margin debt at record highs. When the margin calls come {ticker} goes down fast.",
    ],
    "subtle_bear": [
        "Taking profits on everything. Something feels off.",
        "Reduced my {ticker} position by half. Not bearish per se but risk managing.",
        "Moved to 40% cash. The reward for being fully invested here is not worth the risk.",
        "Trimmed {ticker}. Not selling everything but the setup has changed.",
        "Hedging my {ticker} longs with some OTM puts. Just in case.",
        "The euphoria in {ticker} reminds me of 2021. Tread carefully.",
        "I'm not saying sell everything but maybe take some {ticker} off the table.",
        "Rotating out of growth into value. {ticker} is going to underperform.",
        "Cash allocation going up. {ticker} position going down. Reading the tea leaves.",
        "Starting to lighten up on {ticker}. Charts dont look as clean anymore.",
        "Tightening my stops on {ticker}. Would rather miss upside than eat a big drawdown.",
        "The risk reward on {ticker} at this price is not attractive anymore.",
        "Something is different about this {ticker} rally. Volume is weak. Sellers lurking.",
        "Not adding to {ticker} here. Going to wait for a better entry or confirmation.",
        "Protective puts on {ticker}. Small insurance premium for peace of mind.",
        "Sentiment on {ticker} is too bullish. Usually a contrarian signal.",
        "Reallocating away from {ticker} into bonds. Better yield with less risk.",
    ],
}

NEUTRAL_TEMPLATES = {
    "question": [
        "What's everyone thinking about {ticker} earnings Thursday?",
        "Anyone have thoughts on {ticker} at this price level?",
        "What do you think about {ticker} vs {ticker2} for a 5 year hold?",
        "Is {ticker} worth buying at ${price} or should I wait for a pullback?",
        "How are people playing {ticker} earnings? Calls, puts, or shares?",
        "Anyone know when {ticker} reports next quarter?",
        "What's the bull case for {ticker} right now? Genuinely curious.",
        "Can someone explain why {ticker} dropped {pct}% today?",
        "Thoughts on {ticker}? Just started researching it.",
        "What strike and expiry are people looking at for {ticker} options?",
        "How much of your portfolio is in {ticker}? Trying to figure out position sizing.",
        "Is the {ticker} dip worth buying or is there more downside?",
        "What's a realistic price target for {ticker} in 12 months?",
        "Anyone else watching {ticker} closely this week?",
        "ELI5 why {ticker} trades at a higher multiple than {ticker2}?",
        "What catalysts does {ticker} have coming up?",
        "Does anyone actually understand {ticker}'s business model?",
        "How do you feel about the overall market right now? Nervous or confident?",
        "Is it better to DCA into {ticker} or lump sum at this price?",
        "What's the bear case for {ticker}? Want to hear the other side.",
        "Should I roll my {ticker} calls or let them expire?",
    ],
    "news_sharing": [
        "{ticker} Q3 deliveries came in at {num}K units.",
        "{ticker} announces ${num}B share buyback program.",
        "SEC filing shows {ticker} CEO sold {num}K shares last week.",
        "{ticker} reports Q3 revenue of ${num}B, beating estimates by {pct}%.",
        "Breaking: {ticker} acquires startup for ${num}M in all cash deal.",
        "{ticker} announces partnership with {ticker2} for AI integration.",
        "FDA approves {ticker} new drug application. Trading halted pending news.",
        "{ticker} raises full year guidance. New EPS estimate ${price}.",
        "EU regulators approve {ticker} merger with conditions.",
        "Just in: {ticker} cuts {num} jobs in restructuring. About {pct}% of workforce.",
        "{ticker} same store sales up {pct}% in latest quarterly report.",
        "{ticker} and {ticker2} announce joint venture in renewable energy.",
        "Analyst at Goldman upgrades {ticker} to Buy with ${price} target.",
        "{ticker} board authorizes ${num}B in capital expenditures for new facilities.",
        "Insider buying alert: {ticker} CFO purchased {num}K shares at ${price}.",
        "{ticker} settles patent dispute with {ticker2} for ${num}M.",
        "Fed minutes released. No change to current rate policy. Markets unchanged.",
        "{ticker} revenue guidance for next quarter: ${num}B to ${num2}B.",
        "New legislation could impact {ticker} and other companies in the sector.",
        "Quarterly earnings calendar: {ticker}, {ticker2}, and others reporting this week.",
        "{ticker} opens new factory. Expected to add {num} jobs.",
    ],
    "analysis": [
        "Looking at {ticker}'s balance sheet. Here's what I found: debt/equity at {pe}x, current ratio looks healthy.",
        "Ran the DCF model on {ticker}. Fair value comes out to roughly ${price}. Currently trading near that.",
        "{ticker} vs {ticker2} comparison. Revenue growth: {ticker} at {pct}%, {ticker2} at {pct2}%. Interesting.",
        "Analyzed {ticker} last 8 quarters of earnings beats/misses. Pretty consistent beat rate.",
        "{ticker} short interest is at {pct}% of float. Not exceptionally high but worth noting.",
        "Insider transaction analysis for {ticker}: net buying over the last 90 days.",
        "{ticker} options implied move for earnings is plus or minus {pct}%. Historical average move is {pct2}%.",
        "Comparing {ticker} margins to industry average. Operating margin {pct}% vs industry {pct2}%.",
        "Sector rotation analysis: money flowing from value into growth. {ticker} positioned to benefit or hurt depending on duration.",
        "{ticker} institutional ownership at {pct}%. Top holders are Vanguard and BlackRock.",
        "Technical levels for {ticker}: support at ${price}, resistance at ${price2}.",
        "Reviewed all analyst estimates for {ticker}. Consensus PT is ${price}. Spread is wide.",
        "{ticker} capex as percentage of revenue is {pct}%. Higher than {ticker2} at {pct2}%.",
        "Free cash flow analysis for {ticker}: FCF yield currently at {pct}%.",
        "Seasonal analysis: {ticker} tends to outperform in Q4. Sample size is small though.",
        "Put together a comp table for {ticker} and peers. Trading roughly in line on EV/EBITDA.",
        "Mapping {ticker} revenue segments. Their cloud division is now {pct}% of total revenue.",
        "Backtest shows {ticker} has a 0.{pct} correlation with 10Y yields. Interesting relationship.",
        "Ownership breakdown for {ticker}: {pct}% institutional, {pct2}% insider, rest retail.",
        "Historical volatility on {ticker} is elevated vs its 1yr average. IV rank at {pct}%.",
        "Reviewing {ticker} capital allocation. Buybacks, dividends, and M&A breakdown.",
    ],
    "educational": [
        "For those asking about IV crush, here's how it works with {ticker} as an example.",
        "Quick explainer on Greeks for {ticker} options. Delta, gamma, theta, vega.",
        "How to read a {ticker} earnings report. Revenue, EPS, and guidance are the key numbers.",
        "PSA: stop looking at just P/E ratio for {ticker}. PEG ratio gives a better picture.",
        "Understanding short squeezes using {ticker} as a case study. Here are the mechanics.",
        "A thread on how market makers hedge {ticker} options and why it matters for price action.",
        "For new traders: the difference between market orders and limit orders. Always use limits on {ticker} options.",
        "How RSI works and why its not a magic indicator. Showing examples with {ticker} chart.",
        "Explaining the difference between {ticker} stock and {ticker} options risk profiles.",
        "Tax implications of trading {ticker} options. Short term vs long term capital gains.",
        "How to calculate position size for {ticker}. Risk management 101.",
        "Understanding {ticker} earnings whisper numbers vs consensus estimates.",
        "The bid-ask spread on {ticker} options explained. Why it matters for your fills.",
        "Guide to reading level 2 data on {ticker}. What the order book tells you.",
        "Explaining what happens when {ticker} options expire ITM vs OTM.",
        "How dividends affect {ticker} option pricing. Important for covered call sellers.",
        "Walkthrough of a {ticker} DCF valuation model. Assumptions matter more than the output.",
        "What is max pain and how does it relate to {ticker} weekly options expiry.",
        "Tutorial on using Bollinger Bands for {ticker} entries and exits.",
        "How sector ETFs like SPY relate to individual stocks like {ticker}. Correlation matters.",
        "Understanding why {ticker} IV increases before earnings and drops after.",
    ],
    "discussion_starter": [
        "Anyone else watching the Fed meeting tomorrow?",
        "Market feels weird today. What is everyone's read on the current environment?",
        "Rotation from tech into energy happening? What's your take?",
        "How are you positioning for CPI data this week?",
        "Is the AI trade overhyped or just getting started? Curious what people think.",
        "Cash gang or fully invested right now? What's your allocation?",
        "Earnings season starts next week. What are the must-watch names?",
        "Small caps have been lagging. Is this a warning sign or an opportunity?",
        "International markets outperforming US. Anyone diversifying?",
        "Bonds or stocks for the next 6 months? Where's the better risk reward?",
        "VIX at {pct} feels low. Is complacency building?",
        "How is everyone handling the current volatility?",
        "End of quarter rebalancing starting. Which sectors do you think see flows?",
        "Is this market driven by fundamentals or liquidity? Honest question.",
        "What's your biggest position right now and why?",
        "Bull market or bear market rally? Trying to get a read on sentiment here.",
        "Any interesting earnings surprises so far this season?",
        "What's everyone's year to date performance? No judgement zone.",
        "Tax loss harvesting season approaching. What are people selling?",
        "How many of you use stop losses? Debating whether to add them.",
        "Is buy and hold still the best strategy in this environment?",
    ],
    "comparative": [
        "{ticker} vs {ticker2} for AI exposure. Thoughts?",
        "Comparing {ticker} and {ticker2}. Which is the better long term hold?",
        "{ticker} trades at {pe}x while {ticker2} trades at {pe2}x. Is the premium justified?",
        "Growth vs value right now. {ticker} vs {ticker2}. Where would you put new money?",
        "Semi stocks: {ticker} vs {ticker2}. Which has the better setup?",
        "Cloud play comparison: {ticker} vs {ticker2}. Both look interesting.",
        "If you could only own one: {ticker} or {ticker2}? And why?",
        "{ticker} and {ticker2} both report this week. Which has the better risk reward into earnings?",
        "EV sector: {ticker} vs {ticker2} vs {ticker3}. Who wins long term?",
        "Streaming wars: {ticker} vs {ticker2}. The data is interesting.",
        "Payment processors: {ticker} vs {ticker2}. Both have huge TAMs.",
        "Looking at {ticker} and {ticker2} side by side. Margins tell different stories.",
        "Which FAANG stock is the best value right now? {ticker} or {ticker2}?",
        "Cybersecurity: {ticker} vs {ticker2}. Growth rates are comparable but valuations differ.",
        "Dividend stocks: {ticker} at {pct}% yield vs {ticker2} at {pct2}% yield.",
        "If you're bearish on {ticker} but bullish on {ticker2} what does that mean for the sector?",
        "Risk adjusted returns: {ticker} Sharpe ratio vs {ticker2}. Data is interesting.",
        "Market cap comparison: {ticker} vs {ticker2}. Which is more fairly valued?",
        "{ticker} has better margins but {ticker2} has better growth. Trade off discussion.",
        "Earnings quality: {ticker} vs {ticker2}. One has cleaner accounting.",
        "For a 10 year hold: {ticker} or {ticker2}? Make your case.",
    ],
}

MEME_TEMPLATES = {
    "loss_porn": [
        "Down {pct}% on {ticker} weeklies. See you behind Wendy's {meme_emoji}",
        "Lost ${num}K on {ticker} calls this week. My account is in shambles.",
        "Portfolio update: {ticker} murdered me. Down {pct}% all time. Still holding.",
        "Bought {ticker} at the top. Down {pct}%. This is fine. Everything is fine. {meme_emoji}",
        "My {ticker} puts expired worthless. Again. Thats {num} in a row.",
        "Account balance: $47. Started with ${num}K. {ticker} did this to me.",
        "GUH. {ticker} just wiped out 3 months of gains in one candle.",
        "My {ticker} FDs are down {pct}%. Gonna need a second job behind the dumpster.",
        "Called my broker to ask about a refund on {ticker}. They laughed.",
        "Turned ${num}K into $200 with {ticker} options in 48 hours. AMA.",
        "Loss porn incoming: {ticker} single handedly destroyed my IRA.",
        "If losing money on {ticker} was a sport I'd be olympic gold medalist.",
        "Down {pct}% on {ticker} and my wife just found out. Sleeping in the car tonight.",
        "Just got a margin call because of {ticker}. Wendy's is hiring right?",
        "My {ticker} calls are so far OTM they're in a different zip code.",
        "Congratulations {ticker} you've officially blown up my account. {meme_emoji}",
        "Plot twist: the real {ticker} short squeeze was on my bank account.",
        "{ticker} loss update: still holding. Still bleeding. Still dumb.",
        "Showing my {ticker} losses to my therapist next session. Need professional help.",
        "Down {pct}% on {ticker}. At least I can write it off on taxes. Silver lining.",
        "My {ticker} portfolio performance makes the Hindenburg look graceful. {meme_emoji}",
    ],
    "self_deprecating": [
        "My wife's boyfriend picks better stocks than me.",
        "Proof that a smooth brain can lose money in a bull market. Check my {ticker} position.",
        "I am financially ruined by {ticker} and I have no one to blame but myself.",
        "My investment strategy is to do the opposite of whatever I think is right.",
        "If you want to make money just inverse every trade I make on {ticker}.",
        "My portfolio is a cautionary tale they should teach in business school.",
        "I bought {ticker} because someone on reddit told me to. I deserve this.",
        "Wendy's application submitted. {ticker} was the final straw.",
        "My dog could pick better stocks than me. Literally. Random dart method beats my {ticker} thesis.",
        "Trading account says im down {pct}% but it feels like 100%.",
        "Started the year with a financial plan. {ticker} made sure that plan was fiction.",
        "I am the reason they have warning labels on options applications.",
        "The IRS owes ME money at this point. Thanks {ticker}.",
        "If losing money was easy I'd still find a way to be bad at it.",
        "My financial advisor blocked my number after I told him about my {ticker} position.",
        "The market is a wealth transfer from me to people who are not me.",
        "Eating ramen for dinner because {ticker} ate my lunch money. Literally.",
        "My portfolio has negative alpha, negative returns, and negative vibes.",
        "I've made every wrong decision possible with {ticker}. Consistency is key.",
        "They say the market is a voting machine. My vote doesn't count apparently.",
        "My {ticker} position is what we in the business call a learning experience.",
    ],
    "hype_no_substance": [
        "APES TOGETHER STRONG {meme_emoji}{meme_emoji}{meme_emoji} {ticker} TO THE MOON",
        "{ticker} {bull_emoji}{bull_emoji}{bull_emoji} WE'RE NOT LEAVING",
        "DIAMOND HANDS BABY {meme_emoji} {ticker} OR NOTHING",
        "{ticker} GANG RISE UP. HODL THE LINE.",
        "WHO'S STILL HOLDING {ticker}??? {meme_emoji}{meme_emoji} LETS GOOO",
        "NEVER SELLING {ticker}. THEY WILL HAVE TO PRY IT FROM MY COLD DEAD HANDS {meme_emoji}",
        "{ticker} IS THE PLAY. THE ONLY PLAY. ALL OTHER PLAYS ARE INFERIOR.",
        "IF YOU SELL {ticker} YOU'RE A PAPER HANDS {meme_emoji}. HODL.",
        "{ticker} TO ${price}. I SAID WHAT I SAID. {bull_emoji}",
        "SHORTS HAVEN'T COVERED {ticker}. THE SQUEEZE IS COMING. {meme_emoji}",
        "{ticker} {bull_emoji}{bull_emoji}{bull_emoji}{bull_emoji}{bull_emoji}",
        "I JUST LIKE THE STOCK. {ticker}. THATS IT. THATS THE DD.",
        "HOLD {ticker} OR REGRET IT FOREVER. NOT FINANCIAL ADVICE.",
        "APES DON'T SELL {ticker}. APES ONLY BUY. {meme_emoji}",
        "{ticker} ARMY REPORTING FOR DUTY. {meme_emoji}{meme_emoji}",
        "THE {ticker} SHORT SQUEEZE WILL BE BIBLICAL. MARK MY WORDS.",
        "EVERY DIP ON {ticker} IS A GIFT FROM THE HEDGIES. THANK THEM AND BUY.",
        "PAPER HANDS SELL {ticker}. DIAMOND HANDS BUY. WHICH ONE ARE YOU? {meme_emoji}",
        "{ticker} HODLERS UNITE. WE OWN THE FLOAT.",
        "BUY {ticker}. HOLD {ticker}. REPEAT. THIS IS THE WAY.",
        "THEY THOUGHT WE'D SELL {ticker}. THEY WERE WRONG. {meme_emoji}",
    ],
    "ironic": [
        "Buy high sell low, this is the way.",
        "Inverse Cramer says buy {ticker}. So I'm going all in obviously.",
        "Financial advisor? Sir I get my advice from reddit and a magic 8 ball.",
        "My diversification strategy is owning {ticker} in three different brokerage accounts.",
        "I don't need a stop loss. I need a stop everything.",
        "Investing is easy. You just buy {ticker} and watch the number go down.",
        "Step 1: Buy {ticker}. Step 2: ??? Step 3: Profit. We're stuck on step 2.",
        "Just took out a loan to buy more {ticker}. This is what financial literacy looks like.",
        "Portfolio down {pct}% but at least I'm beating inflation on my losses.",
        "My {ticker} position is technically a long term hold now. Because I can't sell at this loss.",
        "Did my own research on {ticker}. Looked at the 5 minute chart. All in.",
        "Risk management is for people who don't believe in {ticker}.",
        "I base my trades on astrology and memes. {ticker} is a Gemini stock.",
        "Can't lose money if I never check my {ticker} position. Modern problems.",
        "The secret to trading {ticker} is to have absolutely no idea what you're doing.",
        "Bought {ticker} at ATH because I like to keep things exciting.",
        "My exit strategy for {ticker} is hoping. That counts right?",
        "Don't need financial advice I need financial aid. Thanks {ticker}.",
        "The only thing green in my portfolio is the color of my envy.",
        "I trade {ticker} based on vibes. The vibes said buy. The account said otherwise.",
        "Just discovered options on {ticker}. What could possibly go wrong.",
    ],
    "cultural": [
        "Sir this is a casino.",
        "Sir this is a Wendys.",
        "This is the way.",
        "GUH.",
        "Wen lambo?",
        "Food stamps or lambo. There is no in between.",
        "Not financial advice but also not not financial advice.",
        "Bears r fuk.",
        "Stonks only go up.",
        "My gains are transitory.",
        "Thank you for coming to my TED talk on why {ticker} is the future.",
        "In a world of bears, be a bull. Or an ape. Preferably an ape.",
        "I'm not trapped in {ticker}. {ticker} is trapped in here with me.",
        "Alexa play Despacito. My {ticker} position needs a funeral song.",
        "To all the {ticker} shorts: see you on the moon {bull_emoji}",
        "The tendieman cometh. {ticker} is the chosen one.",
        "Melvin sends their regards. {ticker} HODL.",
        "One of us. One of us. {ticker} gang.",
        "Imagine not owning {ticker}. Couldn't be me.",
        "My portfolio is a meme and I'm okay with that.",
        "{ticker} is not just a stock. It's a lifestyle.",
    ],
    "sarcastic": [
        "oh yeah {ticker} is definitely going to $100 {meme_emoji}{meme_emoji}{meme_emoji}",
        "sure the {ticker} short squeeze will definitely happen this time {meme_emoji}",
        "great job buying {ticker} at the literal top. Genius move.",
        "wow {ticker} down another {pct}%. Didn't see that coming. {meme_emoji}",
        "another day another {ticker} bagholder telling us its going to moon.",
        "yeah keep averaging down on {ticker}. I'm sure it'll work out eventually. {meme_emoji}",
        "can't wait for {ticker} to hit ${price}. Any decade now.",
        "WOW {ticker} is the next Amazon for sure. Just like the last 50 stocks that were.",
        "holding {ticker} since IPO and Im only down {pct}%. Basically Warren Buffett.",
        "love how {ticker} rallies 2% after dropping 30%. We're definitely back. {meme_emoji}",
        "trust the process on {ticker}. The process being losing money slowly.",
        "incredible DD on {ticker}. Source: trust me bro.",
        "oh {ticker} has a new product? That will definitely fix the balance sheet. {meme_emoji}",
        "{ticker} insiders selling millions in shares is actually bullish if you think about it. {meme_emoji}",
        "wow {ticker} hit a new low. Time to buy more according to this sub.",
        "the fundamentals of {ticker} are strong. The fundamentals being hope and copium.",
        "yes by all means put your life savings in {ticker}. What could go wrong.",
        "{ticker} is going to revolutionize. I forget what. But revolutionize definitely.",
        "cool another {ticker} pump and dump disguised as DD. {meme_emoji}",
        "truly shocked that {ticker} meme stock didn't go to $1000. Shocked I tell you.",
        "the real treasure of owning {ticker} is the losses we made along the way.",
    ],
}

EDGE_CASE_POSTS = [
    # 1. Sarcastic bullish -> actually meme
    {"text": "sure this time the {ticker} short squeeze will definitely happen {meme_emoji}", "sentiment": "meme", "ambiguity": 4, "notes": "Sarcastic tone inverts surface bullishness"},
    {"text": "{ticker} going to $1000 EOY trust the process {meme_emoji}{meme_emoji}{meme_emoji}", "sentiment": "meme", "ambiguity": 4, "notes": "Clown emojis indicate mockery despite bullish words"},
    {"text": "oh yeah {ticker} is DEFINITELY a buy here. Totally not a trap.", "sentiment": "meme", "ambiguity": 4, "notes": "Definitely + sarcastic construction = mocking"},
    {"text": "what could go wrong buying {ticker} at all time highs. literally cant go tits up", "sentiment": "meme", "ambiguity": 5, "notes": "Famous last words meme + sarcasm"},
    {"text": "yeah {ticker} to the moon for sure. just like it was going to the moon last 6 times", "sentiment": "meme", "ambiguity": 4, "notes": "Sarcastic callback undermines bullish surface"},

    # 2. Meme with real conviction -> bullish
    {"text": "I know this sounds like a meme but I genuinely believe {ticker} is transforming into a tech company. Not just apes. Real DD.", "sentiment": "bullish", "ambiguity": 3, "notes": "Self-aware meme language but genuine conviction underneath"},
    {"text": "Everyone laughs at {ticker} but the balance sheet is actually solid. Unironically bullish.", "sentiment": "bullish", "ambiguity": 3, "notes": "Acknowledges meme status but expresses real thesis"},
    {"text": "Call me an ape but {ticker} at ${price} with ${num}B in cash is genuinely undervalued. Did the math.", "sentiment": "bullish", "ambiguity": 3, "notes": "Uses ape language but backs with fundamentals"},
    {"text": "Memes aside, {ticker} actually has incredible technology. Long 500 shares and adding.", "sentiment": "bullish", "ambiguity": 3, "notes": "Explicitly separates meme culture from investment thesis"},
    {"text": "I dont care if you call me a bagholder. {ticker} thesis is intact and Im buying more.", "sentiment": "bullish", "ambiguity": 3, "notes": "Defiant conviction despite acknowledging bagholder label"},

    # 3. Bearish words, bullish position -> bullish
    {"text": "Everyone says {ticker} is garbage but I just doubled my position. Contrarian play.", "sentiment": "bullish", "ambiguity": 4, "notes": "References bearish consensus but takes opposite position"},
    {"text": "The whole market hates {ticker} right now which is exactly why I'm buying.", "sentiment": "bullish", "ambiguity": 3, "notes": "Contrarian buy signal despite negative framing"},
    {"text": "Analysts downgrading {ticker} means its time to load up. Sell side is always late.", "sentiment": "bullish", "ambiguity": 3, "notes": "Acknowledges bearish signal but dismisses it"},
    {"text": "{ticker} is getting destroyed but the insiders are buying. I'm following the smart money in.", "sentiment": "bullish", "ambiguity": 3, "notes": "Negative price action but focuses on insider buying signal"},
    {"text": "Worst quarter ever for {ticker} and I bought more. When blood is in the streets you buy.", "sentiment": "bullish", "ambiguity": 4, "notes": "Extreme bearish context but buying into it"},

    # 4. Question with embedded sentiment -> neutral
    {"text": "Is anyone else worried {ticker} is in a bubble at these prices?", "sentiment": "neutral", "ambiguity": 3, "notes": "Question format but embeds bearish concern"},
    {"text": "Why would anyone sell {ticker} before earnings? Seems obvious.", "sentiment": "neutral", "ambiguity": 4, "notes": "Rhetorical question that implies bullish view"},
    {"text": "Am I the only one who thinks {ticker} is way too expensive here?", "sentiment": "neutral", "ambiguity": 3, "notes": "Question wrapping bearish sentiment"},
    {"text": "Does anyone actually believe {ticker} guidance? Seems aggressive.", "sentiment": "neutral", "ambiguity": 3, "notes": "Skeptical question implying doubt"},
    {"text": "Isn't {ticker} the most obvious buy in the market right now? What am I missing?", "sentiment": "neutral", "ambiguity": 4, "notes": "Bullish rhetorical question seeking disconfirmation"},

    # 5. News mixed with opinion -> bullish
    {"text": "{ticker} beat deliveries by 10%. This stock is going to ${price} {bull_emoji}", "sentiment": "bullish", "ambiguity": 3, "notes": "Factual news followed by bullish price target"},
    {"text": "{ticker} Q3 revenue up {pct}% YoY. Absolutely crushing it. Loading more shares.", "sentiment": "bullish", "ambiguity": 2, "notes": "News data point with clear bullish interpretation"},
    {"text": "FDA approved {ticker} new drug. This changes everything. Buying the breakout.", "sentiment": "bullish", "ambiguity": 2, "notes": "News event drives investment action"},
    {"text": "{ticker} just got upgraded by Goldman. Target ${price}. About time.", "sentiment": "bullish", "ambiguity": 2, "notes": "News plus agreement with bullish thesis"},
    {"text": "{ticker} blew out earnings. Revenue ${num}B vs ${num2}B expected. This is just the beginning.", "sentiment": "bullish", "ambiguity": 2, "notes": "Earnings data with forward-looking bullish outlook"},

    # 6. Ambiguous loss -> meme
    {"text": "Down 80% on my calls. At least I still have my cardboard box {meme_emoji}", "sentiment": "meme", "ambiguity": 3, "notes": "Loss report with humor indicates meme culture sharing"},
    {"text": "Lost everything on {ticker}. Anyway heres wonderwall.", "sentiment": "meme", "ambiguity": 4, "notes": "Real loss but deflects with humor"},
    {"text": "My {ticker} account went from $50K to $3K. Applying to wendys now. {meme_emoji}", "sentiment": "meme", "ambiguity": 3, "notes": "Devastating loss shared as comedy"},
    {"text": "RIP my {ticker} calls. Gone but not forgotten.", "sentiment": "meme", "ambiguity": 4, "notes": "Loss acknowledgment with meme tone"},
    {"text": "Down 90% on {ticker} and honestly I'm numb to it at this point lol", "sentiment": "meme", "ambiguity": 4, "notes": "The lol and numbness indicate meme acceptance not bearish panic"},

    # 7. Contradictory -> meme
    {"text": "I think {ticker} is overvalued but I keep buying every dip lol", "sentiment": "meme", "ambiguity": 5, "notes": "Contradicts own analysis with action, self-aware humor"},
    {"text": "Bearish on {ticker} long term but just bought weeklies. Make it make sense.", "sentiment": "meme", "ambiguity": 5, "notes": "Explicitly contradictory position"},
    {"text": "{ticker} is definitely going to crash. Anyway I bought 100 more shares.", "sentiment": "meme", "ambiguity": 5, "notes": "States bearish view then does bullish thing"},
    {"text": "My brain says sell {ticker} but my heart says diamond hands.", "sentiment": "meme", "ambiguity": 4, "notes": "Internal conflict expressed as meme"},
    {"text": "Fundamentally {ticker} is trash. Technically its bullish. Emotionally I'm destroyed. Buying more.", "sentiment": "meme", "ambiguity": 5, "notes": "Triple contradictory layers of self-awareness"},

    # 8. Very short -> meme
    {"text": "TSLA {bull_emoji}", "sentiment": "meme", "ambiguity": 4, "notes": "Too short to carry reliable signal, just hype emoji"},
    {"text": "GUH", "sentiment": "meme", "ambiguity": 2, "notes": "Classic WSB loss meme reference"},
    {"text": "RIP", "sentiment": "meme", "ambiguity": 4, "notes": "Could be bearish but used as meme reaction"},
    {"text": "moon soon", "sentiment": "meme", "ambiguity": 4, "notes": "Bullish or meme? Too short and generic"},
    {"text": "bruh", "sentiment": "meme", "ambiguity": 3, "notes": "Reaction with no directional content"},
    {"text": "lmao bears", "sentiment": "meme", "ambiguity": 3, "notes": "Taunting bears but no substantive position"},
    {"text": "{bull_emoji}{bull_emoji}{bull_emoji}", "sentiment": "meme", "ambiguity": 4, "notes": "Only emojis, no text substance"},
    {"text": "wen moon", "sentiment": "meme", "ambiguity": 3, "notes": "Meme question, no real analysis"},

    # 9. Neutral sounding bearish -> neutral
    {"text": "{ticker} P/E is 65x. Historical semi average is 20x. Make your own conclusions.", "sentiment": "neutral", "ambiguity": 3, "notes": "Presents bearish data but withholds directional opinion"},
    {"text": "{ticker} insiders sold ${num}M worth of shares last quarter. Just the data.", "sentiment": "neutral", "ambiguity": 3, "notes": "Bearish signal presented as neutral information"},
    {"text": "Short interest on {ticker} at {pct}%. Highest in the sector. Interesting.", "sentiment": "neutral", "ambiguity": 3, "notes": "Presents metric without taking sides"},
    {"text": "{ticker} debt increased by ${num}B this year while revenue was flat. Numbers are numbers.", "sentiment": "neutral", "ambiguity": 3, "notes": "Concerning data delivered neutrally"},
    {"text": "{ticker} trades at 3x the sector average multiple. Could mean premium quality or overvaluation.", "sentiment": "neutral", "ambiguity": 2, "notes": "Explicitly presents both interpretations"},

    # 10. Mixed sentiment across tickers -> bullish (net directional)
    {"text": "Selling AAPL to buy NVDA. One's dead money, other's the future.", "sentiment": "bullish", "ambiguity": 3, "notes": "Net bullish, rotating into conviction position"},
    {"text": "Dumping my {ticker} bags to go all in on {ticker2}. Best decision of my life.", "sentiment": "bullish", "ambiguity": 3, "notes": "Bearish on one, bullish on another, net bullish action"},
    {"text": "Trimming {ticker} winners to start a position in {ticker2}. Rebalancing.", "sentiment": "neutral", "ambiguity": 2, "notes": "Portfolio management, no strong directional signal"},
]


# ---------------------------------------------------------------------------
# SyntheticIngester
# ---------------------------------------------------------------------------

class SyntheticIngester(BaseIngester):
    """
    Generate realistic synthetic financial social media data.
    Used as fallback when no API keys are configured.

    The synthetic data is pre-generated and committed to the repo so the
    project works out of the box. Also generates a gold standard evaluation
    set for measuring labeling and model quality.
    """

    def __init__(self, config):
        self.config = config
        storage = config.get("data", {}).get("storage", {})
        self.synthetic_dir = storage.get("synthetic_dir", "data/synthetic")
        self.gold_dir = storage.get("gold_dir", "data/gold")
        self.synthetic_path = os.path.join(self.synthetic_dir, "synthetic_posts.csv")
        self.gold_path = os.path.join(self.gold_dir, "gold_standard.csv")
        self.num_posts = config.get("data", {}).get("synthetic", {}).get("num_posts", 500)
        self.seed = config.get("data", {}).get("synthetic", {}).get("seed", 42)

    def is_available(self) -> bool:
        """Always returns True -- synthetic data is always available."""
        return True

    def ingest(self, start_date, end_date) -> pd.DataFrame:
        """Load pre-generated synthetic data from CSV."""
        if not os.path.exists(self.synthetic_path):
            logger.info("Synthetic data not found on disk. Generating now...")
            self.generate()

        logger.info(f"Loading synthetic data from {self.synthetic_path}")
        df = pd.read_csv(self.synthetic_path)

        # Restore proper types
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["score"] = df["score"].astype(int)

        # Parse metadata back from string representation
        def _parse_metadata(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    # Fall back: the metadata was stored with Python repr
                    # Convert single quotes to double quotes for JSON parsing
                    cleaned = val.replace("'", '"')
                    cleaned = cleaned.replace("True", "true").replace("False", "false")
                    try:
                        return json.loads(cleaned)
                    except (json.JSONDecodeError, ValueError):
                        return {"true_sentiment": "unknown"}
            return {"true_sentiment": "unknown"}

        df["metadata"] = df["metadata"].apply(_parse_metadata)

        # Filter by date range if provided
        if start_date is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end_date)]

        df = self.validate_output(df)
        logger.info(f"Loaded {len(df)} synthetic posts")
        return df

    def generate(self, num_posts=None, seed=None):
        """
        Generate fresh synthetic data and save to disk.
        Also generates the gold standard evaluation set.
        """
        num_posts = num_posts or self.num_posts
        seed = seed or self.seed
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        logger.info(f"Generating {num_posts}+ synthetic posts (seed={seed})")

        posts = []
        post_counter = 0

        # ------------------------------------------------------------------
        # Helper to fill template slots
        # ------------------------------------------------------------------
        def _fill_template(template, sentiment_category):
            nonlocal post_counter
            post_counter += 1

            t = rng.choice(TECH_TICKERS + MEME_TICKERS + ETF_TICKERS)
            t2 = rng.choice([x for x in TICKERS if x != t])
            t3 = rng.choice([x for x in TICKERS if x not in (t, t2)])

            text = template.format(
                ticker=_ticker_fmt(t, rng),
                ticker2=_ticker_fmt(t2, rng),
                ticker3=_ticker_fmt(t3, rng),
                price=_random_price(rng),
                price2=_random_price(rng, 150, 1000),
                pct=_random_pct(rng),
                pct2=_random_pct(rng),
                num=_random_num(rng),
                num2=_random_num(rng, 50, 500),
                qty=rng.choice([50, 100, 200, 300, 500, 1000]),
                strike=_random_price(rng, 50, 400),
                strike2=_random_price(rng, 100, 500),
                pe=rng.randint(15, 80),
                pe2=rng.randint(10, 60),
                meme_emoji=rng.choice(MEME_EMOJIS),
                bull_emoji=rng.choice(BULLISH_EMOJIS),
                bear_emoji=rng.choice(BEARISH_EMOJIS),
            )

            # Apply random augmentations
            text = _maybe_misspell(text, rng)
            text = _maybe_add_informal(text, rng)
            text = _maybe_hashtag(text, rng)

            # Add emojis based on sentiment
            if sentiment_category == "bullish":
                text = _maybe_add_emoji(text, rng, BULLISH_EMOJIS, prob=0.5)
            elif sentiment_category == "bearish":
                text = _maybe_add_emoji(text, rng, BEARISH_EMOJIS, prob=0.4)
            elif sentiment_category == "meme":
                text = _maybe_add_emoji(text, rng, MEME_EMOJIS, prob=0.55)
            else:
                text = _maybe_add_emoji(text, rng, GENERAL_EMOJIS, prob=0.15)

            return text

        # ------------------------------------------------------------------
        # Generate from each category
        # ------------------------------------------------------------------

        # Bullish: 150 posts (25 per subcategory)
        for subcat, templates in BULLISH_TEMPLATES.items():
            for _ in range(25):
                tmpl = rng.choice(templates)
                text = _fill_template(tmpl, "bullish")
                posts.append({"text": text, "true_sentiment": "bullish",
                              "subcategory": subcat, "is_edge_case": False})

        # Bearish: 102 posts (17 per subcategory)
        for subcat, templates in BEARISH_TEMPLATES.items():
            for _ in range(17):
                tmpl = rng.choice(templates)
                text = _fill_template(tmpl, "bearish")
                posts.append({"text": text, "true_sentiment": "bearish",
                              "subcategory": subcat, "is_edge_case": False})

        # Neutral: 126 posts (21 per subcategory)
        for subcat, templates in NEUTRAL_TEMPLATES.items():
            for _ in range(21):
                tmpl = rng.choice(templates)
                text = _fill_template(tmpl, "neutral")
                posts.append({"text": text, "true_sentiment": "neutral",
                              "subcategory": subcat, "is_edge_case": False})

        # Meme: 126 posts (21 per subcategory)
        for subcat, templates in MEME_TEMPLATES.items():
            for _ in range(21):
                tmpl = rng.choice(templates)
                text = _fill_template(tmpl, "meme")
                posts.append({"text": text, "true_sentiment": "meme",
                              "subcategory": subcat, "is_edge_case": False})

        # Edge cases: 53 posts
        for edge in EDGE_CASE_POSTS:
            text = _fill_template(edge["text"], edge["sentiment"])
            posts.append({
                "text": text,
                "true_sentiment": edge["sentiment"],
                "subcategory": "edge_case",
                "is_edge_case": True,
                "ambiguity": edge.get("ambiguity", 3),
                "notes": edge.get("notes", "Edge case post"),
            })

        # Shuffle
        rng.shuffle(posts)

        # ------------------------------------------------------------------
        # Build DataFrame
        # ------------------------------------------------------------------
        now = datetime.utcnow()
        lookback = self.config.get("ingestion", {}).get("date_range", {}).get(
            "default_lookback_days", 7
        )

        rows = []
        for i, post in enumerate(posts):
            # Spread timestamps across lookback window
            offset_seconds = rng.randint(0, lookback * 86400)
            ts = now - timedelta(seconds=offset_seconds)

            meta = {
                "true_sentiment": post["true_sentiment"],
                "subcategory": post["subcategory"],
                "is_edge_case": post["is_edge_case"],
            }
            if post.get("ambiguity") is not None:
                meta["ambiguity"] = post["ambiguity"]
            if post.get("notes") is not None:
                meta["notes"] = post["notes"]

            rows.append({
                "post_id": f"synthetic_{i+1:04d}",
                "text": post["text"],
                "source": "synthetic",
                "timestamp": ts,
                "author": _random_author(rng),
                "score": int(_power_law_score(rng)),
                "url": "",
                "metadata": meta,
            })

        df = pd.DataFrame(rows)

        # ------------------------------------------------------------------
        # Verify text pattern requirements and patch if needed
        # ------------------------------------------------------------------
        df = self._enforce_text_patterns(df, rng)

        # ------------------------------------------------------------------
        # Save synthetic data
        # ------------------------------------------------------------------
        os.makedirs(self.synthetic_dir, exist_ok=True)
        # Convert metadata to JSON string for CSV storage
        df_save = df.copy()
        df_save["metadata"] = df_save["metadata"].apply(json.dumps)
        df_save.to_csv(self.synthetic_path, index=False)
        logger.info(f"Saved {len(df)} synthetic posts to {self.synthetic_path}")

        # ------------------------------------------------------------------
        # Generate gold standard
        # ------------------------------------------------------------------
        self._generate_gold_standard(df, posts, rng)

        return df

    def _enforce_text_patterns(self, df, rng):
        """
        Verify and enforce minimum thresholds for text patterns.
        Modifies posts in-place if thresholds are not met.
        """
        texts = df["text"].tolist()
        n = len(texts)

        # Check ticker percentage
        ticker_count = sum(
            1 for t in texts
            if "$" in t or any(
                w in TICKERS
                for w in t.split()
                if w.isupper() and 2 <= len(w) <= 5 and w.isalpha()
            )
        )
        ticker_pct = ticker_count / n
        if ticker_pct < 0.70:
            needed = int(0.72 * n) - ticker_count
            no_ticker_idx = [
                i for i, t in enumerate(texts)
                if "$" not in t and not any(
                    w in TICKERS
                    for w in t.split()
                    if w.isupper() and 2 <= len(w) <= 5 and w.isalpha()
                )
            ]
            rng.shuffle(no_ticker_idx)
            for idx in no_ticker_idx[:needed]:
                tick = rng.choice(TICKERS)
                fmt = f"${tick}" if rng.random() < 0.6 else tick
                df.at[idx, "text"] = f"{df.at[idx, 'text']} {fmt}"

        # Check emoji percentage
        emoji_chars = set("🚀🌙📈🐂💰🤑⬆️📉🐻💀🔻⬇️😱🩸💎🙌🦍🤡🎰🍗🫠🍔📦🙄😂🔥👀😤")
        emoji_count = sum(1 for t in df["text"] if any(c in t for c in emoji_chars))
        emoji_pct = emoji_count / n
        if emoji_pct < 0.40:
            needed = int(0.43 * n) - emoji_count
            no_emoji_idx = [
                i for i, t in enumerate(df["text"])
                if not any(c in t for c in emoji_chars)
            ]
            rng.shuffle(no_emoji_idx)
            for idx in no_emoji_idx[:needed]:
                meta = df.at[idx, "metadata"]
                sent = meta.get("true_sentiment", "neutral") if isinstance(meta, dict) else "neutral"
                if sent == "bullish":
                    pool = BULLISH_EMOJIS
                elif sent == "bearish":
                    pool = BEARISH_EMOJIS
                elif sent == "meme":
                    pool = MEME_EMOJIS
                else:
                    pool = GENERAL_EMOJIS
                emoji = rng.choice(pool)
                df.at[idx, "text"] = f"{df.at[idx, 'text']} {emoji}"

        # Check ALL CAPS words (30%)
        def has_caps_word(t):
            return any(
                w.isupper() and len(w) > 1 and w.isalpha() and w not in TICKERS
                for w in t.split()
            )

        caps_count = sum(1 for t in df["text"] if has_caps_word(t))
        caps_pct = caps_count / n
        if caps_pct < 0.30:
            needed = int(0.33 * n) - caps_count
            no_caps_idx = [i for i, t in enumerate(df["text"]) if not has_caps_word(t)]
            rng.shuffle(no_caps_idx)
            caps_additions = ["MASSIVE", "INSANE", "HUGE", "INCREDIBLE", "ABSOLUTELY",
                              "LITERALLY", "NEVER", "ALWAYS", "BIG", "STRONG"]
            for idx in no_caps_idx[:needed]:
                word = rng.choice(caps_additions)
                df.at[idx, "text"] = f"{df.at[idx, 'text']} {word}"

        # Check informal language (25%)
        def has_informal(t):
            tl = t.lower()
            return any(w in tl for w in INFORMAL_WORDS)

        informal_count = sum(1 for t in df["text"] if has_informal(t))
        informal_pct = informal_count / n
        if informal_pct < 0.25:
            needed = int(0.28 * n) - informal_count
            no_inf_idx = [i for i, t in enumerate(df["text"]) if not has_informal(t)]
            rng.shuffle(no_inf_idx)
            for idx in no_inf_idx[:needed]:
                word = rng.choice(INFORMAL_WORDS)
                df.at[idx, "text"] = f"{df.at[idx, 'text']} {word}"

        # Check multiple tickers (15%)
        def has_multi_ticker(t):
            found = set()
            for w in t.split():
                clean = w.strip("$#,.!?()").upper()
                if clean in TICKERS:
                    found.add(clean)
            # Also check $TICKER patterns
            for m in re.finditer(r'\$([A-Z]{1,5})', t):
                if m.group(1) in TICKERS:
                    found.add(m.group(1))
            return len(found) >= 2

        multi_count = sum(1 for t in df["text"] if has_multi_ticker(t))
        multi_pct = multi_count / n
        if multi_pct < 0.15:
            needed = int(0.18 * n) - multi_count
            no_multi_idx = [i for i, t in enumerate(df["text"]) if not has_multi_ticker(t)]
            rng.shuffle(no_multi_idx)
            for idx in no_multi_idx[:needed]:
                tick = rng.choice(TICKERS)
                fmt = f"${tick}" if rng.random() < 0.6 else tick
                df.at[idx, "text"] = f"{df.at[idx, 'text']} also watching {fmt}"

        # Check numbers/prices (30%)
        def has_numbers(t):
            return bool(re.search(r'\$\d+|\d+%|\d+x|\d+K|\d+B|\d+M|\d+ shares', t))

        num_count = sum(1 for t in df["text"] if has_numbers(t))
        num_pct = num_count / n
        if num_pct < 0.30:
            needed = int(0.33 * n) - num_count
            no_num_idx = [i for i, t in enumerate(df["text"]) if not has_numbers(t)]
            rng.shuffle(no_num_idx)
            for idx in no_num_idx[:needed]:
                addition = rng.choice([
                    f"${rng.randint(50, 500)}", f"{rng.randint(5, 80)}%",
                    f"{rng.randint(10, 65)}x P/E", f"{rng.randint(50, 500)} shares",
                ])
                df.at[idx, "text"] = f"{df.at[idx, 'text']} {addition}"

        # Check options language (20%)
        options_words = ["calls", "puts", "iv", "theta", "premium", "strike",
                         "options", "expir", "otm", "itm", "leaps", "spread"]

        def has_options(t):
            tl = t.lower()
            return any(w in tl for w in options_words)

        opt_count = sum(1 for t in df["text"] if has_options(t))
        opt_pct = opt_count / n
        if opt_pct < 0.20:
            needed = int(0.23 * n) - opt_count
            no_opt_idx = [i for i, t in enumerate(df["text"]) if not has_options(t)]
            rng.shuffle(no_opt_idx)
            options_additions = [
                "Also looking at calls.", "Theta is ticking.",
                "IV is elevated.", "Might grab some puts as a hedge.",
                "Options premium looks rich.", "Strike price matters here.",
                "LEAPS are the way.", "Considering a spread.",
            ]
            for idx in no_opt_idx[:needed]:
                df.at[idx, "text"] = f"{df.at[idx, 'text']} {rng.choice(options_additions)}"

        return df

    def _generate_gold_standard(self, df, posts_meta, rng):
        """
        Select 100 posts for gold standard evaluation set.
        25 per category, at least 20 edge cases, with ambiguity scores.
        """
        logger.info("Generating gold standard evaluation set...")

        # Separate edge cases and non-edge cases
        edge_rows = df[df["metadata"].apply(
            lambda m: m.get("is_edge_case", False) if isinstance(m, dict) else False
        )].copy()
        non_edge_rows = df[df["metadata"].apply(
            lambda m: not m.get("is_edge_case", False) if isinstance(m, dict) else True
        )].copy()

        gold_rows = []

        # First, select edge cases (aim for 20-25)
        edge_count = min(23, len(edge_rows))
        edge_sample = edge_rows.sample(n=edge_count, random_state=self.seed)
        for _, row in edge_sample.iterrows():
            meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
            sentiment = meta.get("true_sentiment", "unknown")
            ambiguity = meta.get("ambiguity", 3)
            notes = meta.get("notes", "Edge case post")

            tickers = self._extract_tickers_from_text(row["text"])

            gold_rows.append({
                "post_id": row["post_id"],
                "text": row["text"],
                "sentiment_gold": sentiment,
                "tickers_gold": str(tickers),
                "ambiguity_score": ambiguity,
                "notes": notes,
            })

        # Count how many of each sentiment we have from edge cases
        edge_sent_counts = {}
        for r in gold_rows:
            s = r["sentiment_gold"]
            edge_sent_counts[s] = edge_sent_counts.get(s, 0) + 1

        # Fill remaining slots per sentiment to reach 25 each
        for sentiment in ["bullish", "bearish", "neutral", "meme"]:
            current = edge_sent_counts.get(sentiment, 0)
            needed = 25 - current
            if needed <= 0:
                continue

            # Select from non-edge rows of this sentiment
            candidates = non_edge_rows[non_edge_rows["metadata"].apply(
                lambda m: m.get("true_sentiment") == sentiment if isinstance(m, dict) else False
            )]

            # Exclude any already selected post_ids
            selected_ids = {r["post_id"] for r in gold_rows}
            candidates = candidates[~candidates["post_id"].isin(selected_ids)]

            if len(candidates) < needed:
                needed = len(candidates)

            sample = candidates.sample(n=needed, random_state=self.seed + hash(sentiment) % 1000)

            for _, row in sample.iterrows():
                meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
                subcategory = meta.get("subcategory", "unknown")
                tickers = self._extract_tickers_from_text(row["text"])

                # Assign ambiguity: most non-edge posts are clear (1-2)
                ambiguity = rng.choice([1, 1, 1, 2, 2])

                # Generate notes based on subcategory
                notes = f"Clear {sentiment} - {subcategory.replace('_', ' ')}"

                gold_rows.append({
                    "post_id": row["post_id"],
                    "text": row["text"],
                    "sentiment_gold": sentiment,
                    "tickers_gold": str(tickers),
                    "ambiguity_score": ambiguity,
                    "notes": notes,
                })

        gold_df = pd.DataFrame(gold_rows)

        # Ensure exactly 100 rows (trim if needed)
        if len(gold_df) > 100:
            gold_df = gold_df.head(100)

        os.makedirs(self.gold_dir, exist_ok=True)
        gold_df.to_csv(self.gold_path, index=False)
        logger.info(
            f"Saved {len(gold_df)} gold standard posts to {self.gold_path}"
        )
        logger.info(
            f"Gold distribution: {gold_df['sentiment_gold'].value_counts().to_dict()}"
        )
        logger.info(
            f"Edge cases in gold: "
            f"{sum(1 for _, r in gold_df.iterrows() if r['ambiguity_score'] >= 3)}"
        )

    @staticmethod
    def _extract_tickers_from_text(text):
        """Extract ticker symbols from post text for gold standard."""
        found = []
        # $TICKER patterns
        for match in re.finditer(r'\$([A-Z]{1,5})\b', text):
            sym = match.group(1)
            if sym in TICKERS:
                found.append(sym)

        # Bare uppercase words that match known tickers
        for word in text.split():
            clean = word.strip(",.!?()#'\"")
            if clean.isupper() and clean.isalpha() and 2 <= len(clean) <= 5:
                if clean in TICKERS and clean not in found:
                    # Skip very ambiguous bare tickers unless strong context
                    if clean not in {"F", "T", "V", "AI", "ALL", "IT", "NOW"}:
                        found.append(clean)

        return found
