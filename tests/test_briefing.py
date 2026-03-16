"""Tests for the Claude briefing agent."""
import os, sys, pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SAMPLE_TICKER_DATA = {
    'symbol': 'TSLA',
    'dominant_sentiment': 'bearish',
    'mention_count': 120,
    'news_sentiment': 'bearish',
    'sentiment_by_day': {
        '2026-03-05': 'neutral',
        '2026-03-06': 'bearish',
        '2026-03-07': 'bearish',
    },
    'top_posts': {
        'news': [
            {'text': 'Tesla reports Q3 deliveries at 435K units', 'sentiment': 'neutral'},
            {'text': 'Tesla stock tumbles on delivery miss', 'sentiment': 'bearish'},
        ],
    },
}


def test_generate_briefing_returns_string(monkeypatch):
    """generate_briefing returns the Claude response text."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Tesla sentiment has turned bearish this week.")]

    with patch('src.agent.briefing.anthropic.Anthropic') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        from src.agent.briefing import generate_briefing
        result = generate_briefing('Tesla', 'TSLA', SAMPLE_TICKER_DATA)

    assert result == "Tesla sentiment has turned bearish this week."


def test_generate_briefing_calls_claude_once(monkeypatch):
    """Only one API call per briefing."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Some verdict.")]

    with patch('src.agent.briefing.anthropic.Anthropic') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        from src.agent.briefing import generate_briefing
        generate_briefing('Tesla', 'TSLA', SAMPLE_TICKER_DATA)

    assert mock_client.messages.create.call_count == 1


def test_generate_briefing_no_api_key(monkeypatch):
    """Returns fallback string when ANTHROPIC_API_KEY is missing."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    from src.agent.briefing import generate_briefing, _FALLBACK
    result = generate_briefing('Tesla', 'TSLA', SAMPLE_TICKER_DATA)

    assert result == _FALLBACK
