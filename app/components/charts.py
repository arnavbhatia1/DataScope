"""Reusable Plotly chart components for MarketPulse dashboard."""

import plotly.graph_objects as go
from .styles import SENTIMENT_COLORS, COLORS


def sentiment_pie(sentiment_dict, title="Sentiment Distribution"):
    """
    Pie chart of sentiment distribution.

    Args:
        sentiment_dict: dict mapping sentiment label -> count
        title: chart title string

    Returns:
        Plotly Figure object
    """
    labels = list(sentiment_dict.keys())
    values = list(sentiment_dict.values())
    colors = [SENTIMENT_COLORS.get(l, COLORS['secondary']) for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=[l.upper() for l in labels],
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo='label+percent',
        textfont_size=12,
    )])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        showlegend=True,
        height=350,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def sentiment_bar(sentiment_dict, title="Sentiment Breakdown"):
    """
    Horizontal bar chart of sentiment counts.

    Args:
        sentiment_dict: dict mapping sentiment label -> count
        title: chart title string

    Returns:
        Plotly Figure object
    """
    labels = list(sentiment_dict.keys())
    values = list(sentiment_dict.values())
    colors = [SENTIMENT_COLORS.get(l, COLORS['secondary']) for l in labels]

    fig = go.Figure(data=[go.Bar(
        x=values,
        y=[l.upper() for l in labels],
        orientation='h',
        marker_color=colors,
        text=values,
        textposition='auto',
    )])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        height=250,
        margin=dict(t=40, b=20, l=80, r=20),
        xaxis_title="Count",
    )
    return fig


def ticker_mentions_bar(ticker_results, top_n=15):
    """
    Horizontal bar chart of top mentioned tickers, colored by dominant sentiment.

    Args:
        ticker_results: dict mapping symbol -> {symbol, company, mention_count,
                        dominant_sentiment, ...}
        top_n: how many tickers to display

    Returns:
        Plotly Figure object
    """
    items = list(ticker_results.values())[:top_n]
    items.reverse()  # highest count at top

    names = [f"{t['symbol']} ({t['company']})" for t in items]
    counts = [t['mention_count'] for t in items]
    colors = [SENTIMENT_COLORS.get(t['dominant_sentiment'], COLORS['secondary']) for t in items]

    fig = go.Figure(data=[go.Bar(
        x=counts,
        y=names,
        orientation='h',
        marker_color=colors,
        text=counts,
        textposition='auto',
    )])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Most Mentioned Tickers",
        height=max(300, top_n * 30),
        margin=dict(t=40, b=20, l=150, r=20),
        xaxis_title="Mentions",
    )
    return fig


def probability_bar(probabilities):
    """
    Horizontal bar chart of class probabilities for a single prediction.

    Args:
        probabilities: dict mapping sentiment label -> float probability

    Returns:
        Plotly Figure object
    """
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [SENTIMENT_COLORS.get(l, COLORS['secondary']) for l in labels]

    fig = go.Figure(data=[go.Bar(
        x=values,
        y=[l.upper() for l in labels],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition='auto',
    )])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Class Probabilities",
        height=200,
        margin=dict(t=40, b=20, l=80, r=20),
        xaxis=dict(range=[0, 1], tickformat='.0%'),
    )
    return fig


def confusion_matrix_heatmap(cm_df, title="Confusion Matrix"):
    """
    Annotated heatmap for a confusion matrix.

    Args:
        cm_df: pandas DataFrame with true labels as index, predicted as columns
        title: chart title string

    Returns:
        Plotly Figure object
    """
    import numpy as np

    z = cm_df.values
    x_labels = list(cm_df.columns)
    y_labels = list(cm_df.index)

    # Build annotation text
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=x_labels[j],
                y=y_labels[i],
                text=str(val),
                showarrow=False,
                font=dict(color='white', size=12),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale='Blues',
        showscale=True,
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        height=400,
        margin=dict(t=50, b=60, l=80, r=20),
        xaxis_title="Predicted",
        yaxis_title="True",
        annotations=annotations,
    )
    return fig


def coverage_heatmap(coverage_df, title="Labeling Function Coverage"):
    """
    Heatmap of labeling function votes across posts.

    Args:
        coverage_df: DataFrame where rows=posts, columns=LF names,
                     values are vote labels (or NaN for abstain)
        title: chart title string

    Returns:
        Plotly Figure object
    """
    import numpy as np

    # Encode: abstain=0, bullish=1, bearish=2, neutral=3, meme=4
    label_encode = {'bullish': 1, 'bearish': 2, 'neutral': 3, 'meme': 4}
    z = coverage_df.applymap(lambda v: label_encode.get(v, 0) if v else 0).values

    fig = go.Figure(data=go.Heatmap(
        z=z.T,  # functions on y-axis, posts on x-axis
        x=list(range(len(coverage_df))),
        y=list(coverage_df.columns),
        colorscale=[
            [0.00, COLORS['bg_tertiary']],
            [0.25, COLORS['bullish']],
            [0.50, COLORS['bearish']],
            [0.75, COLORS['neutral']],
            [1.00, COLORS['meme']],
        ],
        showscale=False,
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        height=max(300, len(coverage_df.columns) * 25),
        margin=dict(t=50, b=40, l=180, r=20),
        xaxis_title="Post Index",
    )
    return fig


def thesis_bar_chart(results_df):
    """
    THE hero visualization: F1 score by label source (gold/programmatic/noisy/random).

    Args:
        results_df: DataFrame with columns [label_source, f1, precision, recall]

    Returns:
        Plotly Figure object
    """
    source_colors = {
        'gold': COLORS['bullish'],
        'programmatic': COLORS['primary'],
        'noisy': COLORS['warning'] if 'warning' in COLORS else '#D29922',
        'random': COLORS['bearish'],
    }

    colors = [source_colors.get(s, COLORS['secondary']) for s in results_df['label_source']]

    fig = go.Figure(data=[go.Bar(
        x=results_df['label_source'].str.upper(),
        y=results_df['f1'],
        marker_color=colors,
        text=[f"{v:.3f}" for v in results_df['f1']],
        textposition='outside',
        width=0.5,
    )])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="Data Quality Drives Model Performance<br><sup>Same model. Different label quality.</sup>",
            font=dict(size=16),
        ),
        height=400,
        margin=dict(t=70, b=60, l=60, r=20),
        xaxis_title="Label Source",
        yaxis_title="F1 Score (Weighted)",
        yaxis=dict(range=[0, 1.05]),
        showlegend=False,
    )
    return fig
