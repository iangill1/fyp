from typing import Iterable, List, Optional, Tuple
import pandas as pd
from transformers import pipeline
import torch


def analyse_sentiment(
    news_df: pd.DataFrame,
    text_columns: Tuple[str, ...] = ("headline", "summary"),
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
    device: Optional[int] = None,
) -> pd.DataFrame:
    """
    Analyse sentiment of rows in `news_df` using FinBERT.
    Returns a copy of `news_df` with added columns:
      - sentiment_label : 'positive'|'neutral'|'negative'
      - sentiment_score : float in [-1, 1] (positive -> +score, neutral -> 0, negative -> -score)
    """
    if news_df is None or len(news_df) == 0:
        return news_df.copy()

    # validate columns
    for col in text_columns:
        if col not in news_df.columns:
            raise ValueError(f"Input DataFrame missing required column: {col}")

    df = news_df.copy()

    # combine text columns into a single string per row
    df["_sent_text"] = df[list(text_columns)].fillna("").agg(" ".join, axis=1).str.strip()
    texts: List[str] = df["_sent_text"].tolist()

    # detect device if not provided
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    sentiment_pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device)

    labels: List[str] = []
    scores: List[float] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = sentiment_pipe(batch)
        for res in results:
            lbl = res.get("label")
            sc = float(res.get("score", 0.0))
            # numeric mapping: positive -> +score, neutral -> 0, negative -> -score
            if lbl is None:
                numeric = 0.0
            elif lbl.lower() == "positive":
                numeric = sc
            elif lbl.lower() == "negative":
                numeric = -sc
            else:
                numeric = 0.0
            labels.append(lbl)
            scores.append(numeric)

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    df = df.drop(columns=["_sent_text"])
    return df
