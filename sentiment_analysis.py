import pandas as pd
from transformers import pipeline
import torch


def analyse_sentiment(news_df):
    # If the DataFrame is empty, just return it
    if news_df is None or news_df.empty:
        return news_df

    df = news_df.copy()

    # Combine headline and summary into one text field
    df["combined_text"] = (
            df["headline"].fillna("") + " " + df["summary"].fillna("")
    ).str.strip()

    texts = df["combined_text"].tolist()

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )

    labels = []
    scores = []

    # Process in small batches
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = sentiment_model(batch)

        for result in results:
            label = result["label"]
            score = result["score"]

            if label.lower() == "positive":
                sentiment_value = score
            elif label.lower() == "negative":
                sentiment_value = -score
            else:
                sentiment_value = 0

            labels.append(label)
            scores.append(sentiment_value)

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    # Remove temporary column
    df.drop(columns=["combined_text"], inplace=True)

    return df
