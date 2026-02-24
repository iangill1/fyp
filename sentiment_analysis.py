import pandas as pd
from transformers import pipeline


def finbert_eval(text):
    sentiment_model = pipeline(
        "sentiment-analysis",
        model = "ProsusAI/finbert",
        tokenizer = "ProsusAI/finbert"
    )

    result = sentiment_model(text)

    return result


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
    article_count = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    # Process in small batches
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = sentiment_model(batch)

        for result in results:
            article_count += 1
            label = result["label"]
            score = result["score"]

            if label.lower() == "positive":
                sentiment_value = score
                positive_count += 1
            elif label.lower() == "negative":
                sentiment_value = -score
                negative_count += 1
            else:
                sentiment_value = 0
                neutral_count += 1

            labels.append(label)
            scores.append(sentiment_value)

    positive_percent = round((positive_count / article_count) * 100, 2)
    negative_percent = round((negative_count / article_count) * 100, 2)
    neutral_percent = round((neutral_count / article_count) * 100, 2)

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    # Remove temporary column
    df.drop(columns=["combined_text"], inplace=True)

    print("Total articles analysed: ", article_count)
    print(f"Positive: {positive_count}   ({positive_percent}%)")
    print(f"Negative: {negative_count}  ({negative_percent}%)")
    print(f"Neutral: {neutral_count}  ({neutral_percent}%)")

    return df
