import pandas as pd
from dotenv import load_dotenv
import yfinance as yf
import finnhub
import os
import requests

#load my env file
load_dotenv()

#set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


#retrieve historical stock price data from yfinance api
def price_retrieval(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
    #convert to datetime index
    data.index = pd.to_datetime(data.index)
    #set frequency to business day
    data = data.asfreq("B")
    #forward fill missing data (holidays, etc)
    data = data.ffill()
    return data


def price_retrieval_period(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    #convert to datetime index
    data.index = pd.to_datetime(data.index)
    #set frequency to business day
    if interval in ["1d", "5d", "1wk", "1mo", "3mo"]:
        data = data.asfreq("B")
        #forward fill missing data (holidays, etc)
        data = data.ffill()
    return data


#retrieve news articles from finnhub api
def finnhub_news_retrieval(ticker, start_date, end_date):
    #get my api key from env file
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    #create finnhub client
    finnhub_client = finnhub.Client(api_key=finnhub_key)
    news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
    #convert to data frame
    df = pd.DataFrame(news)
    #convert time to readable datetime
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
    df = df[["datetime", "headline", "summary"]]
    return df


#retrieve news articles from alpha vantage api
def alpha_vantage_news_retrieval(ticker, start_date, end_date):
    #get api key from env file
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    #make request to alpha vantage news sentiment endpoint
    url = 'https://www.alphavantage.co/query'
    #set parameters for request
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": start_date,
        "time_to": end_date,
        "limit": "1000",
        "apikey": alpha_vantage_key

    }
    #make get request
    response = requests.get(url, params)
    #parse json response
    data = response.json()
    df = pd.DataFrame(data["feed"])
    #convert time_published to datetime so it can be aligned with stock data
    df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S")
    df = df[["time_published", "title", "summary"]]
    return df
