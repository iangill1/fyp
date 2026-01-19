import pandas as pd
from dotenv import load_dotenv
import yfinance as yf
import finnhub
import os
import requests

#load my env file
load_dotenv()


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


#retrieve news articles from finnhub api
def finnhub_news_retrieval(ticker, start_date, end_date):
    #get my api key from env file
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    #create finnhub client
    finnhub_client = finnhub.Client(api_key=finnhub_key)
    news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
    return news


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
    return data
