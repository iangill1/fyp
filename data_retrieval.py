from dotenv import load_dotenv
import yfinance as yf
import finnhub
import os

#load my env file
load_dotenv()

#retrieve historical stock price data from yfinance api
def price_retrieval(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

#retrieve news articles from finnhub api
def news_retrieval(ticker, start_date, end_date):
    #get my api key from env file
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    #create finnhub client
    finnhub_key = finnhub.Client(api_key=finnhub_key)
    news = finnhub_key.company_news(ticker, _from=start_date, to=end_date)
    return news
