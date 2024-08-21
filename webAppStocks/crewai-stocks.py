# Import das libs
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env.

import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Criando Yahoo Finance Tool


def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock


yahoo_finance_tool = Tool(
    name="Yahoo finance Tool",
    description="fetches stocks price for {ticket} form the last year about a specific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)
)


# Importandoo OpenAi LLM - GPT
llm = ChatOpenAI(model="gpt-4o-mini")


# Agente analista de preço
stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory=""" You're highly experienced in analyzing the price of an specific stock
  and make predictions about its future price.
  """,
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False
)


getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a trend analysis of up, down or sideways",
    expected_output=""" Specify the current trend stock price - up, down, or sideways.
  eg. stock= 'APPL, price UP'
  """,
    agent=stockPriceAnalyst
)

# Agente analista de notícias


# Importando a tool do search
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)


newsAnalyst = Agent(
    role="Senior Stock News Analyst",
    goal="""Create a short summary of the market news related to the stock ticket 
  {ticket} company. Specify the current trend - up, down, sideways - with the news context.
  For each requested stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed""",
    backstory=""" You're highly experienced in analyzing the market trends and news and have tracked assets for more than 10 years.
  
  You're alsoo master level analyst in the traditional market and have deep understanding of human psychology
  
  You understand news, their titles and information, but you look at those with a health dose of skepticism.
  You consider also the source of the news article.
  """,
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[search_tool],
    allow_delegation=False
)

getNews = Task(
    description=f"""Take the stock and always include BTC to it (if not requested).
  Use the search tool to search each one individualy.

  The current date is {datetime.now()}

  Compose the results into a helpful report.
  """,
    expected_output="""A summary of the overall market and one sentece summary for each request asset.
  Include a fear/greed score for each asset based on the news. Use the format:
  <STOCK ASSET>
  <SUMMARY BASED ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED SCORE>
  """,
    agent=newsAnalyst,
)


# Agente escritor da análise
stockAnalystWriter = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyze the trends price and news and write an insightful compelling and informative 3 paragraph long newsletter based on the  stock report and price trend.""",
    backstory="""You're widely acceped as the best stock analyst in the market. You understand complex concepts and create compelling stories
  and narratives that resonate with wider audiences.
  
  You understand macro factors and combine multiple theories - e.g. cycle theory and fundamental analysis.
  You're able to hold multiple opinions when analysing anything.
  """,
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    # tools=[search_tool]
    allow_delegation=True
)


writeAnalysis = Task(
    description="""Use the stock price trend and the stock news report to create an analysis and write the newsletter about the ticket {ticket} company
  that is brief and highlights the most important points.
  Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
  Include the previous analysis of stock trend and news summary.
  """,
    expected_output="""An eloquent 3 paragraph newsletter formated as markdown (without the ``` / code block marker, just the text in markdown) in an easy readable manner. It should contains:
  - 3 bullets executive summary
  - Introduction - set the overall picture and spike up the interest
  - main part provides the meat of the analysis including the news summary and fear/greed scores
  - summary - key facts and concrete future trend prediction - up, down or sideways
  
  """,
    agent=stockAnalystWriter,
    context=[getStockPrice, getNews]
)

# Criado a equipe
crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks=[getStockPrice, getNews, writeAnalysis],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

# Executando


# final_output = results.raw


with st.sidebar:
    st.header("Enter the Stock to Research")

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results = crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of your research:")
        st.write(results.raw)
