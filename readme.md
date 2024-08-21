# Disclaimer

This project is for educational and informational purposes only. The analysis and predictions generated by the AI agents are based on historical data and news articles and should not be considered as financial advice. The authors and contributors of this project are not responsible for any financial decisions made based on the information provided by the software.

Always conduct your own research and consult with a qualified financial advisor before making any investment decisions.

# CrewAI Stocks Analysis

This project leverages multiple AI-driven agents to analyze stock price trends and market news for a specific stock ticker, ultimately generating a well-informed and insightful newsletter. The agents are built using the `crewai` framework, and the application is hosted using Streamlit.

## Overview

The application performs the following tasks:

1. **Stock Price Analysis**: Fetches historical stock price data from Yahoo Finance and analyzes the trend.
2. **Market News Analysis**: Gathers recent news related to the stock and performs sentiment analysis to assess the market's mood.
3. **Newsletter Generation**: Combines the results from the stock price and news analysis to generate a comprehensive, easy-to-read stock analysis report.

## How It Works

1. The application uses a series of AI agents:

   - **StockPriceAnalyst**: Analyzes the historical stock price data.
   - **NewsAnalyst**: Analyzes the latest market news related to the stock ticker.
   - **StockAnalystWriter**: Generates a newsletter combining the insights from the stock price and news analysis.

2. The agents are organized into a crew, and tasks are assigned to them to perform in a hierarchical process.

3. The application runs through Streamlit, providing a user-friendly interface to input a stock ticker and receive a detailed analysis.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/jjakob10/stockAgents.git
   cd crewai-stocks
   ```

2. **Install the required Python packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment variables**

   Create a `.env` file in the root directory of the project and add your OpenAI API key:

   ```bash
   OPENAI_API_KEY=your-openai-api-key
   ```

4. **Run the Application**

   Use the following command to start the Streamlit app:

   ```bash
   streamlit run crewai-stocks.py
   ```

## Usage

1. Open the application in your web browser (typically, it will be available at `http://localhost:8501`).

2. Enter the stock ticker you want to analyze in the sidebar.

3. Click "Run Research" to start the analysis.

4. View the results of the analysis directly within the application.

## Future Enhancements

- **Integration with more data sources**: Expand the scope by integrating additional financial and news APIs.
- **Improved Analysis Techniques**: Incorporate more sophisticated trend analysis methods.
- **User Authentication**: Add user authentication for personalized experiences.

## Credits

This project is from the "IA na prática" event hosted by [Rocketseat](https://www.rocketseat.com.br/).

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
