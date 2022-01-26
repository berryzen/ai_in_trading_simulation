# Ai in trading simulation

Here is AI

# Technically 
This is model free reinforcement learning written from scratch, including optimization, neural networks, feature engineering, environment and backtesting

Optimizer is custom algorithm inspired by genetic algorithms, gradient descent and swarm optimization. Neural network is LSTM. Feature engineering includes signal processing techniques like simple moving average, exponential moving average and direction of change. Environment is space where agent gets observed information and makes actions based on it. Backtesting is simulation where AI tries to trade.

Computation is optimized for speed with numba and numpy libaries. 

# Example
- You can run this example from trading_simulation.ipynb
- Agent tries to choose when to buy and when to sell while running through sp500 market data. Cost of buy or sell order is not included in simulation so result seem bit better.

- Dataset can be any stock dataset with close price data and more than 2000 days/hours/minutes/rows of trading
- Agents runs throught sp500 close price data in training and testing 
- Invested money is 100€ for both strategies. Then trading begins.

# Results
- Orange line is AI's profit fluctuation while trading sp500.
- Blue is profit of baseline strategy, no trades, just scaled price data.
- Invested money is 100€ for both strategies. Then trading begins.

![experiement](https://user-images.githubusercontent.com/93252944/150828656-e51b3e7b-e71c-4442-b86b-bf11880b9919.png)
