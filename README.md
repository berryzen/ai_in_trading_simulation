# Ai in trading simulation

Here is AI, Model free reinforcement learning written from scratch, including optimization, neural networks, feature engineering, environment and backtesting

Optimizer is custom algorithm inspired by genetic algorithms, gradient descent and swarm optimization. Neural network is LSTM. Feature engineering includes signal processing techniques like simple moving average, exponential moving average and direction of change. Environment is space where agent gets observed information and makes actions based on it. Backtesting is simulation where AI tries to trade.

Computation is optimized for speed with numba and numpy libaries. 

Agent tries to choose when to buy and when to sell while running through sp500 market data.
Cost of buy or sell order is not included in simulation so result seem little bit better.
 Markup : 1. Orange line is AI's profit fluctuation while trading sp500  
          2. Blue is profit of baseline strategy, no trades, just scaled price data

![experiement](https://user-images.githubusercontent.com/93252944/150828656-e51b3e7b-e71c-4442-b86b-bf11880b9919.png)
