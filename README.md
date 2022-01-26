# Ai in trading simulation

This is model free reinforcement learning written from scratch, including optimization, neural networks, feature engineering, environment and backtesting.

Optimizer is custom algorithm inspired by genetic algorithms, gradient descent and swarm optimization. Neural network is LSTM. Feature engineering includes signal processing techniques like simple moving average, exponential moving average and direction of change. Environment is space where agent gets observed information and makes actions based on it. Backtesting is simulation where AI tries to trade.

Computation is optimized for speed with numba and numpy libaries. 

## Usage
- Go to trading_simulation.ipynb
- Dataset can be any stock dataset with close price data of days/hours/minutes/rows of trading. Preferred length is atleast 3000 rows.
- There you can configure optimizer, lstm-architecture and so on.

## Example
- You can run this example from trading_simulation.ipynb
- Agent tries to choose when to buy or when to sell while running through sp500 market data. Cost of buy or sell order is not included in simulation so result seem bit better.

- Agents runs throught sp500 close price data in training and testing.
- Invested money is 100€ for both strategies. Then trading begins.

## Results
- Orange line is AI's profit fluctuation while trading sp500.
- Blue line is profit of baseline strategy, no trades. Line follows close price data.
 
![trading_sim](https://user-images.githubusercontent.com/93252944/151146667-bb15c991-e2c6-4a3f-a80f-38e10e22f1ec.png)

