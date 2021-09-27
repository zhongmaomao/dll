import torch
from torch import nn
from torch.nn import init
import torch.optim as optim
import numpy as np
import random
import pandas
import ccxt
import time



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': 'cFYwdaq9165KJQdro08mZkjXpsO1rETGFC0ENSu4E0YihuCDnLlyklTB6QqyyJQE',
    'secret': 'Luk0BTp4yDqDqpagKApDaC1frb1Qb9heyo03A3ylsRT4oMuKuXlOaUSDYpcWsNst',
    'timeout': 30000,
    'enableRateLimit': True,
})

#print(exchange.fetch_order_book("BTC/USDT"))
#exchange.load_markets()
kldata = exchange.fetch_ohlcv('BTC/USDT', '1d')
#if exchange.has['fetchOHLCV']:
#    for symbol in exchange.markets:
#        time.sleep (exchange.rateLimit / 1000) # time.sleep wants seconds
#        print (symbol, exchange.fetch_ohlcv (symbol, '1d')) # one day