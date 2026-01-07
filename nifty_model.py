import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

nse = pd.read_csv('nse_indexes.csv')

nft = nse[nse.Index == 'NIFTY 50']

cols_to_drop = ['Index', 'Open', 'High', 'Low', 'Volume', 'Currency']
nft = nft.drop(cols_to_drop, axis=1)

nft['Date'] = pd.to_datetime(nft['Date'])
nft = nft.set_index('Date')
nft = nft.sort_index()

nft = nft.asfreq('B')
nft = nft.ffill()

model = auto_arima(
    nft,
    seasonal=False,
    m=1,
    stepwise=True,
    suppress_warnings=True
)

o = model.order
so = model.seasonal_order

model1 = SARIMAX(
    nft,
    order=o,
    seasonal_order=so
)

result = model1.fit()

joblib.dump(result, 'Nifty_Predictor.joblib')