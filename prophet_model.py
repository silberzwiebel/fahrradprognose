import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# https://raw.githubusercontent.com/codeformuenster/open-data/master/verkehrsdaten/fahrrad/Fahrradzaehlstellen-Stundenwerte.csv
df = pd.read_csv('example_wp_log_peyton_manning.csv')
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)

fig_plotly = plot_plotly(m, forecast)
fig_plotly.show()
