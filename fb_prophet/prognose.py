import pandas as pd
import numpy as np
import prophet
from prophet.serialize import model_to_json
import json
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

daten = pd.read_csv('station_based_data/prepared_Wolbecker_Strasse.csv', index_col=0, parse_dates=True)
daten.head()
daten['y'] = np.log(daten['y']+1)
daten.interpolate('time', inplace=True)
daten.head()

m = prophet.Prophet(weekly_seasonality=True,daily_seasonality=True)
m.add_country_holidays(country_name='DE')
m.add_regressor('Temperatur_C')
m.add_regressor('Windstaerke _km_h')
m.add_regressor('Regen')
#m.add_regressor('ferien', mode='multiplicative')
#m.add_regressor('feiertag', mode='multiplicative')
m.fit(daten)


#df_cv = cross_validation(m, initial='876 days', period='180 days', horizon = '365 days')
#df_p = performance_metrics(df_cv)
#df_p.head()
#fig = plot_cross_validation_metric(df_cv, metric='mae')
#fig.show()


# future_daten = m.make_future_dataframe(periods=5)

future = pd.read_csv('request_Wolbecker_Strasse.csv', index_col=0, parse_dates=True)
#future = pd.read_csv('example_future.csv', parse_dates=['ds'])

future_daten = pd.concat((daten, future))

future_daten.tail(n = 10)

forecast = m.predict(future_daten)

#with open('serialized_model.json', 'w') as fout:
#    json.dump(model_to_json(m), fout)  # Save model

print(forecast)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
fig1.show()
fig2.show()

fig_plotly = plot_plotly(m, forecast)
fig_plotly.show()
