import pandas as pd
import prophet
from prophet.serialize import model_to_json
import json

daten = pd.read_csv('station_based_data/prepared_Wolbecker_Strasse.csv', index_col=0, parse_dates=True)
daten.interpolate('time', inplace=True)

m = prophet.Prophet(weekly_seasonality=True,daily_seasonality=True)
m.add_regressor('Temperatur_C', mode='multiplicative')
m.add_regressor('Windstaerke _km_h', mode='multiplicative')
m.add_regressor('Regen', mode='multiplicative')
m.add_regressor('ferien', mode='multiplicative')
m.add_regressor('feiertag', mode='multiplicative')
m.fit(daten)

future_daten = pd.read_csv('request_Wolbecker_Strasse.csv', index_col=0, parse_dates=True)
forecast = m.predict(future_daten, freq='H')
with open('serialized_model.json', 'w') as fout:
    json.dump(model_to_json(m), fout)  # Save model
print(forecast)
