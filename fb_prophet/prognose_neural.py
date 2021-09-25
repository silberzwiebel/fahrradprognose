import pandas as pd
import neuralprophet
import json

daten = pd.read_csv('station_based_data/prepared_Wolbecker_Strasse.csv', index_col=0, parse_dates=True)
daten = daten.drop(['Wochentag','Monate','dunkel','Gewitter','Glätte'], axis=1)
daten.interpolate('time', inplace=True)

neuralprophet.set_random_seed(42)


m = neuralprophet.NeuralProphet(weekly_seasonality=3,daily_seasonality=8,impute_missing=True)
m.add_future_regressor(name='Temperatur_C')
m.add_future_regressor(name='Windstaerke _km_h')
m.add_future_regressor(name='Regen')
m.add_future_regressor(name='Schnee')
m.add_future_regressor(name='ferien')
m.add_future_regressor(name='feiertag')
m.fit(daten, freq='H')

#future_daten = pd.read_csv('request_Wolbecker_Strasse.csv', parse_dates=['ds'])
#future_daten = future_daten.drop(['Wochentag','Monate','dunkel','Gewitter','Glätte'], axis=1)
regressors = pd.DataFrame(data={'Temperatur_C': [25,20,15,10,10], 'Windstaerke _km_h': [25,20,19,18,18], 'Regen': [0,1,1,1,1], 'Schnee': [0,0,0,0,0], 'ferien': [0,0,0,0,0], 'feiertag': [0,0,0,0,0]})
future_daten = m.make_future_dataframe(df=daten, regressors_df=regressors, periods=5)
#future_daten.to_csv('future.csv')
forecast = m.predict(future_daten)
#with open('serialized_model.json', 'w') as fout:
    #json.dump(model_to_json(m), fout)  # Save model
print(forecast)
