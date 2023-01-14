# Project - Predicting Daily Electricity Demand in New York, USA

The project is a machine learning prediction service that predicts the daily electricity demand in megawatthours in New York, USA. 

As features, the project uses historical daily demand, daily average temperature (celsius), and whether the date was a US Federal holiday or not. 

Training data consists of the years 2017 - 2021. The data for historical daily demand is obtained from [US Energy Information Administration (EIA)](https://www.eia.gov/) ([Link to API](https://www.eia.gov/opendata/browser/electricity/rto/daily-region-data)) and the data for historical average daily temperature is obtained from [National Oceanic and Atmospheric Administration (NOAA)](https://www.noaa.gov/). The US Federal holidays are obtained from the `USFederalHolidayCalendar` library in `pandas.tseries.holiday`. The temperature data for the daily scheduled batch predictions is obtained from [WeatherAPI](https://api.weatherapi.com/).

Along with the model's predictions we also provide the forecasted demand from EIA and find that our model's predicted values turn out to be fairly close to those of the official EIA forecast.

## Links
- Dashboard
    - https://rscolati-electricity-monitoring.hf.space/
- Interactive
    - https://rscolati-electricity.hf.space/
- Code
    - Source code in `./proj`

## Pipelines

**feature.ipynb**


**training.ipynb**


**feature-daily.py**


**batch-daily.py**


## Model