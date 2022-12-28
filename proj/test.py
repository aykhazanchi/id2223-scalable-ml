import requests

url = ('https://api.eia.gov/v2/electricity/rto/daily-region-sub-ba-data/data/'
       '?frequency=daily'
       '&data[0]=value'
       '&facets[subba][]=ZONJ'
       '&facets[timezone][]=Eastern'
       '&start=2020-01-01'
       '&end=2020-12-31'
       '&sort[0][column]=period'
       '&sort[0][direction]=desc'
       '&offset=0'
       '&length=5000'
       '&api_key=gxfTV1lJaU0R5y0MFUjNR0GurPkbLk7a8ZrIJbk1')

data = requests.get(url).json()['response']['data']

print(len(data))
