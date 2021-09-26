import pandas as pd
from libs.model import HGPforecaster, PiecewiseLinearChangepoints
from libs.metrics import calculate_metrics, metrics_to_table
from libs.pre_processing import generate_groups_data_flat, generate_groups_data_matrix
from libs.visual_analysis import visualize_fit, visualize_predict, visualize_prior
import json

# Tourism dataset

data = pd.read_csv('../data/TourismData_v3.csv')
data['Year'] = data['Year'].fillna(method='ffill')

d = dict((v,k) for k,v in enumerate(calendar.month_name))
data.Month = data.Month.map(d)
data = data.assign(t=pd.to_datetime(data[['Year', 'Month']].assign(day=1))).set_index('t')
data = data.drop(['Year', 'Month'], axis=1)
data = data.round()

groups_input = {
    'state': [0,1],
    'zone': [0,2],
    'region': [0,3],
    'purpose': [3,6]
}

groups = generate_groups_data_flat(data, groups_input, seasonality=12, h=24)

m = HGPforecaster(groups_data=groups,
                  n_iterations=100000,
                  changepoints = 4)

m.fit_vi()
print('Fitting Model...')
m.predict()
print('Predicting...')

dictionary_data = results
a_file = open("restults.json", "w")
json.dump(dictionary_data, a_file)

