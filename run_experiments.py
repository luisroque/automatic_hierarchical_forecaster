import os
# one of
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'

import warnings
import pandas as pd
import arviz as az
import calendar
from libs.metrics import calculate_metrics
from libs.pre_processing import generate_groups_data_flat, generate_groups_data_matrix
from libs.visual_analysis import visualize_fit, visualize_predict, traceplot, visualize_prior, model_graph, plot_elbo
from libs.model_minibatch_series import HGPforecaster, PiecewiseLinearChangepoints
import theano
import json
theano.config.compute_test_value='raise'

# Outside Piecewise Linear

data = pd.read_csv('./data/TourismData_v3.csv')
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

groups = generate_groups_data_flat(y = data, 
                               groups_input = groups_input, 
                               seasonality=12, 
                               h=24)

m = HGPforecaster(groups_data=groups,
                  n_iterations=100000,
                  piecewise_out=True,
                  changepoints = 4)

m.fit_vi()

m.predict()

results = calculate_metrics(m.pred_samples_predict, groups)

dictionary_data = results
a_file = open("restults.json", "w")
json.dump(dictionary_data, a_file)

