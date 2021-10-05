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
from libs.model_minibatch_series import HGPforecaster, PiecewiseLinearChangepoints
import numpy as np
import theano
theano.config.compute_test_value='raise'

# Read in the data
INPUT_DIR = './data/m5-data'
cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
stv = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
sellp = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')

# Transform column wide days to single column 

stv = stv.melt(list(stv.columns[:6]), var_name='day', value_vars=list(stv.columns[6:]), ignore_index=True)

# Group by the groups to consider (remove product_id as there are 3049 unique) 

stv = stv.groupby(['dept_id', 'cat_id', 'store_id', 'state_id', 'item_id', 'day']).sum('value').reset_index()

days_calendar = np.concatenate((stv['day'].unique().reshape(-1,1), cal['date'][:-56].unique().reshape(-1,1)), axis=1)
df_caldays = pd.DataFrame(days_calendar, columns = ['day','Date'])

# Add calendar days

stv = stv.merge(df_caldays, how='left', on='day')

stv['Date'] = stv['Date'].astype('datetime64[ns]')
stv.dtypes

# stv = stv.loc[stv['dept_id']=='FOODS_1']

# Transform in weekly data

stv_weekly = stv.groupby(['dept_id', 'store_id', 'state_id', 'item_id']).resample('W', on='Date')['value'].sum()

# Build the structure to then apply the grouping transformation

stv_pivot = stv_weekly.reset_index().pivot(index='Date',columns=['dept_id', 'store_id', 'state_id', 'item_id'], values='value')
stv_pivot = stv_pivot.fillna(0)

groups_input = {
    'Department': [0],
    'Store': [1],
    'State': [2],
    'Item': [3]
}

# The dataset results from the removal of item_id (groupby by the ohter groups) and by downsampling to weekly data (the dataset was daily)

groups = generate_groups_data_flat(stv_pivot, groups_input, seasonality=52, h=18)

# Instantiate the model class
m = HGPforecaster(groups_data=groups,
                  n_iterations=100000,
                  changepoints = 4,
                  piecewise_out=True)

# Fit and predict
m.fit_vi()

m.predict()

results = calculate_metrics(m.pred_samples_predict, groups)

dictionary_data = results
a_file = open("results_m5.json", "w")
json.dump(dictionary_data, a_file)

