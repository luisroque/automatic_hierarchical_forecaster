# automatic_hierarchical_forecaster
Automatic Hierarchical Time Series Forecasting using Gaussian Processes

Note that the key dependency of automatic_hierarchical_forecaster is **PyMc3** a library that depends on **Theano**.

## Key Features
* Hierarchical forecasting
* Additive seasonality, medium term irregularities and noise modeled by additive Gaussian Processes
* Trend modeled by hierarchical piece-wise linear function
* Fitting and plotting
* Fitting with AVDI (possibility to define minibatches)

## Tourism example
Predicting the Australia tourism timeseries:
```python
import pandas as pd
from libs.model import *
from libs.metrics import *
from libs.pre_processing import *
from libs.visual_analysis import *
import theano
theano.config.compute_test_value='raise'

data = pd.read_csv('../data/TourismData_v3.csv')
data['Year'] = data['Year'].fillna(method='ffill')

# Preprocess the data
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

# Instantiate the model class
m = HGPforecaster(groups_data=groups,
                  n_iterations=100000,
                  changepoints = 4,
                  piecewise_out=True)

# Fit and predict
m.fit_vi()
m.predict()

# Visualize your predictions and credible intervals
visualize_predict(groups, m.pred_samples_predict, 8)

# Print the overall resuls
results = calculate_metrics(m.pred_samples_predict, groups)
metrics_to_table(results)
```

![Predictions](https://raw.githubusercontent.com/luisroque/automatic_hierarchical_forecaster/main/example_notebooks/images/visualize_predict.png)
