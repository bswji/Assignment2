import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump,load
import numpy as np
pd.set_option('display.max_columns', None)

traindf = pd.read_csv('data/sales_train.csv')
testdf = pd.read_csv('data/sales_test.csv')
calendar = pd.read_csv('data/calendar.csv')
calendar_events = pd.read_csv('data/calendar_events.csv')
item_price = pd.read_csv('data/items_weekly_sell_prices.csv')

#Create date column
date_column = []
for col in traindf.columns:
    if col.startswith('d_'):
        date_column.append(col)

#Transpose date 
df = pd.melt(traindf,  id_vars=['id', 'item_id', 'dept_id','cat_id','store_id','state_id'],value_vars=date_column, var_name='d')

print(df.head(40))

#Add date column
date_added = pd.merge(df,calendar, on='d',how = "left")

#Add event column
event_added = pd.merge(date_added,calendar_events, on ='date',how='left')

#Remove spaces and punctuation and change to lower case for calendar events
event_added['event_name'] = event_added['event_name'].str.replace(r'[ \-\'"]', '', regex=True)
event_added['event_name'] = event_added['event_name'].str.lower()

#Check event type col and change to lower case
event_added['event_type'] = event_added['event_type'].str.lower()

#Check null values and change to 'normalday'
event_added.isnull().sum()
event_added['event_type'] = event_added['event_type'].fillna("normalday")
event_added['event_name'] = event_added['event_name'].fillna("normalday")

#Add sell_price column
final_df = pd.merge(event_added, item_price, on =['store_id', 'item_id', 'wm_yr_wk'], how='left')
final_df.head(40)
final_df['sell_price'].isnull().sum()

#Replace null in sell_price to 0 and rename value column to units sold
final_df['sell_price'] = final_df['sell_price'].fillna("0")
final_df.rename(columns={'value': 'units_sold'}, inplace=True)
final_df.head(5)

#Change data types of columns
final_df['date'] = pd.to_datetime(final_df['date'])
final_df['sell_price'] = final_df['sell_price'].astype(float)
final_df['units_sold'] = final_df['units_sold'].astype(float)
final_df.info()
#Create total revenue column
final_df['total_revenue'] = final_df['units_sold'] * final_df['sell_price']
final_df['total_revenue'] = final_df['total_revenue'].astype(float)

#Create forecast dataset
forecast_df = final_df.groupby(final_df['date'])['total_revenue'].sum().reset_index()

#Create forecast dataset with units sold and sell price
forecast_df2 = final_df.groupby('date').agg(total_revenue=('total_revenue', 'sum'),units_sold_sum=('units_sold', 'sum'),sell_price_avg=('sell_price', 'mean')).reset_index()
forecast_df2.head(5)

#Check for instances where units_sold and sell_price both = 0 and remove
((final_df['units_sold'] != 0) & (final_df['sell_price'] == 0)).sum()
((final_df['units_sold'] == 0) & (final_df['sell_price'] == 0)).sum()
((final_df['units_sold'] == 0) & (final_df['sell_price'] != 0)).sum()
final_df = final_df[~((final_df['units_sold'] == 0) & (final_df['sell_price'] == 0))]

#drop ID column, drop d column, drop wm_yr_wk column
final_df = final_df.drop('id', axis=1)
final_df = final_df.drop('d', axis=1 )
final_df = final_df.drop('wm_yr_wk', axis=1 )
#Transform date column into year,month,day
final_df['year'] = final_df['date'].dt.year
final_df['month'] = final_df['date'].dt.month
final_df['day'] = final_df['date'].dt.day
final_df = final_df.drop('date', axis=1 )

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in final_df.columns:
    if final_df[column].dtype == 'object':
        label_encoder = LabelEncoder()  
        final_df[column] = label_encoder.fit_transform(final_df[column]) 
        label_encoders[column] = label_encoder 
#Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(final_df[["item_id", "dept_id", "cat_id", "store_id", "state_id", 'units_sold', "event_name", "event_type", 'sell_price', "year", "month", "day"]], final_df["total_revenue"], test_size=0.3, random_state=5)

#Train imputer
from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
knn_imputer.fit(X_train)

#Create validation set
cols_to_attach = ['id','item_id','dept_id','cat_id','store_id','state_id']
val_df = pd.concat([testdf, traindf[cols_to_attach]], axis=1)

date_column_val = []
for col in val_df.columns:
    if col.startswith('d_'):
        date_column_val.append(col)

#Transpose date 
val_df_trans = pd.melt(val_df,  id_vars=['id', 'item_id', 'dept_id','cat_id','store_id','state_id'], value_vars=date_column_val, var_name='d')

#Add date column
val_date_added = pd.merge(val_df_trans,calendar, on='d',how = "left")

#Add event column
calendar_events['date'] = pd.to_datetime(calendar_events['date'])
val_date_added['date'] = pd.to_datetime(val_date_added['date'])
val_event_added = pd.merge(val_date_added,calendar_events, on ='date',how='left')

#Remove spaces and punctuation and change to lower case for calendar events
val_event_added['event_name'] = val_event_added['event_name'].str.replace(r'[ \-\'"]', '', regex=True)
val_event_added['event_name'] = val_event_added['event_name'].str.lower()

#Check event type col and change to lower case
val_event_added['event_type'] = val_event_added['event_type'].str.lower()

#Check null values and change to 'normalday'
val_event_added['event_type'] = val_event_added['event_type'].fillna("normalday")
val_event_added['event_name'] = val_event_added['event_name'].fillna("normalday")

#Add sell_price column
val_final = pd.merge(val_event_added, item_price, on =['store_id', 'item_id', 'wm_yr_wk'], how='left')
val_final.head(40)
val_final['sell_price'].isnull().sum()

#Replace null in sell_price to 0 and rename value column to units sold
val_final['sell_price'] = val_final['sell_price'].fillna("0")
val_final.rename(columns={'value': 'units_sold'}, inplace=True)
val_final.head(5)

#Change data types of columns
val_final['date'] = pd.to_datetime(val_final['date'])
val_final['sell_price'] = val_final['sell_price'].astype(float)
val_final['units_sold'] = val_final['units_sold'].astype(float)
val_final.info()

#Create total revenue column
val_final['total_revenue'] = val_final['units_sold'] * val_final['sell_price']
val_final['total_revenue'] = val_final['total_revenue'].astype(float)

#Check for instances where units_sold and sell_price both = 0 and remove
((val_final['units_sold'] != 0) & (val_final['sell_price'] == 0)).sum()
((val_final['units_sold'] == 0) & (val_final['sell_price'] == 0)).sum()
((val_final['units_sold'] == 0) & (val_final['sell_price'] != 0)).sum()
val_final = val_final[~((val_final['units_sold'] == 0) & (val_final['sell_price'] == 0))]

#Create forecast validation set
forecast_val = val_final.groupby('date').agg(total_revenue=('total_revenue', 'sum'),units_sold_sum=('units_sold', 'sum'),sell_price_avg=('sell_price', 'mean')).reset_index()

#drop ID column, drop d column, drop wm_yr_wk column
val_final = val_final.drop('id', axis=1)
val_final = val_final.drop('d', axis=1 )
val_final = val_final.drop('wm_yr_wk', axis=1)

#Transform date column into year,month,day
val_final['year'] = val_final['date'].dt.year
val_final['month'] = val_final['date'].dt.month
val_final['day'] = val_final['date'].dt.day
val_final = val_final.drop('date', axis = 1)

#Transform categorical columns
val_final['event_name'] = label_encoder.fit_transform(val_final['event_name'])
val_final['event_type'] = label_encoder.fit_transform(val_final['event_type'])
val_final['item_id'] = label_encoder.fit_transform(val_final['item_id'])
val_final['dept_id'] = label_encoder.fit_transform(val_final['dept_id'])
val_final['store_id'] = label_encoder.fit_transform(val_final['store_id'])
val_final['state_id'] = label_encoder.fit_transform(val_final['state_id'])
val_final['cat_id'] = label_encoder.fit_transform(val_final['cat_id'])

#Split into features and target
val_target = val_final[['total_revenue']]
val_features = val_final.drop('total_revenue', axis=1)

# - - - -- Modelling - - - - - - - - - - - - - - - - -

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

final_df.info()

#Random forest
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
re = load('models/regressor.joblib')
pred = re.predict(X_test)
#calculate RMSE and R2 of test set
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(rmse)
r2 = r2_score(y_test, y_pred)
print(r2)

#XGboost
import xgboost as xgb 
xgbmodel = xgb.XGBRegressor()
xgbmodel.fit(X_train, y_train)
xgb_pred = xgbmodel.predict(X_test)

#calculate RMSE 
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_rmse = sqrt(xgb_mse)
print(xgb_rmse)

val_features.head(4)
#Calculate RMSE with no units sold, sell price
X_train2, X_test2, y_train2, y_test2 = train_test_split(final_df[["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name", "event_type", "year", "month", "day"]], final_df["total_revenue"], test_size=0.3, random_state=5)
xgbmodel2 = xgb.XGBRegressor()
xgbmodel2.fit(X_train2, y_train2)

xgb_pred2 = xgbmodel2.predict(X_test2)
xgb_mse2 = mean_squared_error(y_test2, xgb_pred2)
xgb_rmse2 = sqrt(xgb_mse2)
print(xgb_rmse2)

#Test no units sold and sell price on validation
val_no_cols = val_features.drop(['units_sold','sell_price'],axis=1)
a = xgbmodel2.predict(val_no_cols)
val_no_cols_mse = mean_squared_error(val_target, a)
val_no_cols_rmse = sqrt(val_no_cols_mse)
print(val_no_cols_rmse)

#Fill units_sold and sell_price with null
val_features['units_sold'] = np.nan
val_features['sell_price'] = np.nan
val_features.head(3)
pred_nan = xgbmodel.predict(val_features)
nan_mse = mean_squared_error(val_target, pred_nan)
nan_rmse = sqrt(nan_mse)
print(nan_rmse)

#Create new features
a = X_train
average_units_sold_by_category = a.groupby('cat_id')['units_sold'].mean()
a['avg_units_sold_cat'] = a['cat_id'].map(average_units_sold_by_category)

average_sell_price_by_category = a.groupby('cat_id')['sell_price'].mean()
a['avg_sell_price_cat'] = a['cat_id'].map(average_sell_price_by_category)

average_units_sold_by_dept = a.groupby('dept_id')['units_sold'].mean()
a['avg_units_sold_dept'] = a['dept_id'].map(average_units_sold_by_dept)

average_sell_price_by_dept = a.groupby('dept_id')['sell_price'].mean()
a['avg_sell_price_dept'] = a['dept_id'].map(average_sell_price_by_dept)

average_units_sold_by_store = a.groupby('store_id')['units_sold'].mean()
a['avg_units_sold_store'] = a['store_id'].map(average_units_sold_by_store)

average_sell_price_by_store = a.groupby('store_id')['sell_price'].mean()
a['avg_sell_price_store'] = a['store_id'].map(average_sell_price_by_store)

average_units_sold_year_month_cat = a.groupby(['year', 'month', 'cat_id', 'store_id', 'day'])['units_sold'].mean().reset_index()
average_units_sold_year_month_cat = average_units_sold_year_month_cat.rename(columns={'units_sold': 'average_units_sold_year_month_cat'})
a = a.merge(average_units_sold_year_month_cat, on=['year', 'month', 'cat_id','store_id', 'day'], how='left')

average_sell_price_year_month_cat = a.groupby(['year', 'month', 'cat_id' , 'store_id', 'day'])['sell_price'].mean().reset_index()
average_sell_price_year_month_cat = average_sell_price_year_month_cat.rename(columns={'sell_price': 'average_sell_price_year_month_cat'})
a = a.merge(average_sell_price_year_month_cat, on=['year', 'month', 'cat_id','store_id', 'day'], how='left')

a['date'] = pd.to_datetime(a[['year', 'month', 'day']])
a['day_of_week'] = a['date'].dt.dayofweek
a = a.drop('date',axis = 1)
a = a.drop(['units_sold', 'sell_price'],axis = 1)
#Train model
new_feature_model = xgb.XGBRegressor()
new_feature_model.fit(a, y_train)

#Test on validation
c = val_features
c['avg_units_sold_cat'] = c['cat_id'].map(average_units_sold_by_category)
c['avg_sell_price_cat'] = c['cat_id'].map(average_sell_price_by_category)
c['avg_units_sold_dept'] = c['dept_id'].map(average_units_sold_by_dept)
c['avg_sell_price_dept'] = c['dept_id'].map(average_sell_price_by_dept)
c['avg_units_sold_store'] = c['store_id'].map(average_units_sold_by_store)
c['avg_sell_price_store'] = c['store_id'].map(average_sell_price_by_store)
c = c.merge(average_units_sold_year_month_cat, on=['year', 'month', 'cat_id','store_id', 'day'], how='left')
c = c.merge(average_sell_price_year_month_cat, on=['year', 'month', 'cat_id','store_id', 'day'], how='left')
c['date'] = pd.to_datetime(c[['year', 'month', 'day']])
c['day_of_week'] = c['date'].dt.dayofweek
c = c.drop('date',axis = 1)
c = c.drop(['units_sold', 'sell_price'],axis = 1)

val_new_pred = new_feature_model.predict(c)
val_new_feature_mse = mean_squared_error(val_target, val_new_pred)
val_new_rmse = sqrt(val_new_feature_mse)
print(val_new_rmse)

#Hist gradient
from sklearn.ensemble import HistGradientBoostingRegressor
reg = HistGradientBoostingRegressor()
reg.fit(X_train, y_train)
his_prd = reg.predict(X_test)

#calculate RMSE 
his_mse = mean_squared_error(y_test, his_prd)
his_rmse = sqrt(his_mse)
print(his_rmse)

#Calculate RMSE with no units sold, sell price
histmodel2 = HistGradientBoostingRegressor()
histmodel2.fit(X_train2, y_train2)
hist_pred2 = histmodel2.predict(X_test2)
hist_mse2 = mean_squared_error(y_test2, hist_pred2)
hist_rmse2 = sqrt(hist_mse2)
print(hist_rmse2)

#Test on validation
sample = int(0.01 * len(val_final))
random_sample = val_final.sample(n=sample, random_state=42) 
random_sample.head(5)
random_sample['sell_price'] = np.nan
random_sample['units_sold'] = np.nan
random_sample_features = random_sample.drop('total_revenue', axis = 1)
random_sample_target = random_sample['total_revenue']

#Tune hyper parameters
from sklearn.model_selection import RandomizedSearchCV

params = {
    'learning_rate':[0.1,0.2],
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
}
histmodel2 = RandomizedSearchCV(estimator=histmodel2, param_distributions=params, scoring='neg_mean_squared_error', cv=5, n_iter=10)
histmodel2.fit(X_train2, y_train2)
best_params = histmodel2.best_params_
best_model = histmodel2.best_estimator_

#Model forecast data
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

forecast_df = pd.read_csv('data/forecast_df.csv')
forecast_df.set_index('date', inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the time series data
ax.plot(forecast_df.index, forecast_df['total_revenue'], label='total_revenue', color='blue')

# Customize the plot
ax.set_title('Time Series Data')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.grid(True)
ax.legend()

# Display the graph
plt.show()

#Initial ADF
result = adfuller(forecast_df['total_revenue'])
p_value = result[1]
print(p_value)

#Initial KPSS
result2 = kpss(forecast_df['total_revenue'], regression='c')
p_value2 = result2[1]
print(p_value2)

from sklearn.linear_model import LinearRegression

#Detrend using linear model
# Fit model 
X = [i for i in range(0, len(forecast_df))]
X = np.reshape(X, (len(X), 1))
y = forecast_df["total_revenue"].values
model = LinearRegression()
model.fit(X, y)

# Calculate trend
trend = model.predict(X)

# Detrend
d = forecast_df
d["detrend"] = d["total_revenue"].values - trend

#ADF after detrending 
result = adfuller(d['detrend'])
p_value = result[1]
print(p_value)

#KPSS after detrending
result2 = kpss(d['detrend'], regression='c')
p_value2 = result2[1]
print(p_value2)

#Model with ARIMA
#Find best values for p,d,q
from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima
import statsmodels.api as sm

arima_data = d.drop('total_revenue', axis=1)
arima_data.set_index('date',inplace=True)
arima_data.head(3)
train_data, test_data = train_test_split(arima_data, train_size=0.8,shuffle=False)
model = auto_arima(train_data, seasonal=True, m=12,stepwise=True, trace=True)

#Make test set predictions
n_periods = len(test_data)
forecast, conf_int = model.predict(n_periods=n_periods, exog=None, return_conf_int=True)
forecasted_df = pd.DataFrame({'Date': test_data.index, 'Forecasted_Values': forecast})
forecasted_df.set_index('Date', inplace=True)
print(forecasted_df)

#Calculate rmse 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rmse = np.sqrt(mean_squared_error(test_data['detrend'], forecasted_df['Forecasted_Values']))
print(rmse)

#SARIMAX
sarimax_df = pd.read_csv('data/forecast_df.csv')
sarimax_df.info()
#Merge with calendar events to get event_type and event_name cols
sarimax_df['date'] = pd.to_datetime(sarimax_df['date'])
calendar_events['date'] = pd.to_datetime(calendar_events['date'])
merged_df = pd.merge(sarimax_df, calendar_events, on='date', how = 'left')
merged_df['event_type'] = merged_df['event_type'].fillna("normalday")
merged_df['event_name'] = merged_df['event_name'].fillna("normalday")
#Encode event_type and event_name
label_encoder = LabelEncoder()
merged_df['event_type'] = label_encoder.fit_transform(merged_df['event_type'])
merged_df['event_name'] = label_encoder.fit_transform(merged_df['event_name'])
#Create year,month,day,quarter
merged_df['year'] = merged_df['date'].dt.year
merged_df['month'] = merged_df['date'].dt.month
merged_df['day'] = merged_df['date'].dt.day
merged_df['quarter'] = merged_df['date'].dt.quarter

sarimax_data = merged_df
sarimax_data['date'] = pd.to_datetime(sarimax_data['date'])
sarimax_data.set_index('date', inplace=True)
#Detrend using linear model
# Fit model 
X = [i for i in range(0, len(sarimax_data))]
X = np.reshape(X, (len(X), 1))
y = sarimax_data["total_revenue"].values
model = LinearRegression()
model.fit(X, y)

# Calculate trend
trend = model.predict(X)

# Detrend
sarimax_detrend = sarimax_data
sarimax_detrend["detrend"] = sarimax_detrend["total_revenue"].values - trend
sarimax_detrend.head(3)

#Split into train and test set
test_size = 0.2
split_point = int(len(sarimax_detrend) * (1 - test_size))
train_data = sarimax_detrend.iloc[:split_point]
test_data = sarimax_detrend.iloc[split_point:]

#Create endog and exog lists
endog = sarimax_detrend['detrend'].values
exog = sarimax_detrend[['event_name','event_type','year','month','day','quarter']].values

train_endog = train_data['detrend'].values
train_exog = train_data[['event_name','event_type','year','month','day','quarter']].values
test_endog = test_data['detrend'].values
test_exog = test_data[['event_name','event_type','year','month','day','quarter']].values

#Train model
model = sm.tsa.SARIMAX(train_endog, exog=train_exog, order=(5, 1, 2), seasonal_order=(0, 0, 1, 12))
sarimax_fitted = model.fit()

#Predict values on test set
predicted_values = sarimax_fitted.get_forecast(steps = len(test_exog), exog=test_exog)
pred_vals = predicted_values.predicted_mean

#Calculate RMSE
mse = mean_squared_error(test_endog, pred_vals)
rmse = np.sqrt(mse)
print(rmse)

#Use XGBOOST
#Split date into year, month, day
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

xgb_df = forecast_df2
xgb_df.info()
#Merge with calendar events to get event_type and event_name cols
xgb_df['date'] = pd.to_datetime(xgb_df['date'])
calendar_events['date'] = pd.to_datetime(calendar_events['date'])
merge = pd.merge(xgb_df, calendar_events, on='date', how = 'left')
merge['event_type'] = merge['event_type'].fillna("normalday")
merge['event_name'] = merge['event_name'].fillna("normalday")
#Encode event_type and event_name
label_encoder = LabelEncoder()
merge['event_type'] = label_encoder.fit_transform(merge['event_type'])
merge['event_name'] = label_encoder.fit_transform(merge['event_name'])
#Create year,month,day,quarter
merge['year'] = merge['date'].dt.year
merge['month'] = merge['date'].dt.month
merge['day'] = merge['date'].dt.day
merge['quarter'] = merge['date'].dt.quarter
#Create lag features
merge['lag'] = merge['total_revenue'].shift(1)
merge['lag2'] = merge['total_revenue'].shift(7)

#Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(merge[['year','month','day','quarter','event_name','event_type','lag','lag2']],merge['total_revenue'],test_size=0.3,shuffle=False)

#Predict test set
xgb_forecast = xgb.XGBRegressor()
xgb_forecast = xgb_forecast.fit(X_train, y_train)
y_pred = xgb_forecast.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

#Tune hyperparameters
from sklearn.model_selection import RandomizedSearchCV

params = {
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
}
random_search = RandomizedSearchCV(estimator=xgb_forecast, param_distributions=params, scoring='neg_mean_squared_error', cv=5, n_iter=10)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
best_model = random_search.best_estimator_

#Predict test set
xgb_tuned_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

#Test on validation set
forecast_val.head(3)
val = forecast_val
val['year'] = val['date'].dt.year
val['month'] = val['date'].dt.month
val['day'] = val['date'].dt.day
val['quarter'] = val['date'].dt.quarter
val = pd.merge(val, calendar_events, on='date', how = 'left')
val['event_type'] = val['event_type'].fillna("normalday")
val['event_name'] = val['event_name'].fillna("normalday")
val = val.drop('date',axis = 1)
#Encode event_type and event_name
label_encoder = LabelEncoder()
val['event_type'] = label_encoder.fit_transform(val['event_type'])
val['event_name'] = label_encoder.fit_transform(val['event_name'])
#Create lag cols
val['lag'] = val['total_revenue'].shift(1)
val['lag2'] = val['total_revenue'].shift(7)
#Separate features and target
val_features = val.drop('total_revenue',axis = 1)
val_features = val_features[['year','month','day','quarter','event_name','event_type','lag','lag2']]
val_target = val['total_revenue']
#Predict
val_preds_forecast = xgb_forecast.predict(val_features)
rmse = np.sqrt(mean_squared_error(val_target, val_preds_forecast))
print(rmse)

#Create objects to dump for API
#Dump best models
dump(histmodel2, 'models/histmodel.joblib')
dump(best_model, 'models/xgbforecast.joblib')
#Create calendar_events object
calendar_events = pd.DataFrame(calendar_events)
calendar_events['date'] = pd.to_datetime(calendar_events['date'])
calendar_events['event_name'] = calendar_events['event_name'].str.replace(r'[ \-\'"]', '', regex=True)
calendar_events['event_name'] = calendar_events['event_name'].str.lower()
calendar_events['event_type'] = calendar_events['event_type'].str.lower()
dump(calendar_events, 'models/calendar_events.joblib')
#Dump group by objects
dump(average_units_sold_by_category, 'models/unitssoldcat.joblib')
dump(average_sell_price_by_category, 'models/sellpricecat.joblib')
dump(average_units_sold_by_dept, 'models/unitssolddept.joblib')
dump(average_sell_price_by_dept, 'models/sellpricedept.joblib')
dump(average_units_sold_by_store, 'models/unitssoldstore.joblib')
dump(average_sell_price_by_store, 'models/sellpricestore.joblib')
dump(average_units_sold_year_month_cat, 'models/unitssoldymdcatstore.joblib')
dump(average_sell_price_year_month_cat, 'models/sellpriceymdcatstore.joblib')
#Create label encoder and knn imputer
dump(label_encoders, 'models/label_encoders.joblib')
dump(knn_imputer, 'models/knnimputer.joblib')

