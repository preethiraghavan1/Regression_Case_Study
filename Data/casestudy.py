'''
Year Made
MachineHoursCurrentMeter - missing data
UsageBand [This looks like an interaction between the
above two features] - missing data
Saleprice = is this our y?
Saledate [depending on economics of the time, also looking
at the typical age of machine at sale time]

fiModelDesc : let's see what this looks like

State: place of manufacture, weather conditions,
demand in that area

-----
22-54 are specific features of the equipment--
probably important, but how do we narrow them down?
'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.linear_model import LinearRegression


#open zipped file
zf = ZipFile('Train.zip')
df = pd.read_csv(zf.open('Train.csv'))

#create dummies for datasources & concat
datasources = pd.get_dummies(df['datasource'])
df = pd.concat([df, datasources], axis = 1)
#get rid v old & non-sensiscal year values (note: they may be meaningful)
df=df.loc[df['YearMade']>1930]

#create X and y
y = df['SalePrice']
X = df.drop(['SalePrice'], axis = 1)



#impute values for Machine hours
df_usage=X[['MachineHoursCurrentMeter','YearMade']]

df_usage_vals = df_usage.dropna()

mask_missing = df_usage.isnull().any(axis=1)
df_usage_missing = df_usage[mask_missing]

y_impute = df_usage_vals['MachineHoursCurrentMeter']
x_impute = df_usage_vals.drop(['MachineHoursCurrentMeter'], axis = 1)

#create a Random Forest to fill in machine hour nans
rf_usage = RandomForestRegressor()
rf_usage.fit(x_impute, y_impute)
preds_usage  = rf_usage.predict(df_usage_missing.drop(['MachineHoursCurrentMeter'], axis = 1))

#smoosh them back together
df_usage_missing['MachineHoursCurrentMeter']=pd.Series(preds_usage, index=df_usage_missing.index)
df_usage_total= pd.concat([df_usage_missing, df_usage_vals], axis=0)

# add usages with datasources
ds = df[[121,132,136,149,172, 'SalesID']]
df1 = pd.concat([df_usage_total, ds], axis=1)

#split data
X_train, X_test, y_train, y_test = train_test_split(df1,y, random_state=1)


#Random Forest Model
# rf = RandomForestRegressor()
# rf.fit(X_train.drop('SalesID', axis=1),y_train)
# yhat = rf.predict(X_test.drop('SalesID', axis=1))
#
# rf_cv_mse_scores = cross_val_score(rf, X_test.drop('SalesID', axis=1), y_test, scoring='mean_squared_error', cv=10)
#
# rf_cv_mse = np.mean(np.abs(rf_cv_mse_scores))
#
# #Boosting Model
#
# aba = AdaBoostRegressor()
# aba.fit(X_train.drop('SalesID', axis=1),y_train)
# aba_hat = aba.predict(X_test.drop('SalesID', axis=1))
#
# aba_cv_mse_scores = cross_val_score(aba, X_test.drop('SalesID', axis=1), y_test, scoring='mean_squared_error', cv=10)
#
# aba_cv_mse = np.mean(np.abs(aba_cv_mse_scores))
#
#
# sales_ids = np.savetxt(np.array(X_test['SalesID']))
# rf_preds = np.savetxt('rf_preds.csv', (X_test['SalesID'],yhat)
# ada_preds = np.savetxt('ada_preds.csv', aba_hat)

#Linear Regression Model
lr = LinearRegression()
lr.fit(X_train.drop('SalesID', axis=1),y_train)
lr_hat = lr.predict(X_test.drop('SalesID', axis=1))

lr_preds = np.savetxt('lr_preds.csv', lr_hat)

def year_v_price(df):

    pvy = df[['YearMade', 'SalePrice']]
    pvy = pvy[pvy.YearMade > 1930]
    plt.scatter(pvy.YearMade, pvy.SalePrice)
    lr = LinearRegression()
    lr.fit(pvy.YearMade.reshape(len(pvy.YearMade), 1), \
        pvy.SalePrice.reshape(len(pvy.SalePrice), 1))
    preds = lr.predict(pvy.YearMade)
    sale_id = df[['SalesID', 'YearMade']]
    sale_id = sale_id[sale_id.YearMade > 1930]
    sale_id = sale_id['SalesID']
    np.savetxt('line_preds.csv', preds)
    np.savetxt('sale_id.csv', sale_id)
    #This is super sloppy!
