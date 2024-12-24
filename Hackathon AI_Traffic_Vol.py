# %% [markdown]
# About the Dataset
# The dataset provides critical features like time, weather conditions, holiday information, and other contextual factors that affect traffic flow. Your task is to use these features to predict the traffic volume at a given point in time, creating a highly accurate forecasting model.
# 
# Dataset Description:
# 
# Train.csv: Contains historical data with traffic volume and associated features.
# Test.csv: The dataset on which you will generate your traffic predictions.
# Submission.csv: The format in which your predictions should be submitted.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# %%
train =    pd.read_csv(r"D:\Data Science\Hackathon Projects\Machine Hack AI\Dataset (4)\Dataset\Train.csv")
train.head()

# %%
test =  pd.read_csv(r"D:\Data Science\Hackathon Projects\Machine Hack AI\Dataset (4)\Dataset\Test.csv")
test.head()

# %%
train.shape, test.shape

# %%
train["Holiday"].value_counts()

# %%
train.isnull().sum()

# %%
test["Holiday"].value_counts()

# %%
combined = pd.concat([train, test], axis=0,ignore_index=True)
combined.head()

# %%
#Since Holiday has More Number of Values 
combined.drop(["Holiday"], axis=1, inplace=True)    

# %%
combined.shape

# %%
sns.distplot(combined["Traffic_Vol"])
plt.show()

# %%
combined.head(6)

# %%
combined["Weather"].value_counts()

# %%
def group_weather_conditions(weather):
    if weather in ['Clear skies']:
        return 'Clear Conditions'
    elif weather in ['Cloudy skies', 'Light fog', 'Dense fog']:
        return 'Cloudy Conditions'
    elif weather in ['Rainfall', 'Light rain']:
        return 'Rain'
    elif weather in ['Stormy weather', 'Sudden windstorm']:
        return 'Stormy'
    elif weather in ['Snowfall']:
        return 'Snowy'
    else:
        return 'Other'

combined['Weather_Group'] = combined['Weather'].apply(group_weather_conditions)
combined.head()

# %%
combined["Weather_Group"].value_counts()

# %%
combined["Extract_Hour"] = pd.to_datetime(combined["TimeStamp"]).dt.hour

# %%
combined.head()

# %%
# Extracting day, month, and year from the Date column
combined['Day'] = pd.to_datetime(combined['Date']).dt.day
combined['Month'] = pd.to_datetime(combined['Date']).dt.month
combined['Year'] = pd.to_datetime(combined['Date']).dt.year

# %%
#Aggregating Weather Conditions by Date
combined["Magic_1"] = combined.groupby("Date")["Temperature"].transform("mean")
combined["Magic_2"] = combined.groupby("Date")["Temperature"].transform("std")
combined["Magic_3"] = combined.groupby("Date")["Temperature"].transform("max")
combined["Magic_4"] = combined.groupby("Date")["Temperature"].transform("min")
combined["Magic_5"] = combined.groupby("Date")["Temperature"].transform("median")

# %%
combined["Magic_6"] = combined.groupby("Date")["Rainfall_last_hour"].transform("mean")
combined["Magic_7"] = combined.groupby("Date")["Rainfall_last_hour"].transform("std")
combined["Magic_8"] = combined.groupby("Date")["Rainfall_last_hour"].transform("max")
combined["Magic_9"] = combined.groupby("Date")["Rainfall_last_hour"].transform("min")
combined["Magic_10"] = combined.groupby("Date")["Rainfall_last_hour"].transform("median")


# %%
combined["Magic_11"] = combined.groupby("Date")["Snowfall_last_hour"].transform("mean")
combined["Magic_12"] = combined.groupby("Date")["Snowfall_last_hour"].transform("std")
combined["Magic_13"] = combined.groupby("Date")["Snowfall_last_hour"].transform("max")
combined["Magic_14"] = combined.groupby("Date")["Snowfall_last_hour"].transform("min")
combined["Magic_15"] = combined.groupby("Date")["Snowfall_last_hour"].transform("median")

# %%
combined["Magic_16"] = combined.groupby("Date")["Cloud_Cover"].transform("mean")
combined["Magic_17"] = combined.groupby("Date")["Cloud_Cover"].transform("std")
combined["Magic_18"] = combined.groupby("Date")["Cloud_Cover"].transform("max")
combined["Magic_19"] = combined.groupby("Date")["Cloud_Cover"].transform("min")
combined["Magic_20"] = combined.groupby("Date")["Cloud_Cover"].transform("median")

# %%
combined.shape

# %%
pd.set_option('display.max_columns', None)
combined.head()

# %%
def categorize_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

combined['Time_of_Day'] = combined['Extract_Hour'].apply(categorize_time_of_day)
combined.head()

# %%
def categorize_weekday(date):
    day_of_week = pd.to_datetime(date).dayofweek
    if day_of_week < 5:
        return 'Weekday'
    else:
        return 'Weekend'

combined['Weekday_Weekend'] = combined['Date'].apply(categorize_weekday)
combined.head()

# %%
combined['Traffic_Vol_Lag_1'] = combined['Traffic_Vol'].shift(1)
combined['Traffic_Vol_Lag_2'] = combined['Traffic_Vol'].shift(2)
combined['Traffic_Vol_Lag_3'] = combined['Traffic_Vol'].shift(3)
combined.head()

# %%
combined['Traffic_Vol_Lag_1'] = combined['Traffic_Vol_Lag_1'].fillna(combined['Traffic_Vol_Lag_1'].mean())
combined['Traffic_Vol_Lag_2'] = combined['Traffic_Vol_Lag_2'].fillna(combined['Traffic_Vol_Lag_2'].mean())
combined['Traffic_Vol_Lag_3'] = combined['Traffic_Vol_Lag_3'].fillna(combined['Traffic_Vol_Lag_3'].mean())

# %%
combined.head()

# %%
combined.drop(['Date', 'TimeStamp', 'Weather','Weather_Desc'], axis=1, inplace=True)

# %%
def return_splits(ddf, feature_name, target_name):
    return [ddf[ddf[feature_name] == i][target_name] for i in ddf[feature_name].unique()]

def give_stats_analysis(df, target_column_name):
    from scipy.stats import ttest_ind, chi2_contingency, kruskal
    ddf = df.copy()
    ddf = ddf.dropna()

    features = []
    tests = []
    stats = []
    pvals = []
    count = 0

    target = ddf[target_column_name]
    for i in ddf.columns:
        features.append(i)
        feature = ddf[i]
        
        if (feature.dtype == "O" and (target.dtype == "float" or target.dtype == "int")) or (target.dtype == "O" and (feature.dtype == "float" or feature.dtype == "int")):
            stat, pval, *_ = kruskal(*return_splits(ddf, feature.name, target.name))
            tests.append("Kruskal-Wallis")
            stats.append(stat)
            pvals.append(pval)
            
        
        elif (feature.dtype == "float" or feature.dtype == "int") and (target.dtype == "float" or target.dtype == "int"):
            stat, pval, *_ = ttest_ind(feature, target)
            tests.append("TTest")
            stats.append(stat)
            pvals.append(pval)

        elif feature.dtype == "O" and target.dtype == "O":
            stat, pval, *_ = chi2_contingency(pd.crosstab(feature, target))
            tests.append("Chi-Square")
            stats.append(stat)
            pvals.append(pval)
        
        else:
            tests.append(np.nan)
            stats.append(np.nan)
            pvals.append(np.nan)

        print(f"{feature.name} ■■■ {target_column_name}".ljust(50, "-")+"✅")
    
    return pd.DataFrame({
        "Feature" : features,
        "Target" : [target_column_name]*ddf.shape[1],
        "Statistic Test" : tests,
        "Test Statistic" : stats,
        "P-Value" : pvals
    }).sort_values(by="P-Value")

# %%
give_stats_analysis(combined, "Traffic_Vol")

# %%
combined.head()

# %%
combined = pd.get_dummies(combined,drop_first=True,dtype=int)    

# %%
combined.head()

# %%
# Fill NaN values with the mean for numerical columns except 'Traffic_Vol'
for column in combined.columns:
    if combined[column].dtype in [np.float64, np.int64] and column != 'Traffic_Vol':
        combined[column].fillna(combined[column].mean(), inplace=True)

# Verify if there are any NaN values left
combined.isnull().sum()

# %%
combined.drop(["Traffic_Vol_Lag_1","Traffic_Vol_Lag_2","Traffic_Vol_Lag_3"], axis=1, inplace=True)

# %%
newtrain = combined.loc[0:train.shape[0]-1,:]
newtest = combined.loc[train.shape[0]:,:]


# %%
newtrain.shape, newtest.shape , train.shape, test.shape

# %%
newtest.drop(["Traffic_Vol"], axis=1, inplace=True)

# %% [markdown]
# ### Stats Analysis

# %%
from scipy.stats import stats

# %%
newtrain.select_dtypes(include=[np.number]).columns

# %%
num = ['Magic_1', 'Magic_2', 'Magic_3', 'Magic_4', 'Magic_5', 'Magic_6',
       'Magic_7', 'Magic_8', 'Magic_9', 'Magic_10', 'Magic_11', 'Magic_12',
       'Magic_13', 'Magic_14', 'Magic_15', 'Magic_16', 'Magic_17', 'Magic_18',
       'Magic_19', 'Magic_20']

# %%
#Ho : predictor and Target are independent
#Ha : predictor and Target are dependent

pvalue = []

for i in num:
    teststats,pval = stats.ttest_ind(newtrain[i],newtrain["Traffic_Vol"])
    pvalue.append([i,pval])

# %%
pvalue

# %%
#K Fold Model
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

# %%
kfold = KFold(n_splits=20,shuffle=True)

# %%
from sklearn.metrics import root_mean_squared_error
import numpy as np





# %%
from sklearn.model_selection import train_test_split

# Define features and target variable
X = newtrain.drop('Traffic_Vol', axis=1)
y = newtrain['Traffic_Vol']

# Perform train-test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape

# %%


# %%
#Lasso and Ridge
from sklearn.linear_model import Lasso, Ridge

#Machine Instance
ridge = Ridge(alpha=4000)
pred = ridge.fit(X, y).predict(newtest)
pred

#Analysis of Weights

weights = pd.DataFrame(ridge.coef_, columns=["Weights"], index=X.columns)
weights.sort_values(by="Weights", ascending=False).plot(kind="bar")
plt.title("Alpha is 4000")


# %%
#Lasso and Ridge
from sklearn.linear_model import Lasso, Ridge

#Machine Instance
lasso = Lasso(alpha=4000)
pred = lasso.fit(X, y).predict(newtest)
pred

#Analysis of Weights

weights = pd.DataFrame(lasso.coef_, columns=["Weights"], index=X.columns)
weights.sort_values(by="Weights", ascending=False).plot(kind="bar")
plt.title("Alpha is 4000")


# %%


# %%
#Elastic Net
from sklearn.linear_model import ElasticNet

#Machine Instance
enet = ElasticNet(alpha=7,l1_ratio=1)
pred = enet.fit(X,y).predict(newtest)
pred

#Analysis of Weights

weights = pd.DataFrame(enet.coef_,columns=["Weights"],index=X.columns)
weights.sort_values(by="Weights",ascending=False).plot(kind="bar")



# %%


# %%
lr = LinearRegression()
pred = []
for train_index,test_index in kfold.split(X,y):
    xtrain  = X.iloc[train_index,:]
    xtest   = X.iloc[test_index,:]
    ytrain  = y.iloc[train_index]
    ytest   = y.iloc[test_index]
    #Modeling and Prediction
    pred_lr = lr.fit(xtrain,ytrain).predict(xtest)
    #Cost Function
    cost = root_mean_squared_error(ytest,pred_lr)
    print(cost)
    #Final Prediciton
    pred.append(lr.predict(xtest))

# %%
lr.coef_

# %%
# Define features and target variable
X = newtrain.drop('Traffic_Vol', axis=1)
y = newtrain['Traffic_Vol']

# Target Variable
gbm = GradientBoostingRegressor()
rf = RandomForestRegressor()
from sklearn.metrics import mean_squared_error
pred = []

for train_index, test_index in kfold.split(X, y):
    xtrain = X.iloc[train_index, :]
    xtest = X.iloc[test_index, :]
    ytrain = y.iloc[train_index]
    ytest = y.iloc[test_index]
    
    # Modeling and Prediction
    pred_rf = rf.fit(xtrain, ytrain).predict(xtest)
    # pred_gbm = gbm.fit(xtrain, ytrain).predict(xtest)
    
    # Cost Function
    cost_rf = np.sqrt(mean_squared_error(ytest, pred_rf))
    # cost_gbm = np.sqrt(mean_squared_error(ytest, pred_gbm))
    print(f'Random Forest RMSE: {cost_rf}')
    # print(f'Gradient Boosting RMSE: {cost_gbm}')
    
    # Final Prediction
    pred.append(pred_rf)

# %%
pred = rf.predict(newtest)

# %%
pred

# %%
submission = pd.DataFrame(pred).mean(axis=1)
submission 

# %%
submission1 = pd.read_csv(r"D:\Data Science\Hackathon Projects\Machine Hack AI\Dataset (4)\Dataset\Submission.csv")
submission1.head()

# %%
submission1["Traffic_Vol"] = submission

# %%
submission1

# %%
submission1.to_csv("Submission_RF_NEW_20_W.csv",index=False)

# %%


# %%
from sklearn.model_selection import GridSearchCV



# Hyperparameter tuning for RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1,verbose=3)
grid_search.fit(xtrain, ytrain)

# Best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the model with best parameters
best_rf = RandomForestRegressor(**best_params, random_state=42)
best_rf.fit(xtrain, ytrain)

# Predict and calculate RMSE
pred_rf = best_rf.predict(xtest)
rmse = np.sqrt(mean_squared_error(ytest, pred_rf))
print(f"RMSE: {rmse}")

# %%
pred_rf = best_rf.predict(newtest)

# %%
submission = pd.DataFrame(pred_rf).mean(axis=1)
submission 

# %%
submission1 = pd.read_csv(r"D:\Data Science\Hackathon Projects\Machine Hack AI\Dataset (4)\Dataset\Submission.csv")
submission1.head()
submission1["Traffic_Vol"] = submission

# %%
submission1

# %%
submission1.to_csv("Submission_RF_Best_Params.csv",index=False)

# %%



