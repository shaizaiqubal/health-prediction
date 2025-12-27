import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def project_report(model, X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """
    Prints metrics, feature weights, and plots for a regression model.
    
    Parameters:
    - model: trained regression model (LinearRegression or SGDRegressor)
    - X_train_scaled, X_test_scaled: scaled train/test features (pandas DataFrame)
    - y_train, y_test: train/test target arrays
    - feature_names: list of feature names
    """
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print("==== Model Performance ====")
    print(f"Train MSE: {mse_train:.2f}, R²: {r2_train:.2f}")
    print(f"Test MSE:  {mse_test:.2f}, R²: {r2_test:.2f}")
    print()
    
    # Feature Weights
    print("==== Feature Weights ====")
    weights = model.coef_
    bias = model.intercept_
    print(f"Bias (intercept): {bias[0]:.3f}")
    for f, w in zip(feature_names, weights):
        print(f"{f}: {w:.3f}")
    
    # Scatter plot: Actual vs Predicted
    indices = np.arange(len(y_test))
    plt.figure(figsize=(8,5))
    plt.scatter(indices, y_test, color='blue', alpha=0.6, label='Actual')
    plt.scatter(indices, y_test_pred, color='orange', alpha=0.6, label='Predicted')
    plt.xlabel("Sample Index")
    plt.ylabel("Systolic BP")
    plt.title("Actual vs Predicted Blood Pressure")
    plt.legend()
    plt.show()
    
    # Feature importance bar plot
    plt.figure(figsize=(6,4))
    plt.bar(feature_names, weights)
    plt.ylabel("Weight Value")
    plt.title("Feature Importance")
    plt.show()


df=pd.read_csv('data/data.csv')
x=df[['age','BMI','exercise_frequency',]]
x['age2']=x['age']**2
x['bme']=x['BMI']*x['exercise_frequency']
x['lbmi']=np.log(x['BMI'])

y=df['systolic_bp']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

sdgr=SGDRegressor(random_state=40)
sdgr.fit(x_train_scaled,y_train)
#print(f"iterations:{sdgr.n_iter_} \n weights: {sdgr.t_}")

b=sdgr.intercept_
w=sdgr.coef_

#print(f"parameters: w: {w}, b:{b}")

yt_pred=sdgr.predict(x_train_scaled)
y_pred=sdgr.predict(x_test_scaled)

#print(f"training mse={mean_squared_error(y_train,yt_pred)}\n test mse={mean_squared_error(y_test,y_pred)}")

#project_report(sdgr,x_train_scaled,x_test_scaled,y_train,y_test,feature_names=x.columns.to_list())


tree_model=DecisionTreeRegressor(max_depth=1, random_state=40)
tree_model.fit(x_train_scaled,y_train)

yt_pred_tree=tree_model.predict(x_train_scaled)
y_pred_tree=tree_model.predict(x_test_scaled)

#print(f"training mse={mean_squared_error(y_train,yt_pred_tree)}\n test mse={mean_squared_error(y_test,y_pred_tree)}")

randofor=RandomForestRegressor(n_estimators=100,max_depth=2, random_state=40)
randofor.fit(x_train_scaled,y_train)

yt_pred_for=randofor.predict(x_train_scaled)
y_pred_for=randofor.predict(x_test_scaled)

print(f"training mse={mean_squared_error(y_train,yt_pred_for)}\n test mse={mean_squared_error(y_test,y_pred_for)}")

indices = np.arange(len(y_test))
plt.figure(figsize=(8,5))
plt.scatter(indices, y_test, color='blue', alpha=0.6, label='Actual')
plt.scatter(indices, y_pred_for, color='orange', alpha=0.6, label='Predicted')
plt.xlabel("Sample Index")
plt.ylabel("Systolic BP")
plt.title("Actual vs Predicted Blood Pressure")
plt.legend()
plt.show()