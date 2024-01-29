import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
import graphviz
from matplotlib import pyplot as plt
from sklearn.tree import export_text

url = '/Users/qiavo/Documents/A datathon/catA_train.xlsx'
df = pd.read_excel(url)

df = df[df.columns.drop(list(df.filter(regex='Unnamed')))] #Removes the unnamed column from the data
#df.tail() #Displays the first 5 rows of the data


df = df.drop(columns=["AccountID", "8-Digit SIC Code", "8-Digit SIC Description",
                      "Square Footage", "Fiscal Year End",
                      "Company Status (Active/Inactive)"]) #Removes the columns that are not needed

df.fillna(value = {
        "LONGITUDE" : 0,
        "LATITUDE" : 0,
        "Import/Export Status" : "NA",
        "Global Ultimate Company" : "NA",
        "Global Ultimate Country" : "NA",
        "Domestic Ultimate Company" : "NA",

        }
        , inplace = True)

#cleaning data for employees single site column
site_cond = (df["Employees (Domestic Ultimate Total)"] == df["Employees (Global Ultimate Total)"]) & df['Employees (Single Site)'].isna()
df.loc[site_cond, 'Employees (Single Site)'] = df.loc[:,"Employees (Domestic Ultimate Total)"]

df["Employees (Single Site)"] = df.groupby("Parent Company")["Employees (Single Site)"].transform(lambda x: x.fillna(x.max())) #fill it with their parents company num of employee (single site)
#fill remaining as zero

#cleaning data for employee domestic ultimate columns
data_cond = df['Employees (Domestic Ultimate Total)'].isna()
dom_cond_one = (df["Is Domestic Ultimate"] == 0) & data_cond
df.loc[dom_cond_one, 'Employees (Domestic Ultimate Total)'] = 0

dom_cond_two = (df["Is Domestic Ultimate"] == 1) & (df["Is Global Ultimate"] == 1) & data_cond
df.loc[dom_cond_two, 'Employees (Domestic Ultimate Total)'] = df.loc[:, "Employees (Single Site)"]

 #remaining all same as single
df.loc[data_cond, 'Employees (Domestic Ultimate Total)'] = df.loc[:, "Employees (Single Site)"]


#cleaning data for employee global
glo_cond_one = (df["Is Domestic Ultimate"] == 1) & (df["Is Global Ultimate"] == 1) & df['Employees (Global Ultimate Total)'].isna()
df.loc[glo_cond_one, 'Employees (Global Ultimate Total)'] = df.loc[:, "Employees (Single Site)"]
df.fillna(value = {
        "Employees (Global Ultimate Total)" : 0,
        "Employees (Single Site)" : 0,
        "Employees (Domestic Ultimate Total)" : 0,
        }, inplace = True)

df["LATITUDE"] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df["SIC Code"] = df["SIC Code"].astype("Int64")
df["Year Found"] = df["Year Found"].astype("Int64")
df["Employees (Single Site)"] = df["Employees (Single Site)"].astype("Int64")
df["Employees (Domestic Ultimate Total)"] = df["Employees (Domestic Ultimate Total)"].astype("Int64")
df["Employees (Global Ultimate Total)"] = df["Employees (Global Ultimate Total)"].astype("Int64")
df["Is Domestic Ultimate"] = df["Is Domestic Ultimate"].astype("Int64")
df["Is Global Ultimate"] = pd.to_numeric(df['Is Global Ultimate'], errors='coerce').astype("Int64")

x1 = df["SIC Code"]
x2 = df['Sales (Domestic Ultimate Total USD)']
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# Create a range of values for k
k_range = range(1, 4)

# Initialize an empty list to
# store the inertia values for each k
inertia_values = []

# Fit and plot the data for each k value
for k in k_range:
	kmeans = KMeans(n_clusters=k, \
					init='k-means++', random_state=42)
	y_kmeans = kmeans.fit_predict(X)
	inertia_values.append(kmeans.inertia_)
	plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
	plt.scatter(kmeans.cluster_centers_[:, 0],\
				kmeans.cluster_centers_[:, 1], \
				s=100, c='red')
	plt.title('K-means clustering (k={})'.format(k))
	plt.xlabel('SIC Code')
	plt.ylabel('Sales (Domestic Ultimate Total USD)')
	plt.show()

# Plot the inertia values for each k
plt.plot(k_range, inertia_values, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

x11 = df["SIC Code"]
x22 = df['Sales (Global Ultimate Total USD)']
X1 = np.array(list(zip(x11, x22))).reshape(len(x11), 2)

# Create a range of values for k
k_range1 = range(1, 5)

# Initialize an empty list to 
# store the inertia values for each k
inertia_values1 = []

# Fit and plot the data for each k value
for k in k_range1:
	kmeans1 = KMeans(n_clusters=k, \
					init='k-means++', random_state=42)
	y_kmeans1 = kmeans1.fit_predict(X1)
	inertia_values1.append(kmeans1.inertia_)
	plt.scatter(X1[:, 0], X1[:, 1], c=y_kmeans1)
	plt.scatter(kmeans1.cluster_centers_[:, 0],\
				kmeans1.cluster_centers_[:, 1], \
				s=100, c='red')
	plt.title('K-means clustering (k={})'.format(k))
	plt.xlabel('SIC code')
	plt.ylabel('Sales (Global Ultimate Total USD)')
	plt.show()

# Plot the inertia values for each k
plt.plot(k_range1, inertia_values1, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

#---------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




df = df.drop(columns=['LATITUDE',
'LONGITUDE',
'Company',
'Industry',
'Year Found',
'Entity Type',
'Parent Company',
'Ownership Type',
'Company Description',
'Import/Export Status',
'Global Ultimate Company',
'Global Ultimate Country',
'Domestic Ultimate Company',
'Is Domestic Ultimate',
'Is Global Ultimate'])


# Encode categorical variables using one-hot encoding
df1 = pd.get_dummies(df, columns=['Parent Country'])
df2 = df1.copy()

#For global
# Normalize the numerical variables
numerical_cols = ['Employees (Single Site)', 'Employees (Domestic Ultimate Total)', 'Employees (Global Ultimate Total)', 'Sales (Domestic Ultimate Total USD)']

df1[numerical_cols] = (df1[numerical_cols] - df1[numerical_cols].mean()) / df1[numerical_cols].std()

#print(df1.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df1.drop('Sales (Global Ultimate Total USD)', axis=1), df1['Sales (Global Ultimate Total USD)'], test_size=0.2, random_state=42)

# Train and evaluate linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_mse = mean_squared_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

# Train and evaluate decision tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_preds)
dt_mse = mean_squared_error(y_test, dt_preds)
dt_r2 = r2_score(y_test, dt_preds)

# Train and evaluate neural network
nn = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500)
nn.fit(X_train, y_train)
nn_preds = nn.predict(X_test)
nn_mae = mean_absolute_error(y_test, nn_preds)
nn_mse = mean_squared_error(y_test, nn_preds)
nn_r2 = r2_score(y_test, nn_preds)

# Print the evaluation metrics for each model
print('Linear Regression - MAE:', lr_mae, 'MSE:', lr_mse, 'R-squared:', lr_r2)
print('Decision Tree - MAE:', dt_mae, 'MSE:', dt_mse, 'R-squared:', dt_r2)
print('Neural Network - MAE:', nn_mae, 'MSE:', nn_mse, 'R-squared:', nn_r2)


# Print the coefficients and intercept of the linear regression model
print('Coefficients:', lr.coef_)
print('Intercept:', lr.intercept_)


# Evaluate the performance of the model on the test set
lr_preds = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_mse = mean_squared_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

# Print the evaluation metrics for the model
print('Linear Regression - MAE:', lr_mae, 'MSE:', lr_mse, 'R-squared:', lr_r2)


#For domestic
# Normalize the numerical variables

numerical_cols2 = ['Employees (Single Site)', 'Employees (Domestic Ultimate Total)', 'Employees (Global Ultimate Total)', 'Sales (Global Ultimate Total USD)']

df2[numerical_cols2] = (df2[numerical_cols2] - df2[numerical_cols2].mean()) / df2[numerical_cols2].std()

#print(df1.head())

# Split the data into training and testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(df2.drop('Sales (Domestic Ultimate Total USD)', axis=1), df2['Sales (Domestic Ultimate Total USD)'], test_size=0.2, random_state=42)

# Train and evaluate linear regression
lr2 = LinearRegression()
lr2.fit(X_train2, y_train2)
lr2_preds = lr2.predict(X_test2)
lr2_mae = mean_absolute_error(y_test2, lr2_preds)
lr2_mse = mean_squared_error(y_test2, lr2_preds)
lr2_r2 = r2_score(y_test2, lr2_preds)

# Train and evaluate decision tree
dt2 = DecisionTreeRegressor()
dt2.fit(X_train2, y_train2)
dt2_preds = dt2.predict(X_test2)
dt2_mae = mean_absolute_error(y_test2, dt2_preds)
dt2_mse = mean_squared_error(y_test2, dt2_preds)
dt2_r2 = r2_score(y_test2, dt2_preds)

# Train and evaluate neural network
nn2 = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500)
nn2.fit(X_train2, y_train2)
nn2_preds = nn2.predict(X_test2)
nn2_mae = mean_absolute_error(y_test2, nn2_preds)
nn2_mse = mean_squared_error(y_test2, nn2_preds)
nn2_r2 = r2_score(y_test2, nn2_preds)

# Print the evaluation metrics for each model
print('Linear Regression - MAE:', lr2_mae, 'MSE:', lr2_mse, 'R-squared:', lr2_r2)
print('Decision Tree - MAE:', dt2_mae, 'MSE:', dt2_mse, 'R-squared:', dt2_r2)
print('Neural Network - MAE:', nn2_mae, 'MSE:', nn2_mse, 'R-squared:', nn2_r2)


# Print the coefficients and intercept of the linear regression model
print('Coefficients:', lr2.coef_)
print('Intercept:', lr2.intercept_)


# Evaluate the performance of the model on the test set
lr2_preds = lr2.predict(X_test2)
lr2_mae = mean_absolute_error(y_test, lr2_preds)
lr2_mse = mean_squared_error(y_test2, lr2_preds)
lr2_r2 = r2_score(y_test2, lr2_preds)

# Print the evaluation metrics for the model
print('Linear Regression - MAE:', lr2_mae, 'MSE:', lr2_mse, 'R-squared:', lr2_r2)


