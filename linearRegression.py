from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def run_linear_regression():
    df = pd.read_csv("Boston_Housing.csv")
    #print(df.head())
    #print(df.info())
    #print(df.isnull().sum())

    #Check correlation
    # plt.figure(figsize=(10,15))
    # sns.heatmap(df.corr(), cmap= "YlGnBu", annot=True)
    # plt.show()

    #Select features
    x = df[['RM']]
    y = df['MEDV']

    # print(x)
    # print(y)

    #Split training data
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 42)

    # print(x_train.shape)
    # print(x_test.shape)
    # print(x_train)
    # print(y_train)

    #Initialize Linear Regression model
    model = LinearRegression()

    #Train model
    model.fit(x_train, y_train)
    # print(model.coef_ , " ", model.intercept_)

    #Test model
    y = model.predict(x_test)
    # print(y)
    # print(y_test)


    #Check mean_squared_error
    print("Mean Squared Error is : ",mean_squared_error(y, y_test))
    print("Root Mean Squared Error is : ",root_mean_squared_error(y, y_test))

if __name__ == "__main__":
    run_linear_regression()

