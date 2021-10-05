
#Used for continous Data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')
dataset = pd.read_csv('/content/drive/My Drive/Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values  #independant variable Column
y = dataset.iloc[:, -1].values   #dependant variable Column

#spliting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the model. Fit method of the linear regression class does so
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predict= regressor.predict(X_test)

#Training Set esults
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs. Experience(Training Set)')
plt.xlabel('Experience(Years)') 
plt.ylabel('Salary($)')
plt.show()

#Testing Set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs. Experience(Test Set)')
plt.xlabel('Experience(Years)') 
plt.ylabel('Salary($)')
plt.show()

print(regressor.predict([[20]]))

