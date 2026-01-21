from sklearn.linear_model import LinearRegression

import numpy as np
X=np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
Y=np.array[30, 50, 70, 90, 110, 130, 150, 170, 180, 200]

model=LinearRegression()
model.fit(X, Y)
print(model.predict([[6]])) 
print(type(X))
print(type(Y))