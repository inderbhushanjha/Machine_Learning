import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data=pd.read_csv("student-mat.csv", sep=";")
#print(data.head())

data=data[["G1","G2","G3","absences","failures"]]
#print(data.head())
predict="G3"

x=np.array(data.drop([predict],1))
y=np.array(data[predict])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)


'''linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
acc = linear.score(x_test,y_test)
print(acc)
with open("saved_model.pickle","wb") as f:
    pickle.dump(linear,f)'''
pickle_input=open("saved_model.pickle", "rb")
linear = pickle.load(pickle_input)

print("coefficient : ", linear.coef_)
print("intercept : ",linear.intercept_)


prediction = linear.predict(x_test)


for x in range (len(prediction)):
    print(prediction[x], x_test[x], y_test[x])