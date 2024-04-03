from sklearn.model_selection import train_test_split
from sklearn import datasets
iris =datasets.load_iris()
X = iris.data
y = iris.target
#(80,20)으로 분할한다.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
print(X_train.shape)
print(X_test.shape)