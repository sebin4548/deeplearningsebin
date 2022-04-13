#라이브러리를 이용하여 데이터 세트 가져오기
from sklearn import datasets
iris = datasets.load_iris()

from sklearn.model_selection import train_test_split
X = iris.data
y = iris.target
# data 배열은 n_samples * n_features 형식의 2차원 배열
# target 배열은 정답값

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)  
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)

# KNeighborsClassifier 이라는 클래스를 가져와서 knn이라는 객체를 생성. fit는 학습을 수행하는 함수
y_pred = knn.predict(X_test)

from sklearn import metrics
scores = metrics.accuracy_score(y_test, y_pred)
y_pred = knn.predict(X_test)

from sklearn import metrics
scores = metrics.accuracy_score(y_test, y_pred)
