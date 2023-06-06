from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
 
# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
 
# Train the model using the training sets
knn.fit(X_train, y_train)
 
# Predict the response for test dataset
y_pred = knn.predict(X_test)
 
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", knn.score(X_test, y_test))
