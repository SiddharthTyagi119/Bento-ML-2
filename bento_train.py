import bentoml
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier()

iris=load_iris()
X=iris.data[:,:4]
Y=iris.target

model.fit(X,Y)

bentoml_model=bentoml.sklearn.save_model('Kneighbors',model)
print(f"Model saved:{bentoml_model}")