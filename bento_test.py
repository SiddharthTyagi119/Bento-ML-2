import bentoml

#it will load the model from dir itself
clf=bentoml.sklearn.get('kneighbors:latest').to_runner()

#initialize the classifier
clf.init_local()

#do prediction
result=clf.predict.run([[2.4,1.4,3.5,4.6]])

print(result)