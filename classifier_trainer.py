from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from numpy import log, linspace
from pickle import load, dump

model = SVC(kernel='linear', cache_size=1e3)
cparams = list(map(lambda x : 10**x, linspace(-3,3,20)))
parameters = {'C':cparams}
classifier = GridSearchCV(model, parameters)
problem_labels = [1]*2500 + [0]*2500

with open('problem.matrix', 'rb') as f:
    problem_features = load(f)

classifier.fit(problem_features, problem_labels)

with open('classifier.svm', 'wb') as f:
    dump(classifier, f)
