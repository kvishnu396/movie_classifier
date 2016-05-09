from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from numpy import log, linspace

model = LinearSVC()
cparams = list(map(lambda x : 10**x, linspace(-3,3,20)))
parameters = {'C':cparams}
classifier = GridSearchCV(estimator, parameters)

