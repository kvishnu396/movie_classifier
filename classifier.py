from pickle import load, dump

problem_labels = [1]*500 + [0]*500

with open('test.matrix', 'rb') as f:
    problem_features = load(f)

with open('classifier.svm', 'rb') as f:
    classifier = load(f)


predicted_labels = classifier.predict(problem_features)

n = 0
for x, y in zip(problem_labels, predicted_labels):
    if x == y:
        n += 1

print(n/1e3)
