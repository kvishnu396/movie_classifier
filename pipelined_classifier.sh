echo 'Generating the problem matrix'
echo 'This should take ~10 minutes'
python3 problem_generator.py
echo 'Problem matrix is successfully generated'
echo 'Training the classifier'
python3 classifier_trainer.py
echo 'Generating a test matrix'
python3 test.py
echo 'Calculating accuracy on the test matrix'
python3 classifier.py
