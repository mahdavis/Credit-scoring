## Credit-scoring
##Give Me Some Credit Kaggle Python Code 
Simple library and command line utility for credit scoring by predicting the probability that somebody will experience financial distress in the next two years.
## Requirements
* Python 3.5+
* NumPy (`pip install numpy`)
* Pandas (`pip install pandas`)
* Keras (`pip install keras`)
* Tensorflow (`pip install tensorflow`)
* scikit (‘pip install scikit-learn’)
#########
###########
In this code I am trying to solve this classification challenge by implementing several classifiers: An ensemble method of three top rank methods; a gradient boosting algorithm, a regression random forest, and Logistic Regression Classifier by Soft Voting/Majority Rule classifier; Gradient Boosting Classifier, Random Forest Classifier,
Extra Trees Classifier, Logistic Regression Classifier, Gaussian Naive Bayes, Decision Tree Classifier, and A feed-forward neural network with a single hidden layer. 

## Accuracy metric is used to evalutae the performance classifires. 
Accuracy metric is the number of correct predictions made divided by the total number of predictions made.

### How to use it
Run: `train.py

## Sample output
Classifiers are sorted by the descending order of their accuracy values
Rank 1: An ensemble method by Soft Voting/Majority Rule classifier-Accuracy score (test): 0.9314459133616031
Rank 2: Random Forest Classifier-Accuracy score (test): 0.9312796208530806
Rank 3: Gradient Boosting Classifier-Accuracy score (test): 0.9311964745988193
Rank 4: Extra Trees Classifier-Accuracy score (test): 0.9292841107508106
Rank 5: Logistic Regression Classifier-Accuracy score (test): 0.9292841107508106
Rank 6: Gaussian Naive Bayes-Accuracy score (test): 0.9284942213353289
Rank 7: Decision Tree Classifier-Accuracy score (test): 0.8892907624511516
Rank 8: A feed-forward neural network with a single hidden layer-Accuracy score (test): 0.07

