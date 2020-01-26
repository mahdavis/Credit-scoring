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

## Accuracy metric is used to evalutae the performance of classifires. 
Accuracy metric is the number of correct predictions made divided by the total number of predictions made.

### How to use it
Run: `train.py

## Sample output
Classifiers are sorted by the descending order of their accuracy values
Rank 1: An ensemble method by Soft Voting/Majority Rule classifier-Accuracy score (test): 0.767072715033866
Rank 2: Gradient Boosting Classifier-Accuracy score (test): 0.7443418703132818
Rank 3: Logistic Regression Classifier-Accuracy score (test): 0.7403727699009461
Rank 4: Random Forest Classifier-Accuracy score (test): 0.7320771745572275
Rank 5: Extra Trees Classifier-Accuracy score (test): 0.713382551010597
Rank 6: Gaussian Naive Bayes-Accuracy score (test): 0.6815624713479247
Rank 7: Decision Tree Classifier-Accuracy score (test): 0.5914128909514841
Rank 8: A feed-forward neural network with a single hidden layer-Accuracy score (test): 0.0001

