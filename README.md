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

## AUC metric is used to evalutae the performance of classifires. 
### How to use it
Run: `train.py

## Sample output
Classifiers are sorted by the descending order of their accuracy values
Rank 1: An ensemble method by Soft Voting/Majority Rule classifier-AUC score (test): 0.7698976370851371
Rank 2: Gradient Boosting Classifier-AUC score (test): 0.7494981843942843
Rank 3: Random Forest Classifier-AUC score (test): 0.7325280239642828
Rank 4: Logistic Regression Classifier-AUC score (test): 0.7270679816261865
Rank 5: Extra Trees Classifier-AUC score (test): 0.7221539202152788
Rank 6: Gaussian Naive Bayes-AUC score (test): 0.6774939357511388
Rank 7: Decision Tree Classifier-AUC score (test): 0.5970520919376237
Rank 8: A feed-forward neural network with a single hidden layer-AUC score (test): 

