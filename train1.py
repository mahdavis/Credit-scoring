import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score,roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import backend,optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.naive_bayes import GaussianNB
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
###########################
def forward_propagation(X_train, Y_train):
    clf_fp = Sequential()
    # First Hidden Layer
    clf_fp.add(Dense(64, activation='relu', kernel_initializer='random_normal', input_dim=X_train.shape[1]))
    # Output Layer
    clf_fp.add(Dense(1, activation='softmax', kernel_initializer='random_normal'))
    # Compiling the neural network
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam1 = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    clf_fp.compile(optimizer=adam1, loss='binary_crossentropy', metrics=['accuracy'])
    clf_fp.fit(X_train, Y_train, batch_size = 20, epochs = 10,verbose=0)
    return clf_fp
###############################
def Gradient_Boosting_Classifier(X_train, Y_train):
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.25, max_features=2, max_depth=2,
                                        random_state=0)
    gb_clf.fit(X_train, Y_train)

    return gb_clf
########################################
def Random_Forest_Classifier(X_train, Y_train):
    clf_rfc = RandomForestClassifier(n_estimators=100)
    clf_rfc.classes_ = 2
    clf_rfc = clf_rfc.fit(X_train, Y_train)
    return clf_rfc
#####################################
def Logistic_Regression(X_train, Y_train):
    clf_lr = LogisticRegression(random_state=0).fit(X_train, Y_train)
    return clf_lr
#######################################
def Extra_Trees_Classifier(X_train, Y_train):
    clf_etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf_etc.fit(X_train, Y_train)
    return clf_etc
########################################
def Decision_Tree_Classifier(X_train, Y_train):
    clf_dtc = DecisionTreeClassifier()
    clf_dtc = clf_dtc.fit(X_train, Y_train)
    return clf_dtc
##############################################
def Gaussian_NB(X_train, Y_train):
    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train, Y_train)
    return clf_gnb
###############################################
def Voting_Classifier(X_train, Y_train):
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.25, max_features=2, max_depth=2,
                                        random_state=0)
    # clf_gnb = GaussianNB()
    # clf_dtc = DecisionTreeClassifier()
    # clf_etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf_lr = LogisticRegression(random_state=0)
    clf_rfc = RandomForestClassifier(n_estimators=100)
    # eclf1 = VotingClassifier(estimators=[
    #     ('gb_clf', gb_clf), ('clf_gnb', clf_gnb), ('clf_dtc', clf_dtc),('clf_etc', clf_etc),('clf_lr', clf_lr),('clf_rfc', clf_rfc)], voting='hard')
    eclf1 = VotingClassifier(estimators=[
        ('gb_clf', gb_clf),  ('clf_lr', clf_lr),
        ('clf_rfc', clf_rfc)], voting='hard')

    eclf1 = eclf1.fit(X_train, Y_train)
    return eclf1
###############################################
def preprocess_data():
    # load data
    data_path = 'Data/cs-training.csv'
    features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    label = 'SeriousDlqin2yrs'
    data = pd.read_csv(data_path)
    #Drop the rows where at least one element is missing
    data = data.dropna(axis=0, how='any')
    # num_data = X.shape[0]
    X = data[features]
    Y = data[label]
    # Split data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    # scales and translates each feature
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test
#######################################
def Run_classifier():
    list_acc = []
    # list_auc = []
    list_nameclf = []
    ############ Run Gradient Boosting Classifier
    gb_clf = Gradient_Boosting_Classifier(X_train, Y_train)
    predictions = gb_clf.predict(X_test)
    list_acc.append(accuracy_score(predictions, Y_test))
    # list_auc.append(roc_auc_score(predictions, Y_test))
    list_nameclf.append('Gradient Boosting Classifier')
    ############ Run Random Forest Classifier
    clf_rfc = Random_Forest_Classifier(X_train, Y_train)
    predictions = clf_rfc.predict(X_test)
    list_acc.append(accuracy_score(predictions, Y_test))
    # list_auc.append(roc_auc_score(predictions, Y_test))
    list_nameclf.append('Random Forest Classifier')
    ############ Run Logistic Regression Classifier
    clf_lr = Logistic_Regression(X_train, Y_train)
    predictions = clf_lr.predict(X_test)
    list_acc.append(accuracy_score(predictions, Y_test))
    # list_auc.append(roc_auc_score(predictions, Y_test))
    list_nameclf.append('Logistic Regression Classifier')
    ############### Run Extra Trees Classifier
    clf_etc = Extra_Trees_Classifier(X_train, Y_train)
    predictions = clf_etc.predict(X_test)
    list_acc.append(accuracy_score(predictions, Y_test))
    # list_auc.append(roc_auc_score(predictions, Y_test))
    list_nameclf.append('Extra Trees Classifier')
    ############### Run Decision Tree Classifier
    clf_dtc = Decision_Tree_Classifier(X_train, Y_train)
    predictions = clf_dtc.predict(X_test)
    list_acc.append(accuracy_score(predictions, Y_test))
    # list_auc.append(roc_auc_score(predictions, Y_test))
    list_nameclf.append('Decision Tree Classifier')
    ################Run Gaussian Naive Bayes
    clf_gnb = Gaussian_NB(X_train, Y_train)
    predictions = clf_gnb.predict(X_test)
    list_acc.append(accuracy_score(predictions, Y_test))
    # list_auc.append(roc_auc_score(predictions, Y_test))
    list_nameclf.append('Gaussian Naive Bayes')
    ################Run Soft Voting/Majority Rule classifier
    eclf1 = Voting_Classifier(X_train, Y_train)
    predictions = eclf1.predict(X_test)
    list_acc.append(accuracy_score(predictions, Y_test))
    # list_auc.append(roc_auc_score(predictions, Y_test))
    list_nameclf.append('An ensemble method by Soft Voting/Majority Rule classifier')
    ################Run A feed-forward neural network with a single hidden layer
    clf_fp = forward_propagation(X_train, Y_train)
    predictions = clf_fp.predict(X_test)
    list_acc.append(accuracy_score(predictions, Y_test))
    # list_auc.append(roc_auc_score(predictions, Y_test))
    list_nameclf.append('A feed-forward neural network with a single hidden layer')
    return list_acc,list_nameclf

if __name__ == "__main__":
    # main()
    X_train, X_test, Y_train, Y_test = preprocess_data()
    list_acc, list_nameclf = Run_classifier()
    sortindex_acc = np.argsort(np.array(list_acc))
    sortindex_acc1 = sortindex_acc[::-1]
    count_alg = 1
    print('Classifiers are sorted by the descending order of their accuracy values')
    for item in sortindex_acc1:
        print('Rank '+ str(count_alg)+': '+list_nameclf[item]+"-Accuracy score (test):",list_acc[item])
        count_alg += 1

