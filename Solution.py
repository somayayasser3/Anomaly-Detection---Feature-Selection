import pandas as pd
from scipy.stats import chisquare
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
import itertools
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier

DataF = pd.read_csv("data.csv")
threshold = 0.8
dataFrame = DataF
output = DataF["diagnosis"]
features = dataFrame.drop(columns=dataFrame.columns[[0, 1]])


# print(features)


def dtClass(features):
    process = preprocessing.LabelEncoder()
    outProcess = preprocessing.LabelEncoder()

    for x in features:
        process.fit(features[x])
        features[x] = process.transform(features[x])

    outProcess.fit(output)
    outputY = outProcess.transform(output)

    lenFet = len(features)
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(features[:int(lenFet * 0.7)], outputY[:int(lenFet * 0.7)])
    outA = dt.predict(features[int(lenFet * 0.7):])
    acc = accuracy_score(outputY[int(lenFet * 0.7):], outA) * 100
    pre = precision_score(outputY[int(lenFet * 0.7):], outA)
    rec = recall_score(outputY[int(lenFet * 0.7):], outA)
    return pre, rec, acc


def svmClass(features):
    process = preprocessing.LabelEncoder()
    outProcess = preprocessing.LabelEncoder()

    for x in features:
        process.fit(features[x])
        features[x] = process.transform(features[x])

    outProcess.fit(output)
    outputY = outProcess.transform(output)
    clf = svm.SVC(kernel='linear')
    lenFet = len(features)
    clf.fit(features[:int(lenFet * 0.8)], outputY[:int(lenFet * 0.8)])
    outA = clf.predict(features[int(lenFet * 0.8):])
    acc = accuracy_score(outputY[int(lenFet * 0.8):], outA) * 100
    pre = precision_score(outputY[int(lenFet * 0.8):], outA)
    rec = recall_score(outputY[int(lenFet * 0.8):], outA)
    return pre, rec, acc


def logClass(features):
    process = preprocessing.LabelEncoder()
    outProcess = preprocessing.LabelEncoder()
    for x in features:
        process.fit(features[x])
        features[x] = process.transform(features[x])
    sc_x = StandardScaler()
    xtrain = sc_x.fit_transform(features)
    outProcess.fit(output)
    outputY = outProcess.transform(output)
    lenFet = len(features)
    classifier = LogisticRegression(random_state=0)
    classifier.fit(xtrain[:int(lenFet * 0.7)], outputY[:int(lenFet * 0.7)])
    outA = classifier.predict(xtrain[int(lenFet * 0.7):])
    # outB = outProcess.transform(outA)
    acc = accuracy_score(outputY[int(lenFet * 0.7):], outA) * 100
    pre = precision_score(outputY[int(lenFet * 0.7):], outA)
    rec = recall_score(outputY[int(lenFet * 0.7):], outA)
    return pre, rec, acc


def Method1Filter(feature):
    value = chisquare(feature)
    return value


def naiveBayes(features):
    process = preprocessing.LabelEncoder()
    outProcess = preprocessing.LabelEncoder()
    for x in features:
        process.fit(features[x])
        features[x] = process.transform(features[x])
    outProcess.fit(output)
    outputY = outProcess.transform(output)
    X_train, X_test, y_train, y_test = train_test_split(features, outputY, test_size=0.2, random_state=1)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    return pre, rec, acc


def method2Wrapper():
    newFeatures = {}
    outProcess = preprocessing.LabelEncoder()
    outProcess.fit(output)
    outputY = outProcess.transform(output)
    sfs = SFS(LinearRegression(),
              k_features=15,
              forward=True,
              floating=False,
              scoring='r2',
              cv=0)
    sfs.fit(features, outputY)

    for x in features:
        if x in sfs.k_feature_names_:
            newFeatures[x] = features[x]

    NewF = pd.DataFrame(newFeatures)
    return NewF


featureSelection = {}
for col in featureSelection:
    featureSelection[col] = []

for data in features:
    # cross = pd.crosstab(finalFeatures[data], finalFeatures['compactness_mean'])
    if Method1Filter(features[data])[1] > threshold:
        # print(Method1Filter(features[data])[1])
        featureSelection[data] = features[data]

featureSelection = pd.DataFrame(featureSelection)

selectedFeatures = method2Wrapper()
print(svmClass(selectedFeatures))
print(logClass(selectedFeatures))
print(dtClass(selectedFeatures))
print(naiveBayes(selectedFeatures))
