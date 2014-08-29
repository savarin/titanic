
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode
from datetime import datetime


if __name__ == '__main__':

    print "Loading training data."

    df = pd.read_csv('train.csv')

    print "Cleaning training data."

    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)

    mode_embarked = mode(df['Embarked'])[0][0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)

    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)

    df = df.drop(['Sex', 'Embarked'], axis=1)

    cols = df.columns.tolist()
    cols = [cols[1]] + cols[0:1] + cols[2:]
    df = df[cols]

    train_data = df.values
    X = train_data[:, 2:]
    y = train_data[:, 0]

    print "Loading test data."

    df_test = pd.read_csv('test.csv')

    print "Cleaning test data"

    df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    df_test['Age'] = df_test['Age'].fillna(age_mean)

    fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
    df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                                fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                                else x['Fare'], axis=1)

    df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'],
                                prefix='Embarked')], axis=1)

    df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

    X_submission = df_test.values[:, 1:]

    print "Preparing models."

    n_folds = 10
    verbose = True
    shuffle = True

    np.random.seed(1)

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=100, criterion='entropy',
                                    max_features=0.5, max_depth=5,
                                    oob_score=True),
            GradientBoostingClassifier(learning_rate=0.005, n_estimators=250,
                                    max_depth=10, subsample=0.5,
                                    max_features=0.5)]

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    print "Calculating pre-blending values."

    start_time = datetime.now()

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
            dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    end_time = datetime.now()
    time_taken = end_time - start_time

    print "Time taken for pre-blending calculations: ", time_taken

    print "Blending models."

    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test)

    print "Preparing for submission."

    X_index = df_test.values[:,0]
    result = np.c_[X_index.astype(int), y_submission.astype(int)]
    df_result = pd.DataFrame(result, columns=['PassengerId', 'Survived'])

    df_result.to_csv('titanic_blend.csv', index=False)
