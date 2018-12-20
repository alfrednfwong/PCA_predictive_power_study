from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd


def compare_classification_features(
        x_data_list, y_data, num_folds=3,
        random_state=None, verbose=False, title=None
):
    '''

    :param features_data_list:
    :param target_data:
    :param predictors:
    :param num_folds:
    :param random_state:
    :param verbose:
    :param title:
    :return:
    '''

    # doing probability is expensive with SVC, but we need that for
    # "soft voting" << such a misleading term.
    # since we'll have few features, we don't need much regularization,
    # so we are leaving the C pretty large here.
    svc = SVC(kernel='rbf', probability=True, random_state=random_state,
              C=1000)
    # we'll use standardized data, intercept is not necessary
    #     lsvc = SVC(kernel='linear', probability=True,
    #                random_state=random_state, C=1000)
    rfc = RandomForestClassifier(n_estimators=100,
                                 random_state=random_state)
    lr = LogisticRegression(C=1000, random_state=random_state,
                            solver='liblinear', max_iter=500)
    knn = KNeighborsClassifier(n_neighbors=20, weights='distance')

    eclf = VotingClassifier(
        estimators=[('svc', svc), ('rfc', rfc), ('lr', lr),
                    ('knn', knn)],
        voting='soft'
    )
    kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    # loop thru sets of features
    mean_scores_list = []
    for x_data in x_data_list:
        # loop thru each CV split
        score_list = []
        for train_index, test_index in kf.split(x_data):
            x_train = x_data.iloc[train_index]
            y_train = y_data.iloc[train_index]
            x_test = x_data.iloc[test_index]
            y_test = y_data.iloc[test_index]

            eclf.fit(x_train, y_train)
            train_pred = eclf.predict(x_train)
            test_pred = eclf.predict(x_test)
            train_score = roc_auc_score(y_train, train_pred)
            test_score = roc_auc_score(y_test, test_pred)

            score_list.append({'train_score': train_score,
                               'test_score': test_score})
        score_df = pd.DataFrame(score_list,
                                columns=['train_score', 'test_score'])
        score_df.index.name = 'fold'
        if verbose:
            print(f'auc score SD: \n{score_df.std()}')
            print(f'auc score: \n{score_df.mean()}')
            print(f'Detailed results: \n{score_df}')
        mean_scores_list.append(score_df.mean()[1])
    return mean_scores_list


