from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

PATH = './raw/'


def get_data():
    '''
    Gets 10 datasets, do very basic cleaning, and return them in two
    lists of dataframes, the first being the features and the second
    the target variables
    :return:
    list of dataframes - features
    list of dataframes - targets
    '''

    # Musk dataset
    df = pd.read_csv(PATH + 'musk_ver2/clean2.data', header=None)
    column_names = ['mol_name', 'conf_name']
    column_names.extend(list(range(1, 163)))
    column_names.extend(['oxy_dis', 'oxy_x', 'oxy_y', 'oxy_z', 'class_'])
    df.columns = column_names
    y_data_musk = df.class_.astype('int64')
    x_data_musk = df.drop(['class_', 'mol_name', 'conf_name'], axis=1)

    # colposcopy dataset
    df_green = pd.read_csv(PATH + 'colposcopy/green.csv')
    df_hinselmann = pd.read_csv(PATH + 'colposcopy/hinselmann.csv')
    df_schiller = pd.read_csv(PATH + 'colposcopy/schiller.csv')
    df = df_green.append([df_hinselmann, df_schiller])
    df = df.reset_index()
    del df['index']
    y_data_colposcopy = df.consensus
    # columns 62 to 68, starting with "experts", are also target labels.
    # the column 'consensus' is made from these columns
    x_data_colposcopy = df.iloc[:, :62]

    # Z-Alizadeh Sani CAD diagnosis dataset
    df = pd.read_excel(PATH + 'CAD_diagnosis/CAD_diagnosis.xlsx')
    y_data_cad = df.Cath.apply(lambda x: 1 if x == 'Cad' else 0)
    x_data_cad = pd.get_dummies(df.drop('Cath', axis=1), drop_first=True,
                                dtype='int64')

    # Spambase dataset
    df = pd.read_csv(PATH + 'spambase/spambase.data', header=None)
    df.head()
    y_data_spam = df[57]
    x_data_spam = df.drop(57, axis=1)

    # sports articles for objectivity analysis dataset
    df = pd.read_csv(PATH + 'sports_articles_objectivity/features.csv')
    df = df.drop(['TextID', 'URL'], axis=1)
    y_data_sports = df.Label.apply(lambda x: 1 if x == 'subjective' else 0)
    x_data_sports = df.drop('Label', axis=1)

    # sonar detection. mines vs rocks dataset
    df = pd.read_csv(PATH + 'sonar_mines_rocks/sonar.all-data', header=None)
    df.head()
    y_data_sonar = df[60].apply(lambda x: 1 if x == 'R' else 0)
    x_data_sonar = df.iloc[:, :60]

    # first-order theorem proving dataset
    df = pd.read_csv(PATH + 'first_order_theorem_proving/train.csv', header=None)
    df = df.append(pd.read_csv(
        PATH + 'first_order_theorem_proving/test.csv', header=None
    ))
    df = df.append(pd.read_csv(
        PATH + 'first_order_theorem_proving/validation.csv', header=None
    ))
    y_data_thm = df[56].apply(lambda x: 1 if x == 1 else 0)
    x_data_thm = df.iloc[:, :51]

    # secom dataset
    y_data = pd.read_csv(
        PATH + 'secom/secom_labels.data', delimiter=' ', header=None
    )
    x_data = pd.read_csv(PATH + 'secom/secom.data', delimiter=' ', header=None)
    y_data_scm = y_data[0].apply(lambda x: 1 if x == 1 else 0)
    x_data_scm = x_data.fillna(x_data.mean())

    # Epileptic seizure recognition dataset
    df = pd.read_csv(PATH + 'epileptic_seizure/data.csv')
    y_data_epi = df['y'].apply(lambda x: 1 if x == 1 else 0)
    x_data_epi = df.drop(['y', 'Unnamed: 0'], axis=1)

    # Ozone
    path = './raw/ozone/'
    df = pd.read_csv(path + 'onehr.data', header=None)
    for i in range(1, 73):
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna()
    y_data_ozone = df[73]
    x_data_ozone = df.iloc[:, 1:73]

    y_datas = [
        y_data_musk, y_data_colposcopy, y_data_cad, y_data_spam, y_data_sports,
        y_data_sonar, y_data_thm, y_data_scm, y_data_epi, y_data_ozone
    ]
    x_datas_original = [
        x_data_musk, x_data_colposcopy, x_data_cad, x_data_spam, x_data_sports,
        x_data_sonar, x_data_thm, x_data_scm, x_data_epi, x_data_ozone
    ]

    return x_datas_original, y_datas


def do_PCA(n, features_data, random_state=None, print_result=False):
    '''
    Takes an array of feature_data and performs a PCA.
    Array has to be normalized first.

    Parameters:
    ----------
    n: int
        number of components to keep in the PCA. Unlike the standard PCA
        object, cannont take None or decimals.

    features_data: np.array
        numpy array of feature data. shape has to match the row number
        of target data.

    random_state: int, optional
        random seed for both the RandomForestRegressor and the cross
        validation splits

    print_result: bool, defaults to False
        if True, print the details of the PCA results

    Returns:
    ----------
    PCA object
        object containing the info of the transformation.

    DataFrame object
        data after the PCA transformation, with nice column names

    Float
        Explained variance of the group of PCs
    '''
    # PCA
    pc_list = ["PC" + str(i) for i in range(1, n + 1)]
    pca = PCA(n, random_state=random_state)
    x_pca = pca.fit_transform(features_data)
    df_pca = pd.DataFrame(x_pca, columns=pc_list)
    df_explained_var = pd.DataFrame({
        "pc": pc_list,
        "explained_variance": pca.explained_variance_ratio_
    })

    # option to print results
    if print_result:
        print("-" * 30)
        # plotting explained variance

        print("Number of PCs:", n)
        print("Total explained variance:", sum(pca.explained_variance_ratio_))
        print(df_pca.head())
        barplot = sns.barplot(
            x="pc",
            y="explained_variance",
            data=df_explained_variance
        )
        plt.show()
        plt.clf()
        print("PCA completed")

    return pca, df_pca, df_explained_var


def cross_validate_ensemble(
        x_data, y_data, num_folds=3, random_state=None
):
    '''
    Takes a set of features and a binary target, fit an ensemble model on them
    with KFold CV, and return the training score and cross validation score
    :param x_data: array-like(num_obs, num_features). Features
    :param y_data: array-like(num_obs,). target variable. values [0, 1]
    :param num_folds: integer. Number of folds for the cross validation
    :param random_state: random state
    :return: mean training score of the folds
    :return: mean validation score of the folds
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
                            solver='liblinear', max_iter=500,)
    knn = KNeighborsClassifier(n_neighbors=20, weights='distance')

    eclf = VotingClassifier(
        estimators=[('svc', svc), ('rfc', rfc), ('lr', lr),
                    ('knn', knn)],
        voting='soft'
    )
    # eclf = VotingClassifier(
    #     estimators=[('rfc', rfc), ('lr', lr),
    #                 ('knn', knn)],
    #     voting='soft'
    # )

    kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    # loop thru sets of features

    score_list = []
    train_scores = []
    test_scores = []
    # loop thru the folds in CV
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

        train_scores.append(train_score)
        test_scores.append(test_score)
    return np.mean(train_scores), np.mean(test_scores)
