from sklearn.decomposition import PCA
import pandas as pd

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
    '''
    # PCA
    pc_list = ["PC" + str(i) for i in range(1, n + 1)]
    pca = PCA(n, random_state=random_state)
    x_pca = pca.fit_transform(features_data)
    df_pca = pd.DataFrame(x_pca, columns=pc_list)

    # plotting explained variance
    df_explained_variance = pd.DataFrame({
        "pc": pc_list,
        "explained_variance": pca.explained_variance_ratio_
    })

    print("Number of PCs:", n)
    print("Total explained variance:", sum(pca.explained_variance_ratio_))

    # option to print results
    if print_result:
        # checking results
        print("-" * 30)
        print(df_pca.head())
        barplot = sns.barplot(
            x="pc",
            y="explained_variance",
            data=df_explained_variance
        )
        plt.show()
        plt.clf()

    print("PCA completed")
    return pca, df_pca