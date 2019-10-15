import numpy as np
from sklearn import linear_model, svm, ensemble
from sklearn.model_selection import GridSearchCV
import pickle
import os

def  generate_baselines():
    '''
    baseline models to test:
    classification of education levels (under/over-educated):
    SVM, logistic regression, random forests
    (wanted Naive Bayes, sklearn does not have Naive Bayes that can handle
    discrete and continuous RVs and I don't want to throw something together to handle both. We could use a bag-of-words
    model with keyword counts for each article,)

    regression of infant mortality rate:
    linear regression, ridge regression, Naive Bayes regression, SVR

    set up models of interest, give reasons for each model
    set up hyperparameter tuning framework, may depend on runtimes, have cross validation

    each model has dictionary with entries of 'name': model_dict pairs
    each model_dict contains an entry 'model': model object
    as well as an entry 'hyperparams' with entries of 'hyperparameter_name': [hyperparameter values] as a list,
    allowing for grid searches
    '''
    classification_baselines = {}

    # classification models

    # for binary classification, multi_class = 'ovr', solver = 'liblinear'
    # change multi_class to 'multinomial' for multi-class classification (i.e. N_classes > 2) change solver to 'lbfgs'
    # may need to lower n_jobs if processor is completely consumed (-1 = all processors allocated), has no effect if
    # solver = 'liblinear'

    classification_baselines['Logistic Regression'] = {}
    classification_baselines['Logistic Regression']['model'] = linear_model.LogisticRegression(penalty='l2', tol=0.0001,
                                            solver='liblinear', max_iter=100, multi_class='ovr', n_jobs=-1)
    classification_baselines['Logistic Regression']['hyperparams'] = {}
    # C: inverse L2 regularization strength, smaller values >> greater regularization, can also add L1 regularization
    classification_baselines['Logistic Regression']['hyperparams']['C'] = [1e-4, 1e-3, 3e-2, 1e0, 3e1, 1e3]

    # need to specify gamma and C. I assumed an RBF kernel would be reasonable. That could also be cross-validated
    classification_baselines['SVM'] = {}
    classification_baselines['SVM']['model'] = svm.SVC(kernel='rbf')
    classification_baselines['SVM']['hyperparams'] = {}
    classification_baselines['SVM']['hyperparams']['gamma'] = [1e-3, 3e-2, 1e0, 3e1, 1e3, 'scale']
    classification_baselines['SVM']['hyperparams']['C'] = [1e-3, 3e-2, 1e0, 3e1, 1e3]

    classification_baselines['Random Forests'] = {}
    classification_baselines['Random Forests']['model'] = ensemble.RandomForestClassifier()
    # n_estimators='warn', criterion='gini', max_depth=None,
    # min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    # max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    # min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
    # random_state=None, verbose=0, warm_start=False, class_weight=None)
    classification_baselines['Random Forests']['hyperparams'] = {}
    classification_baselines['Random Forests']['hyperparams']['n_estimators'] = [100, 300, 500]
    classification_baselines['Random Forests']['hyperparams']['max_depth'] = [5, 10, None]

    cont_baselines = {}
    # linear regression models
    # ridge regression reduces to least-squares linear regression for alpha = 0
    # ridge regression handles multi-collinearity, was used in the two-step transfer learning paper on Wikipedia articles
    cont_baselines['Ridge Regression'] = {}
    cont_baselines['Ridge Regression']['model'] = linear_model.Ridge()
    cont_baselines['Ridge Regression']['hyperparams'] = {}
    cont_baselines['Ridge Regression']['hyperparams']['alpha'] = [0, 0.1, 0.5, 1, 5, 10, 20, 50, 100, 200, 500, 1000]

    # Another linear method: lasso regression
    cont_baselines['Lasso Regression'] = {}
    cont_baselines['Lasso Regression']['model'] = linear_model.Lasso()
    cont_baselines['Lasso Regression']['hyperparams'] = {}
    cont_baselines['Lasso Regression']['hyperparams']['alpha'] = [0, 0.1, 0.5, 1, 5, 10, 20, 50, 100, 200, 500, 1000]

    # random forests regression
    # uses MSE as criterion, should be converted to R^2 with simple post-cross-validation regression over data
    cont_baselines['Random Forests Regressor'] = {}
    cont_baselines['Random Forests Regressor']['model'] = ensemble.RandomForestRegressor()
    cont_baselines['Random Forests Regressor']['hyperparams'] = {}
    cont_baselines['Random Forests Regressor']['hyperparams']['n_estimators'] = [100, 300, 500]
    cont_baselines['Random Forests Regressor']['hyperparams']['max_depth'] = [5, 10, None]

    return classification_baselines, cont_baselines

def save_baselines(task_name, baselines):
    # check if dumps directory exists
    dump_dir = os.path.abspath('./dumps')
    if not os.path.exists(dump_dir):
        os.mkdirs(dump_dir)
    dump_name = '{}_baselines.pickle'.format(task_name)
    with open(os.path.join(dump_dir, dump_name), 'wb+') as f:
        pickle.dump(baselines, f)


def run_baselines(task_name, baselines, label_pair, n_folds=5):
    # # simple data set for testing the code
    # X = [(0, 0.1), (1, 1), (2, 2)] * 3
    # y = [0, 2.1, 3] * 3
    # n_folds = 2
    X_train, X_test, y_train, y_test = label_pair

    for name, baseline in baselines.items():
        # perform cross-validation, may need to reduce number of folds for compute
        # reduce n_jobs to number of processors to use if CPU overwhelmed (-1 means use all processors) and
        # reduce pre_dispatch to 'n_jobs' if memory errors occur
        clf = GridSearchCV(baseline['model'], baseline['hyperparams'], cv=n_folds, n_jobs=-1, pre_dispatch='2*n_jobs', refit=True)
        clf.fit(X_train, y_train)

        baselines[name]['best_model'] = clf.best_estimator_
        baselines[name]['best_hyperparams'] = clf.best_params_
        print('{}: Best performance of {} on {}-fold CV was achieved with hyperparameters:\n{}'
              .format(name, clf.best_score_, n_folds, clf.best_params_))

    save_baselines(task_name, baselines)

def generate_ed_label_df(cluster_df):
    ed_data = cluster_df[['cluster_id', 'pct_no_education', 'pct_primary_education',
                            'pct_secondary_education','pct_higher_education']]
    ed_data['is_undereducated'] = ((ed_data.pct_no_education + ed_data.pct_primary_education) >
                                    (ed_data.pct_secondary_education + ed_data.pct_higher_education))
    return ed_data[['cluster_id', 'is_undereducated']]

def generate_label_pairs(label_df, wiki_df, label_col, feat_cols,
        train_countries, test_countries, test_size=0.2):
    """Generates numpy label pairs from the provided dataframes
    Parameters
    ----------
    label_df : type
        Description of parameter `label_df`.
    wiki_df : type
        Description of parameter `wiki_df`.
    label_col : string
        column applying to y label
    feat_cols : list
        list of column headers which contain relevant features
    train_countries : list
        list of countries to train on
    test_countries : list
        list of countries to test on
    """
    from sklearn.model_selection import train_test_split

    merged = wiki_df.merge(label_df, on='cluster_id')
    all_countries = set(train_countries + test_countries)
    # don't bother splitting on irrelevant countries
    relevant_countries = merged[merged.country.isin(all_countries)]
    train_df, test_df = train_test_split(relevant_countries, test_size=test_size)
    # filter out into our train and test countries
    train_df = train_df[train_df.country.isin(train_countries)]
    test_df = test_df[test_df.country.isin(test_countries)]

    X_train = train_df[feat_cols].to_numpy()
    X_test = test_df[feat_cols].to_numpy()
    y_train = train_df[label_col].to_numpy()
    y_test = test_df[label_col].to_numpy()
    return X_train, X_test, y_train, y_test

def generate_data():
    '''
    TODO: FINISH
    '''
    data_dir = os.path.abspath('../data')
    processed_dir = os.path.join(data_dir, 'processed')
    cluster_level_combined_file = 'ClusterLevelCombined_5yrIMR_MatEd.csv'

    '''
    Pseudocode
    iterate through X,y pairings, e.g.
        school wiki info, education data
        hospital wiki info, health data
    '''

    cluster_level_combined_df = pd.read_csv(os.path.join(processed_dir, cluster_level_combined_file))
    ed_labels_df = generate_ed_label_df(cluster_level_combined_df)
    health_labels_df = cluster_level_combined_df[['cluster_id', 'imr']]

    # wiki_df = load_wiki_df
    # health_feat_cols = list(col_headers)
    # educ_feat_cols = list(col_headers)

    label_pairs = {}
    label_pairs['ed'] = generate_label_pairs(ed_labels_df, wiki_df, 'is_undereducated', ed_feat_cols)
    label_pairs['health'] = generate_label_pairs(health_labels_df, wiki_df, 'imr', health_feat_cols)

    return label_pairs

def main():
    classification_baselines, cont_baselines = generate_baselines()

    label_pairs = generate_data()

    results = {}
    results['health'] = run_baselines('health', cont_baselines, label_pairs['health'])
    results['ed'] = run_baselines('ed', cont_baselines, label_pairs['ed'])




if __name__ == '__main__':
    main()
