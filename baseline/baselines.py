import numpy as np
from sklearn import linear_model, svm, ensemble
from sklearn.model_selection import GridSearchCV
import pickle

# baseline models to test:
# classification of education levels (under/over-educated):
# SVM, logistic regression, random forests
# (wanted Naive Bayes, sklearn does not have Naive Bayes that can handle
# discrete and continuous RVs and I don't want to throw something together to handle both. We could use a bag-of-words
# model with keyword counts for each article,)

# regression of infant mortality rate:
# linear regression, ridge regression, Naive Bayes regression, SVR

# set up models of interest, give reasons for each model
# set up hyperparameter tuning framework, may depend on runtimes, have cross validation

# each model has dictionary with entries of 'name': model_dict pairs
# each model_dict contains an entry 'model': model object
# as well as an entry 'hyperparams' with entries of 'hyperparameter_name': [hyperparameter values] as a list,
# allowing for grid searches
classification_baselines = {}

# classification models
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


# insert data here
# be sure to make train/test split, only perform CV on train set
X = []
y = []
n_folds = 5

# simple data set for testing the code
X = [(0, 0.1), (1, 1), (2, 2)] * 3
y = [0, 2.1, 3] * 3
n_folds = 2

# choose between regression and classification
# baselines = classification_baselines
baselines = cont_baselines

for name, baseline in baselines.items():
    # perform cross-validation, may need to reduce number of folds for compute
    # reduce n_jobs to number of processors to use if CPU overwhelmed (-1 means use all processors) and
    # reduce pre_dispatch to 'n_jobs' if memory errors occur
    clf = GridSearchCV(baseline['model'], baseline['hyperparams'], cv=n_folds, n_jobs=-1, pre_dispatch='2*n_jobs', refit=True)
    clf.fit(X, y)

    baselines[name]['model'] = clf.best_estimator_
    baselines[name]['hyperparams'] = clf.best_params_
    print('{}: Best performance of {} on {}-fold CV was achieved with hyperparameters:\n{}'
          .format(name, clf.best_score_, n_folds, clf.best_params_))

# can save models and hyperparameters:
# with open('./baseline/baselines.pickle', 'wb+') as f
#     pickle.dump(baselines, f)
