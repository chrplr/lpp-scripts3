import time
import numpy as np
import scipy.stats as stats
from sklearn import linear_model
from sklearn import model_selection


def train_model(X_train, y_train, settings, params):
    # Train a regression model according to chosen method
    if settings.method == 'Ridge':
        model = train_model_ridge(X_train, y_train, settings, params)
    elif settings.method == 'Lasso':
        model = train_model_lasso(X_train, y_train, settings, params)
    elif settings.method == 'Elastic_net':
        model = train_model_elastic_net(X_train, y_train, settings, params)

    return model


def train_model_ridge(X_train, y_train, settings, params):
    # Compute path: collect weight coefficient for all regularization sizes
    coefs = [] # Collect coefs for each regularization size (alpha)
    for a in params.alphas:
        model = linear_model.Ridge(alpha=a, fit_intercept=True)
        model.fit(X_train, y_train)
        coefs.append(model.coef_)

    # Grid search - calculate train/validation error for all regularization sizes
    ridge = linear_model.Ridge()
    tuned_parameters = [{'alpha': params.alphas}]
    model_ridge = model_selection.GridSearchCV(ridge, tuned_parameters, cv=params.CV_fold,
                                               refit=True, return_train_score=True)
    model_ridge.fit(X_train, y_train)

    # Add to struct
    model_ridge.alphas = params.alphas
    model_ridge.coefs = coefs

    return model_ridge


def train_model_lasso(X_train, y_train, settings, params):
    # Compute path
    print("Computing regularization path using the lasso...")
    alphas, coefs_lasso, _ = linear_model.lasso_path(X_train, y_train, eps=params.eps, fit_intercept=True)

    # Grid search - calculate train/validation error for all regularization sizes
    lasso = linear_model.Lasso()
    tuned_parameters = [{'alpha': alphas}]
    model_lasso = model_selection.GridSearchCV(lasso, tuned_parameters, cv=params.CV_fold,
                                               return_train_score=True, refit=True)
    model_lasso.fit(X_train, y_train)

    model_lasso.alphas = alphas
    model_lasso.coefs = np.transpose(coefs_lasso)
    return model_lasso


def train_model_elastic_net(X_train, y_train, settings, params):
    # Compute path
    print("Computing regularization path using the elastic net...")
    alphas, coefs_enet, _ = linear_model.enet_path(X_train, y_train,
                                                   eps=params.eps, l1_ratio=params.l1_ratio, fit_intercept=True)

    # Grid search - calculate train/validation error for all regularization sizes
    enet = linear_model.ElasticNet()
    tuned_parameters = [{'alpha': alphas}]
    model_enet = model_selection.GridSearchCV(enet, tuned_parameters, cv=params.CV_fold,
                                              return_train_score=True, refit=True)
    model_enet.fit(X_train, y_train)

    model_enet.alphas = alphas
    model_enet.coefs = np.transpose(coefs_enet)

    return model_enet


def evaluate_model(model, X_test, y_test, settings, params):
    # ## Evaluate the regression models

    if settings.method == 'Ridge':
        scores = eval_model_ridge(model, X_test, y_test, settings, params)
    elif settings.method == 'Lasso':
        scores = eval_model_lasso(model, X_test, y_test, settings, params)
    elif settings.method == 'Elastic_net':
        scores = eval_model_elastic_net(model, X_test, y_test, settings, params)

    return scores


def eval_model_ridge(model, X_test, y_test, settings, params):
    scores = model.score(X_test, y_test)

    return scores


def eval_model_lasso(model, X_test, y_test, settings, params):
    scores = model.score(X_test, y_test)

    return scores


def eval_model_elastic_net(model_elastic_net, X_test, y_test, settings, param):
    scores_enet = model_elastic_net.score(X_test, y_test)
    # scores.append(model_ridge.score(X_test, y_test))

    return scores_enet


# t1 = time.time()
# model_lasso = linear_model.LassoCV(cv=5, eps=params.eps).fit(X_train, y_train)
# t_lasso_cv = time.time() - t1
# # Display results
# m_log_alphas = -np.log10(params.alphas)
