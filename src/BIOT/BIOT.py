#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions and classes for the BIOT module

@author: Rebecca Marion
"""

import pandas as pd 
import numpy as np
import torch
from TorchL1 import TorchL1

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import math
from scipy.stats import mannwhitneyu

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
import argparse

###############################################################################

# Functions and classes

def Get_W_Lasso (X, Y, lam, fit_intercept = False, mode='cpu'):
    """
    Estimates multiple Lasso models (one for each column of Y).

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of d features (columns) used to explain the embedding
        
    Y : numpy.ndarray
        Embedding matrix containing m dimensions to be explained
        
    lam : float
        Sparsity hyperparameter
        
    fit_intercept : boolean
        If True, an intercept is estimated 
        (Default value = False)

    Returns
    -------
    numpy.ndarray
        Matrix of model weights (d features x m embedding dimensions)
    numpy.ndarray
        Vector of m model intercepts 
    """
    

    k = Y.shape[1]
    d = X.shape[1]
    W = np.zeros((d, k))
    w0 = np.zeros(k)
    if fit_intercept:
        mode = 'cpu'
    print('.', end='', flush=True)
    # Fit Lasso for each dimension of Y   
    if mode != 'cpu':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        model = TorchL1(alpha=lam, max_iter=5000, tol=1e-6)
        model.coef_ = torch.zeros(k, d).to(device).T
        model.alpha *= Y.shape[0]
        X_ = torch.from_numpy(X).float().to(device)
        Y_ = torch.from_numpy(Y).float().to(device)
        model.fit(X=X_, y=Y_)
        W = model.coef_.detach().cpu().numpy()
    else:
        for dim_index in range(k):    
            model = Lasso(alpha = lam, fit_intercept = fit_intercept, max_iter = 5000)
            model.fit(X = X, y = Y[:, dim_index])
            W[:, dim_index] = model.coef_
            if fit_intercept:
                w0[dim_index] = model.intercept_
    return W, w0

def Global_L1_Norm (W):
    """
    Calculates the sum of L1 norms for each column of W.

    Parameters
    ----------
    W : numpy.ndarray
        Matrix of model weights
        
    Returns
    -------
    float
        Sum of L1 norms for each column of W
    """
    
    k = W.shape[1]
    norm_val = 0
    for dim_index in range(k):
        norm_val += np.linalg.norm(W[:, dim_index], ord = 1)
        
    return norm_val

def BIOT_Crit (X, Y, R, W, w0, lam):
    """
    Calculates the criterion value for the BIOT objective function.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of d features (columns) used to explain the embedding
        
    Y : numpy.ndarray
        Embedding matrix containing m dimensions to be explained
         
    R : numpy.ndarray
        Orthogonal transformation matrix (m x m)
        
    W : numpy.ndarray
        Matrix of model weights  (d features x m embedding dimensions)
        
    w0 : numpy.ndarray
        Vector of m model intercepts
        
    lam : float
        Sparsity hyperparameter
        
    Returns
    -------
    float
        Criterion value for the BIOT objective function
    """
    
    n = X.shape[0]
    diffs = (Y @ R) - (np.tile(w0, (n, 1)) + (X @ W))
    LS = np.linalg.norm(diffs)**2
    L1 = Global_L1_Norm(W)
    
    crit = ((1/(2*n)) * LS) + (lam * L1)
    
    return crit

def BIOT (X, Y, lam, max_iter = 500, eps = 1e-6, rotation = False, R = None, fit_intercept = False, mode='cpu'):
    """
    Fits a BIOT model.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of d features (columns) used to explain the embedding
        
    Y : numpy.ndarray
        Embedding matrix containing m dimensions to be explained
        
    lam : float
        Sparsity hyperparameter for BIOT
        
    max_iter : int
        Maximum number of iterations to run
        (Default value = 500)
    eps : float
        Convergence threshold
        (Default value = 1e-6)
    rotation : boolean
        If true, the transformation matrix is constrained to be a rotation matrix 
        (Default value = False)
    R : numpy.ndarray
        Optional orthogonal transformation matrix (if provided, R will not be optimized)
        (Default value = None)
    fit_intercept : boolean
        If True, an intercept is estimated 
        (Default value = False)
         
    Returns
    -------
    numpy.ndarray
        Orthogonal transformation matrix (m x m) 
    numpy.ndarray
        Matrix of model weights (d features x m embedding dimensions) 
    numpy.ndarray
        Vector of m model intercepts 
    """
    print('lambda: ' + str(lam))
    d = X.shape[1]
    n = X.shape[0]
    lam_norm = lam/np.sqrt(d)
    
    # If R is provided, get Lasso solution only
    if R is not None:
        YR = Y @ R
        W, w0 = Get_W_Lasso(X = X, Y = YR, lam = lam_norm, fit_intercept=fit_intercept, mode=mode)
    # Otherwise, run BIOT iterations
    else:
        # Init W
        W, w0 = Get_W_Lasso(X = X, Y = Y, lam = lam_norm, fit_intercept=fit_intercept, mode=mode)
        
        diff = math.inf
        iter_index = 0
        crit_list = [math.inf]
        
        while iter_index < max_iter and diff > eps:
            
            # UPDATE R
            if mode != 'cpu':
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

                De = torch.from_numpy((1/(2*n)) * Y.T @ (np.tile(w0, (n, 1)) + (X @ W))).to(device)
                u, s, v = torch.linalg.svd(De, driver='gesvd')
            
                u = u.detach().cpu().numpy()
                s = s.detach().cpu().numpy()
                v = v.detach().cpu().numpy()
            else:
                u, s, v = np.linalg.svd((1/(2*n)) * Y.T @ (np.tile(w0, (n, 1)) + (X @ W)))

            # rotation matrix desired (counterclockwise)
            if rotation:
                sv = np.ones(len(s))
                which_smallest_s = np.argmin(s)
                sv[which_smallest_s] = np.sign(np.linalg.det(u @ v))
                R = u @ np.diag(sv) @ v
            # orthogonal transformation matrix desired
            else:
                R = u @ v
                
            # UPDATE W
            YR = Y @ R
            W, w0 = Get_W_Lasso(X = X, Y = YR, lam = lam_norm, fit_intercept = fit_intercept, mode=mode)      
            
            # CHECK CONVERGENCE
            crit_list.append(BIOT_Crit(X = X, Y = Y, R = R, W = W, w0 = w0, lam = lam_norm))
            diff = np.absolute(crit_list[iter_index] - crit_list[iter_index + 1])
    
            iter_index += 1
            
    return R, W, w0

class BIOTRegressor(BaseEstimator, RegressorMixin):
    """ 
    BIOT class, inherits methods from BaseEstimator and RegressorMixin classes
    from sklearn.base
    """
    def __init__(self, lam = 1, R = None, rotation = False, fit_intercept = False, feature_names = None, mode='cpu'):
        """
        Parameters
        ----------
        lam : float
            Sparsity hyperparameter
            (Default value = 1)
        R : numpy.ndarray
            Optional orthogonal transformation matrix (if provided, R will not be optimized)
            (Default value = None)
        rotation : boolean
            If true, the transformation matrix is constrained to be a rotation matrix 
            (Default value = False)
        fit_intercept : boolean
            If True, an intercept is estimated 
            (Default value = False)
        feature_names : pandas.core.indexes.base.Index or numpy.ndarray
            Names of the features potentially used to explain the embedding 
            dimensions
            (Default value = None)
        """
        self.mode = mode
        self.lam = lam
        self.R = R
        self.rotation = rotation
        self.fit_intercept = fit_intercept
        self.feature_names = feature_names
        
    def fit(self, X, Y):
        """
        Fit method for BIOTRegressor class.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame or numpy.ndarray
            Matrix of d features (columns) used to explain the embedding (training set)
            
        Y : pandas.core.frame.DataFrame or numpy.ndarray
            Embedding matrix containing m dimensions to be explained (training set)
            
        Returns
        -------
        object
            Fitted estimator
        """
       
        X = check_array(X)
        Y = check_array(Y)
        
        R, W, w0 = BIOT(X = X, Y = Y, lam = self.lam, rotation = self.rotation, R = self.R, fit_intercept = self.fit_intercept, mode=self. mode)
        self.R_ = R
        self.W_ = pd.DataFrame(W, index = self.feature_names)
        self.w0_ = w0
        
        return self
    
    def predict(self, X):
        """
        Predict method for BIOTRegressor class.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame or numpy.ndarray
            Matrix of d features (columns) used to explain the embedding (prediction set)
            
        Returns
        -------
        numpy.ndarray
            Predicted values for Y (Y = (w0 1^t + X W) R^t)
        """
        check_is_fitted(self)
        X = check_array(X)
        n = X.shape[0]
        W = check_array(self.W_)
        R = check_array(self.R_)
        intercept = np.tile(self.w0_, (n, 1))
        
        return (intercept + (X @ W)) @ R.T
    
class myPipe(Pipeline):
    """ 
    Class that inherits from the Pipeline class, making the BIOT solution R, W 
    and w0 accessible in the .best_estimator_ attribute of the GridSearchCV 
    class.
    """

    def fit(self, X, Y):
        """Calls last elements .R_, .W_ and .w0_ method.
        Based on the sourcecode for decision_function(X).
        Link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pipeline.py
        ----------

        Parameters
        ----------
        X : pandas.core.frame.DataFrame or numpy.ndarray
            Matrix of d features (columns) used to explain the embedding (training set)
            
        Y : pandas.core.frame.DataFrame or numpy.ndarray
            Embedding matrix containing m dimensions to be explained (training set)
            
        Returns
        -------
        object
            Fitted estimator
        """

        super(myPipe, self).fit(X, Y)

        self.R_ = self.steps[-1][-1].R_
        self.W_ = self.steps[-1][-1].W_
        self.w0_ = self.steps[-1][-1].w0_       
        
        return

def CV_BIOT (X_train, Y_train, feature_names, lam_list, fit_intercept = False, num_folds=10, random_state = 1, R = None, rotation = False, scoring = 'neg_mean_squared_error', mode='cpu'):
    """
    Cross-validated BIOT.
    
    Performs K-fold cross-validation to select the best lambda value for BIOT, 
    and estimates a final model using all data in X_train and Y_train and the 
    selected lambda value. Predictions are then calculated for X_test.

    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame or numpy.ndarray
        Matrix of d features (columns) used to explain the embedding (training set)
        
    X_test : pandas.core.frame.DataFrame or numpy.ndarray
        Matrix of d features (columns) used to explain the embedding (test set)
        
    Y_train : pandas.core.frame.DataFrame or numpy.ndarray
        Embedding matrix containing m dimensions to be explained (training set)
        
    lam_list : list
        List of lambda values to test during cross-validation 
        
    fit_intercept : boolean
        If True, an intercept is estimated 
        (Default value = False)
    num_folds : int
        Number of folds for K-fold cross-validation
        (Default value = 10)
    random_state : int
        Seed for reproducible results
        (Default value = 1)
    R : numpy.ndarray
        Optional orthogonal transformation matrix (if provided, R will not be optimized)
        (Default value = None)
    rotation : boolean
        If true, the transformation matrix is constrained to be a rotation matrix 
        (Default value = False)
    scoring : str
        Scoring function to use for the selection of the hyperparameter lambda.
        For all options, see the list of regression scoring functions at
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter .
        (Default value = 'neg_mean_squared_error')

    Returns
    -------
    numpy.ndarray
        Matrix of predicted values for Y, calculated based on the final model 
        (W, w0 and R) and X_test: (w0 1^t + X_test W) R^t
    pandas.core.frame.DataFrame
        Matrix of model weights (d features x m embedding dimensions) for the
        final model, feature names contained in the index (if available)
    numpy.ndarray
        Vector of m model intercepts for the final model
    numpy.ndarray
        Orthogonal transformation matrix (m x m) for the final model
    """      
     
    # define the model pipeline
    pipe = myPipe([
         ('sc', StandardScaler()),
         ('BIOT', BIOTRegressor(R = R, rotation = rotation, fit_intercept = fit_intercept, feature_names = feature_names, mode=mode))
     ])        
    
    space = dict()
    space['regressor__BIOT__lam'] = lam_list
    
    # configure the cross-validation procedure
    cv = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    # define search
    estimator = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler(with_std = False))
    search = GridSearchCV(estimator = estimator, param_grid = space, scoring=scoring, cv=cv, refit=True)
    # execute search
    search.fit(X_train, Y_train)
    # get the best performing model fit on the whole training set
    best_model = search.best_estimator_

    # Extract the array of MSEs for each fold.
    search_array = []
    for idx in range(num_folds):
        label = 'split' + str(idx) + "_test_score"
        currentSplit = search.cv_results_[label]
        search_array.append(currentSplit)
    # Transpose it to get the MSEs for a single lambda value on the same row
    search_array = np.transpose(search_array)
    print(search_array)

    minimum_mse = search_array[search.best_index_]
    pval = 0
    best_idx = 0
    for idx in range(search.best_index_+1, num_folds):
        current_mse = search_array[idx]
        pval = 1
        if sum(minimum_mse - abs(current_mse)) != 0:
            result = mannwhitneyu(minimum_mse, current_mse, alternative='two-sided', method='exact', use_continuity=True)
            pval = result[1]
        print(str(idx) + '\t' + str(pval))
        if pval<0.05:
            break;
        best_idx = idx   
    print('The lambda value that produces the sparsest model not significantly different from the model with the lowest mse is at index: ' + str(best_idx))

    sc = StandardScaler()
    X_norm = sc.fit_transform(X)
    sc = StandardScaler(with_std=False)
    Y_norm = sc.fit_transform(Y)
    R, W, w0 = BIOT (X_norm, Y_norm, lam_list[best_idx], max_iter = 500, eps = 1e-6, rotation = False, R = None, fit_intercept = False,mode='gpu')
    
    return best_idx, R, W, w0


def calc_max_lam (X, Y):
    """
    Calculate the smallest lambda value resulting in an empty Lasso model.
    
    Before calculating this value, X is centered and scaled, and Y is centered.

    Parameters
    ----------
    X : pandas.core.frame.DataFrame or numpy.ndarray
        Matrix of d features (columns) used to explain the embedding
        
    Y : pandas.core.frame.DataFrame or numpy.ndarray
        Embedding matrix containing m dimensions to be explained
        
    Returns
    -------
    float
        Smallest lambda value resulting in an empty Lasso model
    """
    n = X.shape[0]
    sc = StandardScaler()
    X_norm = sc.fit_transform(X)
    sc = StandardScaler(with_std=False)
    Y_norm = sc.fit_transform(Y)
    max_lam = np.max(np.absolute(X_norm.T @ Y_norm))/n
    
    return max_lam

def normalize(X, mean, std):
  return (X - mean) / std

parser = argparse.ArgumentParser()
parser.add_argument('-e','--embeddingFile', help='csv file containing the embeddings to be predicted', default="embeddings2.csv")
parser.add_argument('-p','--predictorFile', help='csv file containing the predictors to use to predict the embeddings', default="features2.csv")
parser.add_argument('-d','--dataPath', help='common path for the embedding and predictor files', default="./datasets")
parser.add_argument('-o','--outputPath', help='path to location in which to store the output files', default="./output")
parser.add_argument('-s','--suffix', help='suffix to identify output files from those in other runs', default="best")
parser.add_argument('-l','--numLambdas', help='number of lamba values to calculate', type=int, default=10)
parser.add_argument('-m','--min', help='minimum lambda value', type=float, default=.0001)
parser.add_argument('-x','--max', help='maximum lambda value', type=float, default=3.5)
parser.add_argument('-f','--numFolds', help='maximum lambda value', type=int, default=10)
parser.add_argument('-a','--random_state', help='random state to use', type=int, default=1)
parser.add_argument('-c','--scoring', help='scoring algorithm to use in CV', default='neg_mean_squared_error')
parser.add_argument('-r','--rotation', help='Force rotation not orthogonal transformation', action="store_true")
parser.add_argument('-t','--fit_intercept', help='Force lasso to fit intercept', action="store_true")
parser.add_argument('-u','--mode', help='Processing mode (cpu or gpu)', default='cpu')

args = parser.parse_args()

embeddingPath = f"{args.dataPath}/{args.embeddingFile}"
predictorPath = f"{args.dataPath}/{args.predictorFile}"
Y = pd.read_csv(embeddingPath,header=None)
X = pd.read_csv(predictorPath,header=0)

lam_list = np.geomspace(args.min, args.max, args.numLambdas)
print('Lambda values:')
print(lam_list)

random_state = 1
R = None
         
if isinstance(X, pd.DataFrame):
    feature_names = X.columns       
else:
    feature_names = np.array(range(X.shape[0]))

best_idx, R, W, w0 = CV_BIOT (X, Y, feature_names, lam_list, fit_intercept = args.fit_intercept, num_folds=args.numFolds, random_state = args.random_state, R = R, rotation = args.rotation, scoring = args.scoring, mode=args.mode)

Weights = pd.DataFrame(W)
Weights.index = feature_names
weightfile = f"{args.outputPath}/weights_{suffix}.csv"
Weights.to_csv(weightfile)

rotationfile = f"{args.outputPath}/rotation_{suffix}.csv"
Rotation = pd.DataFrame(R)
Rotation.to_csv(rotationfile)

interceptfile = f"{args.outputPath}/intercepts_{suffix}.csv"
Intercepts = pd.DataFrame(w0)
Intercepts.to_csv(interceptfile)

