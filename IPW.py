# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:42:57 2018

@author: Meron
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import sklearn.datasets
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
class IPW:
    """
    An iterative procedure for variable elimination  - IPW
    This is an iterative elimination procedure where a measure of predictor 
    importance is computed after fitting a model. The importance measure is 
    used both to re-scale the original X-variables and to eliminate the least 
    important variables before subsequent model re-fitting
    
    Parameters   
    ----------
    threshold : float, optional
        Features with a measure of relevancy lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    
    Attributes
    ---------- 
    importances: array [number_of_features,1 ]
                The quantified value of how important a variable is in the same
                sequence as the column in the matrix of predictor input.         


    Development note
    -----
    Fit method can contain filter type, where the measure of importance can be 
    specified. type of measures could be VIP, sMC and so on.

    
    Examples
    --------
    import sklearn.datasets
    from sklearn.linear_model import Ridge
    data = sklearn.datasets.load_boston()
    X = data['data']
    y = data['target']
    r = Ridge()
    ipw = IPW()
    ipw.fit_transform(r,X,y,threshold=0.01)
    
    References
    ----------
    M. Forina, C. Casolino, C. Pizarro Millan, Iterative predictor weighting
    (IPW) PLS: a technique for the elimination of useless predictors in regression problems,
    Journal of Chemometrics 13 (1999) 165-184.
    
    https://github.com/khliland/plsVarSel/blob/master/R/IPW.R    
    """
    
    def __init__(self):
        """
        Initialize self.  See help(type(self)) for accurate signature.

        """
        self.threshold, self.params,  self.importances, self.no_iter = None, None, None, None
        
    
    def fit(self,model,X, y, no_iter=10, threshold=0.01,scale=True):
        """
        Computes the variable importance of the data given

        Get a quantified importance value for each parameter in the matrix X 
        a set of column vectors equal in length to the number of variables 
        included in the model. It contains one column of importance measure for
        each predicted y-block column trough an iterative procedure for 
        variable elimination (IPW).
        
        
        Parameters   
        ----------
        model : object
            a type of model which has the fit method and does have coefficients 
            associated with the different variables in the predictor matrix.
            Nota bene: for desired convergence the model input need to have an
            internal form of regularization, so that the importance converges.
            An ordinary linear regression is not sufficient, excepted models are
            Ridge, Lasso, Elastic net, PLS etc.  
        
        X : ndarray or pandas dataframe
            Predictor matrix which contains the information to predict the
            response value. Shape n,p
            
        y : ndarray or pandas dataframe 
            Response values for classification problem the classes must be  
            onehotencoded
            
        no_iter : int
            number of iterations in the iterative procedure for variable 
            elimination. Default value: 10
            
        threshold : float, optional
            threshold for the measure of importance associated with the different 
            variables. A measure of importance under the given threshold will 
            remove the variable. Default value: 0.01
        
        scale : boleen
            whether or not to scale the variables, if True the variables will be
            scaled. Default value: True
        
        Returns
        -------
        self
        """
        ### let input be pandas dataframe 
        if isinstance(X,pd.DataFrame):
            X = X.get_values()
        if isinstance(y,pd.DataFrame):
            y = y.get_values()
        
        self.threshold = 0.01 if threshold is None else threshold
        self.no_iter = no_iter
        s = np.std(X,axis=0)
        if scale:
            X = (X - np.mean(X,axis=0))/s
        
        if len(set(y))!=2 and np.ndim(y) ==1: #not binary classification or multiclassification problem
            y = (y -np.mean(y))/np.std(y)
        
        n, self._p = np.shape(X)
        
        z = np.ones(self._p) 
        self._storage = {}
        Xorig = X
        for i in range(self.no_iter):
            X = Xorig*z
                
            model.fit(X,y)
            
            # Filter calculation can be other functions as sMC, VIP and so on..
            weights = model.coef_.flatten()
        
            weights[np.isinf(weights,where=True)] = 0 # correct for non-finite weigths
            z = abs(weights)*s
            
            z /= sum(z)
            self._storage[str(i)]=z
            if self.threshold:
                z[z<self.threshold] = 0
                
            if np.sum(z)==0:
                print('The combination of parameters removed all variables. '
                      'Iteration ended at itartation number {0}.'.format(i))
                break
            
        self.importances = z
        self.significant_variables = z[z>0]
        return self 

    def plot_development(self,size=(12,12),columns = None):
        """
        Function to see the convergence of each parameter during the iterations
        in the fitting process.
        
        Development note
        ----------------
        When more than 30 variables are applied one cannot differentiate
        between one or more of the graphs
        
        Parameters   
        ----------
        size : tuple
            size of the plot as a tuple. Default value: (12,12) 
        
        columns: list
            list containing the column name of the different parameters in the
            same sequence as when fitted in the fit method  
        
        Returns
        -------
            Plot

        """
        if self.importances is None:
            raise NotFittedError('importances is not defined, use the fit method to define it') 
        else:
 
            params_evolve ={str(j): [i[j] for i in self._storage.values()] for j in range(self._p)}             
            
            #### plotting####
            plt.figure(figsize=size)
            d = ['-','--','-.']
            for i in range(self._p):
                q = d[min(2,i//10)]
                if columns is None: 
                    plt.plot(list(range(self.no_iter)), params_evolve[str(i)],label='param '+ str(i+1), linestyle = q)                    
                else:
                    plt.plot(list(range(self.no_iter)), params_evolve[str(i)],label = columns[i], linestyle = q)
                
            plt.legend(loc='upper right')
            plt.xlabel('number of iterations')
            plt.ylabel('variabel importance') 
            plt.show()
            
        
        
    def transform(self,X):
        """
        Perform feature reduction by selecting features within from the IPW.
        
        Parameters
        ----------
        X : pandas dataframe or numpy ndarray
            matrix used as predictors.

        
        Notes
        -----

        
        Returns
        -------
        :return value : numpy ndarray or Dataframe, same as the input
            a nxz matrix, where n is the number of samples in x 
            and z are the number of features, z is based upon the 
            threshold value. The features in the returning matrix will have 
            descending order of measure of relevancy provided by
            the IPW iteration.

        """
        if self.importances is not None: 
            self.params = None
            if isinstance(X,pd.DataFrame): # dataframe
                return X[X.columns[np.argsort(self.significant_variables)[::-1]]]                    
                    
            elif isinstance(X,np.ndarray): # numpy array            
                return X[:,np.argsort(self.significant_variables)[::-1]]

            else: 
                raise TypeError('X must be a pandas dataframe or numpy ndarray')
        else: 
            raise NotFittedError('importances is not defined, use the fit method to define them')
        
    def fit_transform(self, model, X, y, no_iter=10, threshold=None, scale=True): 
        """        
        Fits transformer to X, and returns a transformed version of X.
        
        Parameters
        ----------
        model : object
            a type of model which has the fit method and does have coefficients 
            associated with the different variables in the predictor matrix.
            Nota bene: for desired convergence the model input need to have an
            internal form of regularization, so that the importance converge.
            An ordinary linear regression is not sufficient, excepted models are
            Ridge, Lasso, Elastic net, PLS etc.   
        
        X : ndarray or pandas dataframe
            Predictor matrix which contains the information to predict the
            response value. Shape n,p
            
        y : ndarray or pandas dataframe 
            Response values 
            
        no_iter : int
            number of iterations in the iterative procedure for variable 
            elimination. Default value: 10
            
        threshold : float, optional
            threshold for the measure of importance associated with the different 
            variables. A measure of importance under the given threshold will 
            remove the variable. Default value: 0.01
        
        scale : boleen
            whether or not to scale the variables, if True the variables will be
            scaled. Default value: True
                
        Returns
        -------
        :return value : numpy ndarray or Dataframe, same as the input
            a nxz matrix, where n is the number of samples in x 
            and z are the number of features, z is based upon the 
            threshold value. The features in the returning matrix will have 
            descending order of measure of relevancy provided by
            the IPW iteration.
        
        """
        self.params = None
        if isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray):
            self.fit(model,X,y,no_iter,threshold,scale) 
            return self.transform(X) # threshold has already been applied in fitting
        else: # what else? 
            raise TypeError('X must be a pandas dataframe or numpy ndarray')
        

    
    def get_params(self, deep = True): 
        """
        Get parameters for this class.
    
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this module and
            contained subobjects that are class.
            
        
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """ 
           
            
        params = {'threshold':self.threshold, 'p':self.p,
                           'importances':self.importances}

        return params
    
    
    
    def set_params(self,**parameters):
        """
        Set the parameters of this class, will keep old parameters if input
        is None.
        
        The method works on simple instances as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        
        Aquired from https://scikit-learn.org/stable/developers/contributing.html 

        Returns
        -------
        self
        
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        self.params = None
        return self
if __name__ == "__main__":
    
        
    def test_equal_importance():
        """
        For a dataset with equal columns, check that the vip score is equal
        
        Development note
        ----------------
        When columns are the same, all but one pls scores becomes the same.
        """
        data = sklearn.datasets.load_boston()
        X = np.column_stack((data['data'][:,0],data['data'][:,0],data['data'][:,0]))
        y = data['target']
        r = Ridge()
        ipw = IPW()
        ipw.fit(r,X,y,threshold=0,no_iter=100) # long convergence time
        assert (len(set(np.round(ipw.importances,10)))==1) # 3 will be the same  
        
        
    def test_random_columns_low():
        """
        Check that the imporance of randomly generated features are lower than the  
        features who correlates with the response over some number of iterations
    
        """
            
        from sklearn.model_selection import GridSearchCV
        np.random.seed(99)
        pls = PLSRegression()
        no_iter=10
        ipw = IPW()
        no_params = 18
        variable_imp = {key: [] for key in range(no_params)}
        
        for i in range(no_iter): # Iterate so that the importance of the random variables are not due to luck/coincidence
            data = np.random.normal(size=(506,no_params)) # generate random data
            noise_y = np.random.normal(0,1,506) # add noise to the response
            y = 2*data[:,0]-3*data[:,1] +noise_y
            
            pls = PLSRegression()
            params = {'n_components':list(range(1,no_params))}
            
            gs=GridSearchCV(estimator = pls,
                            param_grid=params,
                            scoring='neg_mean_absolute_error',
                            cv=5)
            gs.fit(data,y)
            ipw.fit(gs.best_estimator_,data,y)
            for key in variable_imp.keys():
                variable_imp[key].append(ipw.importances[key])
                
        variable_imp_means = [np.mean(variable_imp[i]) for i in range(no_params)]
        variable_imp_min = np.min(variable_imp_means[:2]) #min importance of orginals features
        r_variable_imp_max = np.max(variable_imp_means[2:]) # max importance of random features
        assert variable_imp_min > r_variable_imp_max


    data = sklearn.datasets.load_boston()
    X = data['data']
    y = data['target']
    r = Ridge()
    ipw = IPW()
    ipw.fit_transform(r,X,y,threshold=0.01)
