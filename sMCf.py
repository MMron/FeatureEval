# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 18:57:54 2018

@author: Meron
"""

import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.cross_decomposition import PLSRegression
import sklearn.datasets
import scipy.stats
from scipy.io import loadmat
class sMC :
    """
     Significance Multivariate Correlation — sMC
    
    
    Parameters
    ----------
    type pls: object
        object from PLS regression.
    
    :param: int 
        optimal number of components of PLS model.
    
    : alpha_mc: float
        the chosen significance level for f-test.    

    Attributes
    ---------- 
    importances: array [number_of_features,1 ]
            The quantified value of how important a variable is in the same
            sequence as the variables used in the input model.
                
    Notes
    -----
    The algorithm assumes that the
    data is provided centred and scaled (optional). 
    
    
    Examples
    --------
    from sklearn.cross_decomposition import PLSRegression
    import sklearn.datasets

    data = sklearn.datasets.load_boston()
    X = data['data']
    y = data['target']
    pls = PLSRegression()
    pls.fit(X,y)
    
    smc = sMC()
    important_dataset = smc.fit_transform(pls,X,alpha_mc=0.0001)
 
    
    References
    ----------
    T.N. Tran*, N.L. Afanador, L.M.C. Buydens, L. Blanchet, 
    Interpretation of variable importance in Partial Least Squares with Significance Multivariate Correlation (sMC), 
    Chemometrics and Intelligent Laboratory Systems, Volume 138, 15 November 2014, Pages 153-160
    DOI: http://dx.doi.org/10.1016/j.chemolab.2014.08.005
    
    https://rdrr.io/github/khliland/plsVarSel/src/R/filters.R
    """
    
    def __init__(self):
        """
        Initialize self.  See help(type(self)) for accurate signature.

        """
        
        self.model, self.importances,  self.significant_variables =  None, None, None
        self.opt, self.params, self.alpha_mc = None, None, None
    

    def fit(self, model,X, opt=None, alpha_mc=None): #pls.object, opt.comp, X, alpha_mc = 0.05)
        """
        Computes the importance to the features given a dataset and a fitted
        classification/regression model with coefficients for each parameter

        Get a quantified importance value for each parameter in the matrix X 
        a set of column vectors equal in length to the number of variables 
        included in the model. It contains one column of mSC scores for each 
        predicted y-block column. The important variables are those who passes 
        the F-value test, and has a F-value over the critical f-value 
        associated with the chosen significance level
        
        Parameters
        ---------- 
        model: object
            object from a classifier or regression model with atribute coef_.
        
        X : Pandas Dataframe or numpy ndarray
            data matrix values
        
        opt : int
            optimal number of components of PLS model.
        
        alpha_mc : float 
            the chosen significance level for f-test. Range <0,1>
            default value: 0.05

        Attributes
        -------
        importances : numpy array
            SMC F-values for the list of variables
        
        smcFcrit: Float 
            F-critical cut-off threshold value for significant important
                    variables (smcF>smcFcrit)
        
        significant_variables: List
            list with false and true values according to smcF>smcFcrit 
            representing the important columns in the given dataset
        
        
        Development note
        ----------------
        should remove model object as input and replace it with the 
        coefficients for the given model, this way the code will be easier to 
        read. Also the code will generalise better since it could be used on
        any model which has coefficients and not be restricted to the sklearn
        and hoggorm packages.
        
        
        Returns
        -------
        self        
        """ 
        self.alpha_mc = 0.05 if alpha_mc is None else alpha_mc

        
        #The algorithm assumes scaled features the next two lines dummy proofs
        if (np.round(np.sum(X),9)!=0): # not scaled
            X -=np.mean(X,axis=0) 
            
        if hasattr(model,'coef_'): #sklearn
            b = model.coef_
        elif hasattr(model,'cvType'):  #hoggorm
            opt = np.shape(model.X_loadings())[1] if opt is None else opt
            b = model.regressionCoefficients(numComp=opt)
        else: 
            raise NotImplementedError('This model object type is not supported '\
                                      'The supported objects are sklearn and '
                                      'hoggorm pls.')

        n = np.shape(X)[0]
        
        yhat = np.dot(X,b)
        Xhat = np.dot(yhat,b.T)/(np.linalg.norm(b)**2)
        Xresidual = X - Xhat
        
        SSCregression = np.sum(Xhat**2,axis=0)
        SSResidual = np.sum(Xresidual**2,axis=0)
        
        MSCregression = SSCregression
        MSResidual = SSResidual/(n-2)
        
        smcF = np.divide(MSCregression,MSResidual)
        self.smcFcrit = scipy.stats.f.ppf(1-self.alpha_mc,1,n-2)
        
        self.importances = smcF
        self.significant_variables = smcF > self.smcFcrit

        return self
    
    
    def transform(self,X):
        """
        Perform feature reduction by selecting features within a f-value 
        threshold associated with the significance certainty threshold described 
        by aplha_mc.
        
        Parameters
        ----------
        X : ndarray or pandas dataframe, shape [n_samples, n_features]
            The data used to scale along the features axis.
        
        
        Returns
        -------
        :returns value : numpy ndarray or Dataframe, same as the input
            a nxz matrix, where n are the number of samples in x 
            and z are the number of features, z is based upon the 
            F-critical cut-off threshold value.
        """
        if self.significant_variables is not None: 
            if isinstance(X,pd.DataFrame): # dataframe
                return X[X.columns[self.significant_variables]]
                
            elif isinstance(X,np.ndarray): # numpy array            
                return X[:,self.significant_variables]       
            else: 
                raise TypeError('X must be a pandas dataframe or numpy ndarray')
        else: 
            raise NotFittedError('This sMC instance is not fitted yet. Call the fit method '\
                                 'with appropriate arguments before using this method')
        
    def fit_transform(self,model, X,opt=None, alpha_mc=None): 
        """
        Fit to data based upon the data and the desired model.
        Later the data is transformed to only include the important variables.
        
        Fits transformer to X, and returns a transformed version of X.
        
        Parameters
        ----------
        model: object
            object from a classifier or regression model with atribute coef_.
            or a PLS object from hoggorm

        X: pandas dataframe or numpy ndarray
            data matrix used as predictors in model.
        
        opt: int
            optimal number of components of PLS model.
        
        alpha_mc: float
            the chosen significance level for f-test.
        

        Returns
        -------
        :returns value: numpy ndarray or Dataframe, same as the input
            a nxz matrix, where n is the number of samples in x 
            and z are the number of features, z is based upon the 
            threshold value.

        """
        
        if isinstance(X,pd.DataFrame):
            self.fit(model,X.get_values(),opt,alpha_mc) 
            return self.transform(X)
        elif isinstance(X,np.ndarray):
            self.fit(model,X,opt,alpha_mc)
            return self.transform(X)
        else: 
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
           
        self._params = {'importances':self.importances, 
                        'significant_variables': self.significant_variables,
                        'smcFcrit':self.smcFcrit,'alpha_mc':self.alpha_mc}
        return self._params
    
    
    
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
        return self
    



if __name__ == "__main__":
    def test_same_as_matlab():
        """
        test that the sMC score is equal to those provided from matlab
        """
        data = sklearn.datasets.load_boston()
        X = data['data']
        y = data['target']
        pls = PLSRegression()
        pls.fit(X,y)
        smc_mat = loadmat('./validering/values_smc_1_centered.mat')['values']
        coef = loadmat('./validering/beta_1_centered')['BETA']
        pls.coef_ = coef[1:] # leave the interception out
        smc = sMC()
        smc.fit(pls,X)
        corrects = np.sum(np.round(smc.importances,10) == np.round(smc_mat,10))
        assert (corrects==np.shape(X)[1])
    
    
    def test_equal_f_value():
        """
        For a dataset with equal columns, check that the F-value is equal
        """
        data = sklearn.datasets.load_boston()
        X = np.column_stack((data['data'][:,0],data['data'][:,0],data['data'][:,0]))
        y = data['target']
        pls = PLSRegression()
        pls.fit(X,y)
        
        smc = sMC()
        smc.fit(pls,X)
        assert len(set(smc.importances))==1
    
    
    def test_correct_dim_out():
        """
        Checks that the output dimensjons are reduced as it should. 
        """
        data = sklearn.datasets.load_boston()
        X = data['data']
        y = data['target']
        pls = PLSRegression()
        pls.fit(X,y)
        smc = sMC()
        smc.fit(pls,X)
        smc.significant_variables= np.array([False,False,False,True,False,False,True,False,False,False,True,True,True])
        assert (np.shape(smc.transform(X))==(506,5)) 
    
    def test_random_columns_low():
        """
        Check that the imporance of randomly generated columns are low over some number of iterations
        """
        from sklearn.model_selection import GridSearchCV
        np.random.seed(99)
        pls = PLSRegression()
        no_iter=10
        smc = sMC()
        no_params = 18
        sampels = 506
        variable_imp = {key: [] for key in range(no_params)}
        
        for i in range(no_iter): 
            data = np.random.normal(size=(sampels,no_params))
            noise_y = np.random.normal(0,1,sampels)
            y = 2*data[:,0] - 3*data[:,1] + noise_y
            
            pls = PLSRegression()
            params = {'n_components':list(range(1,no_params))}
            
            gs=GridSearchCV(estimator = pls,
                            param_grid=params,
                            scoring='neg_mean_absolute_error',
                            cv=5)
            gs.fit(data,y)
            smc.fit(gs.best_estimator_,data)
            
            for key in variable_imp.keys():
                variable_imp[key].append(smc.importances[key])
        
        variable_imp_means = [np.mean(variable_imp[i]) for i in range(no_params)]
        variable_imp_min = np.min(variable_imp_means[:2]) #min importance of orginals features
        r_variable_imp_max = np.max(variable_imp_means[2:]) # max importance of random features
        assert variable_imp_min > r_variable_imp_max    

    data = sklearn.datasets.load_boston()
    X = data['data']
    y = data['target']
    pls = PLSRegression()
    pls.fit(X,y)
    
    smc = sMC()
    smc.fit_transform(pls,pd.DataFrame(X),alpha_mc=0.01)