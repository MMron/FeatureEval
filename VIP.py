# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:03:25 2018

@author: Meron
"""
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import sklearn.datasets
from sklearn.exceptions import NotFittedError
class VIP:
    """
    Variable Importance in Projection  - VIP
    Variable Importance in Projection (VIP) scores estimate the importance of 
    each variable in the projection used in a PLS model and is often used for 
    variable selection.
    
    
    Parameters
    ---------- 
    :type opt: int
    :param opt: optimal number of components of PLS model.
    
    :type p: int
    :param p: number of variables in PLS model.

    :type threshold : float  or tuple    
    :param threshold : quantile significance for automatic selection of variables

             
    Attributes
    ---------- 
    importances: array [number_of_features,1 ]
            The quantified value of how important a variable is in the same
            sequence as the variables used in the pls model. 


    threshold  : float or tuple
            quantile significance for automatic selection of variables 
            float with the desired quantile significance for automatic 
            selection of variables or tuple with the interval of the quantile
            significance desired for the automatic selection of variables
            
    opt : int
        optimal number of components in the PLS model.


    Development Notes
    -----
    v0.010 switch from WW/np.ones((self.p,1))*sum(W*W) to numpy divide
    v0.011 new anaconda env sklearn.pls changed to sklearn.cross_decomposition
    v0.011 set_params updated to sklearn standards
    should deleted self.params and only type out params at demand
    v0.012 updated importances calculations to be matrix operations between Q2TT and WW
    
    Notes
    -----
    It is generally accepted that a variable should be selected if vj>1, 
    but proper threshold between 0.83 and 1.21 can yield more relevant 
    variables according 
    
    The nature of the VIP calculation is such that when the model is rebuilt, 
    new variables will always be below the threshold, so this approach does not 
    lend itself to repeated variable exclusion.
    
    
    Examples
    --------
    from sklearn.cross_decomposition import PLSRegression
    import sklearn.datasets
    data = sklearn.datasets.load_boston()
    X = data['data']
    y = data['target']
    pls = PLSRegression()
    pls.fit(X,y)
    vip = VIP()
    vip.fit(pls)
    vip.importances # to type out the diffrent importances of the features
    X_new = vip.transform(X, threshold = (0.83,1.21))
    
    
    References
    ----------
    A review of variable selection methods in Partial Least Squares Regression.
    Tahir Mehmood , Kristian Hovde Liland, Lars Snipen, Solve Sæbø
    Biostatistics, Department of Chemistry, Biotechnology and Food Sciences, 
    Norwegian University of Life Sciences, Norway
    

    
    https://rdrr.io/github/khliland/plsVarSel/src/R/filters.R
    """
    
    def __init__(self):
        """
        Initialize self.  See help(type(self)) for accurate signature.

        """
        self.threshold, self.importances, self.opt = None, None, None
        
    

    def fit(self, pls, opt=None, p=None):
        """
        Computes the importances to the data given

        Get a quantified importance value for each parameter in the matrix X 
        a set of column vectors equal in length to the number of variables 
        included in the model. It contains one column of VIP scores for each 
        predicted y-block column.
        
        Parameters
        ---------- 
        pls : object
            object from PLS. Supported frame works are PLS from Sklearn and 
            PLS from the hoggorm module. 
            
        opt : int
            optimal number of components in the PLS model.
            
        p : int
            number of variables in PLS model.

        
        Returns
        -------
        self
        """ 
        self.pls = pls
        
        self.opt = opt
        
        ## Check wich module is being used
        if hasattr(self.pls,'y_loadings_'): #Sklearn
            q = self.pls.y_loadings_
            t = self.pls.x_scores_
            W = self.pls.x_weights_
            p = len(pls.coef_) if p is None else p

        elif hasattr(self.pls, 'arrQ_alt'): #hoggorm PLS2 ALT isinstance
            q = self.pls.arrQ_alt
            t = self.pls.arrT
            W = self.pls.arrW
            p = np.shape(W)[0] if p is None else p
        elif hasattr(self.pls,'arrQ'): # Hoggorm PLS1
            q = self.pls.arrQ
            t = self.pls.arrT
            W = self.pls.arrW
            p = np.shape(W)[0] if p is None else p

        else: # not support 
            raise NotImplementedError('This pls object type is not supported '\
                                      'The supported modules are sklearn and '
                                      'hoggorm.')

        WW = np.divide(W*W ,np.ones((p,1))*sum(W*W)) # evt np.sum(W*W,axis=0)
        Q2TT = (np.dot(np.dot((q*q)[0:self.opt],t[:,0:self.opt].T),t[:,0:self.opt]))
        #Q2TT = (q*q)[0:self.opt]@t[:,0:self.opt].T @ t[:,0:self.opt]
        self.importances = np.sqrt(p*np.sum(np.ones((p,1))*Q2TT*WW[:,:self.opt],axis=1)/np.sum(Q2TT))

        self.params = None
        return self
    
    
    def transform(self,X,threshold =None):
        """
        Perform feature reduction by selecting features within a user defined 
        VIP threshold.
        
        Parameters
        ----------
        X : pandas dataframe or numpy ndarray
            data matrix used as predictors in PLS modeling.

        threshold : float or tuple 
            quantile significance for automatic selection of variables 
            float with the desired quantile significance for automatic 
            selection of variables or tuple with the interval of the quantile
            significance desired for the automatic selection of variables        
        
        Notes
        -----
        It is generally accepted that a variable should be selected if vj>1, 
        but proper threshold between 0.83 and 1.21 can yield more relevant 
        variables according to 
        
        
        Returns
        -------
        
        :returns value : numpy ndarray or Dataframe, same as the input
            a nxz matrix, where n are the number of samples in x 
            and z are the number of features, z is based upon the 
            threshold value. If the threshold value is None, then
            the features in the returning matrix will have decending
            order of VIP score
        """
        if self.importances is not None: 
            self.threshold = None if threshold is None else threshold
            if isinstance(X,pd.DataFrame): # dataframe
                if threshold is None :
                    return X[X.columns[np.argsort(self.importances)[::-1]]]
                elif isinstance(threshold,(float,int)):
                    return X[X.columns[self.importances>=threshold]]
                elif isinstance(threshold,tuple):
                    max_q = max(threshold)
                    min_q = min(threshold)
                    return X[X.columns[(min_q<self.importances) & (self.importances>max_q)]]
                else:
                    raise TypeError('threshold must be None, number or tuple')
                    
                    
            elif isinstance(X,np.ndarray): # numpy array            
                if threshold is None :
                    return X[:,np.argsort(self.importances)[::-1]]
                elif isinstance(threshold,(float,int)):
                    return X[:,self.importances>=threshold]
                elif isinstance(threshold,tuple):
                     max_q = max(threshold)
                     min_q = min(threshold)
                     return X[:,(min_q<self.importances) & (self.importances<max_q)]
                else:
                    raise TypeError('threshold must be None, number or tuple')            
            else: 
                raise TypeError('X must be a pandas dataframe or numpy ndarray')
        else: 
            raise NotFittedError('importances is not defined, use the fit method to define it')
        
    def fit_transform(self,pls, X, opt=None, threshold=None, p=None): # mer arbeid med å beame pd og np.array
        """        
        Computes importances for the PLS model and returns a transformed version of X.
        
        Parameters
        ----------
        pls : object 
            object from PLS regression.

        X : pandas dataframe or numpy ndarray
            data matrix used as predictors in PLS modelling.
        
        opt : int 
            optimal number of components of PLS model.
        
        p : int
            number of variables in PLS model.
        
        threshold : float or tuple
            quantile significance for automatic selection of variables
        
        Development note
        ----------------
        
        Notes
        -----
        It is generally accepted that a variable should be selected if vj>1, 
        but proper threshold between 0.83 and 1.21 can yield more relevant 
        variables according to "A review of variable selection methods in 
        Partial Least Squares Regression"
        
        
        Returns
        -------
        :returns value : numpy ndarray or Dataframe, same as the input
            a nxz matrix, where n are the number of samples in x 
            and z are the number of features, z is based upon the 
            threshold value. If the threshold value is None, then
            the features in the returning matrix will have descending
            order of VIP score
        
        """
        
        if isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray):
            self.fit(self.pls,opt=opt,p=p) # have it here since X must have the right type
            return self.transform(X,threshold=threshold)
        else: # what else? 
            raise TypeError('X must be a pandas dataframe or numpy ndarray')
        

    
    def get_params(self, deep = True): # deep=True kontra false, litt usikker der.
        """
        Get parameters for this estimator.
    
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
            
        self._params = {'p':self.p,'importances':self.importances,'threshold':self.threshold, 'opt': self.opt}

        return self._params
    
    
    
    def set_params(self,**parameters):
        """
        Set the parameters of this estimator, will keep old parameters if input
        is None.
        
        The method works on simple estimators as well as on nested objects
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
    
    def test_same_as_matlab():
        """
        test that the vip score is equal to those provided from matlab
        matlab number of components: 2
        """
        from scipy.io import loadmat
    
        data = sklearn.datasets.load_boston()
        X = data['data']
        y = data['target']
        pls = PLSRegression(n_components=2)
        pls.fit(X,y)
        
        pls.y_loadings_ = np.array([[101.729463326136	,34.0591751857485]])
        pls.x_scores_ = loadmat('./validering/xs.mat')['Xs']
        pls.x_weights_ = loadmat('./validering/W.mat')['W']
        scores_mat = loadmat('./validering/vipscores1.mat')['values']
        vip = VIP()
        vip.fit(pls)
        corrects = np.sum(np.round(scores_mat.reshape((13,)),10)==np.round(vip.importances,10))
        assert corrects==np.shape(X)[1]
    
    
    def test_mean_scores_1():
        """
        The mean of the squared VIP scores, by construction, is equal to one.  
        - Source: https://brage.bibsys.no/xmlui/bitstream/handle/11250/2423381/Variable+selection+in+multi-block+regression.pdf?sequence=4
        This test therefore test if the mean of the squared VIP scores is equal to one.
        """
        data = sklearn.datasets.load_boston()
        X = data['data']
        y = data['target']
        pls = PLSRegression()
        pls.fit(X,y)
        
        vip = VIP()
        vip.fit(pls)
        assert round(np.mean(vip.importances**2),10) ==1
    
    def test_equal_vip():
        """
        For a dataset with equal columns, check that the vip score is equal
        
        """
        data = sklearn.datasets.load_boston()
        X = np.column_stack((data['data'][:,1],data['data'][:,0],data['data'][:,0],data['data'][:,0]))
        y = data['target']
        pls = PLSRegression()
        pls.fit(X,y)
        
        vip = VIP()
        vip.fit(pls)
    
        assert (len(set(vip.importances))==2) # 3 vil være like 
    
    def test_correct_dim_out():
        """
        Checks that the output dimensjons are reduced as it should. 
        """
        data = sklearn.datasets.load_boston()
        X = data['data']
        y = data['target']
        pls = PLSRegression()
        pls.fit(X,y)
        vip = VIP()
        
        vip.fit(pls)
        vip.importances= np.array([0,0,0,0,0.89,0.9,0,1,3,99,1,12,1.1]) 
        
        assert (np.shape(vip.transform(X,threshold=(0.8,1.5)))==(506,5)) 
    
    def test_random_columns_low():
        """
        Check that the imporance of randomly generated columns are low over some number of iterations
        """
        from sklearn.model_selection import GridSearchCV
        np.random.seed(99)
        pls = PLSRegression()
        no_iter=10
        vip = VIP()
        no_params = 18
        variable_imp = {key: [] for key in range(no_params)}
        
        for i in range(no_iter): # Iterate so that the importance of the random variables are not due to luck/coincidence
            data = np.random.normal(size=(506,no_params))
            noise_y = np.random.normal(0,1,506)
            y = 2*data[:,0]-3*data[:,1]**2 +noise_y
            
            pls = PLSRegression()
            params = {'n_components':list(range(1,no_params))}
            
            gs=GridSearchCV(estimator = pls,
                            param_grid=params,
                            scoring='neg_mean_absolute_error',
                            cv=5)
            gs.fit(data,y)
            vip.fit(gs.best_estimator_)
            for key in variable_imp.keys():
                variable_imp[key].append(vip.importances[key])
        
        
        variable_imp_means = [np.mean(variable_imp[i]) for i in range(no_params)]
        variable_imp_min = np.min(variable_imp_means[:2]) #min importance of orginals features
        r_variable_imp_max = np.max(variable_imp_means[2:]) # max importance of random features
        assert variable_imp_min > r_variable_imp_max
    
