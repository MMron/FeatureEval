function (pls.object, opt.comp, p = dim(pls.object$coef)[1]) {
  # Variable importance in prediction
  W <- pls.object$loading.weights
  Q <- pls.object$Yloadings
  TT <- pls.object$scores
  Q2 <- as.numeric(Q) * as.numeric(Q)
  Q2TT <- Q2[1:opt.comp] * diag(crossprod(TT))[1:opt.comp]
  WW <- W * W/apply(W, 2, function(x) sum(x * x))
  VIP <- sqrt(p * apply(sweep(WW[, 1:opt.comp, drop=FALSE],2,Q2TT,"*"), 1, sum)/sum(Q2TT))
  VIP
}

function(pls.object, opt.comp, X, alpha_mc = 0.05){
  # Significance Multivariate Correlation
  # [smcF smcFcrit SSCregression SSResidual] = smc(b, X)
  # Output:
  # smcF : SMC F-values for the list of variables
  # smcFcrit: F-critical cutoff threshold value for significant important variables (smcF>smcFcrit)
  #
  # In case of publication of any application of this method,
  # please, cite the original work:
  # T.N. Tran*, N.L. Afanador, L.M.C. Buydens, L. Blanchet, 
  # Interpretation of variable importance in Partial Least Squares with Significance Multivariate Correlation (sMC), 
  # Chemometrics and Intelligent Laboratory Systems, Volume 138, 15 November 2014, Pages 153-160
  # DOI: http://dx.doi.org/10.1016/j.chemolab.2014.08.005
  
  b  <- pls.object$coefficients[,1,opt.comp]
  X   <- unclass(as.matrix(X))
  
  n <- dim(X)[1]
  
  yhat <- X%*%b
  Xhat <- tcrossprod(yhat,b)/crossprod(b)[1]
  Xresidual <- X - Xhat
  
  SSCregression <- colSums(Xhat^2)
  SSResidual    <- colSums(Xresidual^2)
  
  MSCregression <- SSCregression # 1 degrees of freedom
  MSResidual    <- SSResidual/(n-2)
  
  smcF     <- MSCregression/MSResidual;
  smcFcrit <- qf(1-alpha_mc,1,n-2)
  #  list(smcF=smcF, smcFcrit=smcFcrit)
  attr(smcF, "quantile") <- smcFcrit
  smcF
}


library(pls)
X <- as.matrix(test_data1)
y <- unlist(test_data_target1)
pls <- plsr(y ~ X, ncomp=2, validation="LOO", method="oscorespls")
vip <- VIP(pls, 2)
vip

Xc <- scale(X,center=TRUE,scale=FALSE) 
Xc <- as.matrix(Xc)
pls <- plsr(y ~ Xc, ncomp=2, validation="LOO", method="oscorespls")
smc <- sMC(pls,2,Xc)
smc

summary(pls)
pls$coefficients
dim(pls$coefficients)  
pls$coefficients[, 1, 1:6]
