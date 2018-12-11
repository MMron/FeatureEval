function vip = VIP(W,Q,T, kOpt, p)
% Input:  cppls.object= Final fitted model after cross-validation, 
%kOpt= optimum number of components to be used, p= Number of varaibles in X
%Output: Variable Importance on projection for each variable in X
%[Xloadings,Yloadings,Xscores,Yscores,betaPLS] = plsregress(X,y,2);
%q = self.pls.y_loadings_
%t = self.pls.x_scores_
%W = self.pls.x_weights_
Q2 =  Q(:).*Q(:);
WW = (W.*W)./(ones(p,1)*sum(W.*W));
 
vip = sqrt(p * sum((ones(p,1)*(Q2(1:kOpt)'.*diag(T(:,1:kOpt)'*T(:,1:kOpt))')).*WW(:,1:kOpt),2) ...
    / sum(Q2(1:kOpt)'.*diag(T(:,1:kOpt)'*T(:,1:kOpt))'));
end