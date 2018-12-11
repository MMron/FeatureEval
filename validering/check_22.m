%% Kjor
%
% rng(99)
% 
% X = test_data1;         % must 
% X = X(2:end,:);    %a=A(2:end,:); %
% y = test_data_target1;
% y = y(2:end);
% 
% 
% 
% k = 10;                 %number of iterations 
% [n,p] = size(X);
% rand_feat = 5;          % number of random features
% importances = zeros(rand_feat+p,k);
% for i = 1:k
%     r1 = rand(n,rand_feat);
%     new_X= [X,r1];
%     
%     % centering the matrix of predictors 
%     %[n,~] = size(new_X);     % Length of the vector
%     mX = ones(n,1)'*new_X/n; % Finding the mean value of the vector/matrices
%     Xc = (new_X-mX);         % Subtracting the vector by its own mean
%     
%     
%     %[_ _ _ _ beta _ _ _] = plsregress(new_X,y,2);
%     [XL,YL,XS,YS,BETA,PCTVAR,MSE] = plsregress(Xc,y,2);
% 
%     % [smcF smcFcrit SSCregression SSResidual] = smc(b, X)
% 
%     [values smcFcrit l l1] = smc(BETA(2:end),Xc);
%     for j=1:(rand_feat+p)
%         importances(j,i) = values(j); 
%     end
% end 
%% sMC 
% the sMC method assumes that the data is centered 
X = test_data1;         % must 
X = X(2:end,:);    %a=A(2:end,:); %
y = test_data_target1;
y = y(2:end);
% three next lines copied from MATH 280 CA02 assignment spring 2018
[n,~] = size(X);     % Length of the vector
mX = ones(n,1)'*X/n; % Finding the mean value of the vector/matrices
Xc = (X-mX);         % Subtracting the vector by its own mean
[XL,YL,XS,YS,BETA,PCTVAR,MSE] = plsregress(Xc,y,2); 
[values smcFcrit l l1] = smc(BETA(2:end),Xc);
%% VIP 
X = test_data1;         % must 
X = X(2:end,:);    %a=A(2:end,:); %
y = test_data_target1;
y = y(2:end);
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X,y,2);  %[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X,Y,ncomp,...)
vip = VIP(stats.W,YL,XS, 2, 13) %VIP(W,Q,T, kOpt, p)



