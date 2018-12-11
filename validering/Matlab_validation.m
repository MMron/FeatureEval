%% sMC 
% the sMC method assumes that the data is centered 
load('test_data1.csv')
load('test_data_target1.csv')
X = test_data1;         % must 
X = X(2:end,:);    %a=A(2:end,:); %
y = test_data_target1;
y = y(2:end);

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



