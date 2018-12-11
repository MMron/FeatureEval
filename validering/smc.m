function [smcF smcFcrit SSCregression SSResidual] = smc(b, X)
% [smcF smcFcrit SSCregression SSResidual] = smc(b, X)
% Input: 
%   X: (Normalized) Data matrix [n samples * nr. variables]
%   b: PLS regression coefficients (obtained by the PLS model on X)
% Output:
% smcF : SMC F-values for the variables
% smcFcrit: F-critical cutoff threshold value. Significant (important) variables is defined by smcF>smcFcrit
%
% In case of publication of any application of this method,
% please, cite the original work:
%
% T.N. Tran, N.L. Afanador, L.M.C. Buydens, L. Blanchet, 
% Interpretation of variable importance in Partial Least Squares with Significance Multivariate Correlation (sMC), 
% Chemometrics and Intelligent Laboratory Systems, Volume 138, 15 November 2014, Pages 153160
% DOI: http://dx.doi.org/10.1016/j.chemolab.2014.08.005
%
% Corresponding author contact: 
% Thanh N. Tran [1][2], 
% 1. Center for Mathematical Sciences, Merck, Sharp & Dohme, Oss, The Netherlands
% E-mail: thanh.tran@merck.com
% 2. Institute for Molecules and Materials, Analytical Chemistry, Radboud University Nijmegen, The Netherlands
% E-mail: thanh.tran@science.ru.nl
% 
% EXAMPLE: on Octane NIR data, Kalivas, J. H., Two data sets of near infrared spectra. Chemom. Intell. Lab. Syst. 1997, 37 (2), 255-259.
%     load spectra
%     whos NIR octane
%     X = NIR;
%     y = octane;
%     wavelength=900 : 2:1700;
%     mx = mean(X);
%     X = bsxfun(@minus,X,mx);
%     my = mean(y);
%     y = bsxfun(@minus,y,my);
%     [b] = nipals_pls1(X,y,6);       % See NIPALS PLS code attached
%     [smcF smcFcrit] = smc(b, X);
%     figure;plot(smcF); hold on; plot([1 length(smcF)],[smcFcrit smcFcrit],'--r')

    alpha_mc = 0.05; 	% Default alpha level of 5%
    [n ignore]=size(X);	% Get the number of sample/observation

    yhat = X*b;		% Get predicted y (PLS prediction)
    Xhat = (yhat*b')/(norm(b).^2);	% Get predicted X
    Xresidual = X - Xhat;


    SSCregression = sum(Xhat.^2);	% Sum Squared regression variance
    SSResidual =sum(Xresidual.^2);	% Sum Squared residual variance

    MSCregression = SSCregression; % Mean Squared regression variance (with 1 degrees of freedom)
    MSResidual = SSResidual/(n-2);	% Mean Squared residual variance (n -2 degrees of freedom)

    smcF = MSCregression./MSResidual;	% sMC value as F-value
    smcFcrit = finv(1-alpha_mc,1,n-2);	% F-critical value
end
