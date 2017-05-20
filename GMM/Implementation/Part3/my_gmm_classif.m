function [y_est] = my_gmm_classif(X_test, models, labels, K, P_class)
%MY_GMM_CLASSIF Classifies datapoints of X_test using ML Discriminant Rule
%   input------------------------------------------------------------------
%
%       o X_test    : (N x M_test), a data set with M_test samples each being of
%                           dimension N, each column corresponds to a datapoint.
%       o models    : (1 x N_classes) struct array with fields:
%                   | o Priors : (1 x K), the set of priors (or mixing weights) for each
%                   |            k-th Gaussian component
%                   | o Mu     : (N x K), an NxK matrix corresponding to the centroids
%                   |            mu = {mu^1,...mu^K}
%                   | o Sigma  : (N x N x K), an NxNxK matrix corresponding to the
%                   |            Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o labels    : (1 x N_classes) unique labels of X_test.
%       o K         : (1 x 1) number K of GMM components.
%   optional---------------------------------------------------------------
%       o P_class   : (1 x N_classes), the vector of prior probabilities
%                      for each class i, p(y=i). If provided, equal class
%                      distribution assumption is no longer made.
%
%   output ----------------------------------------------------------------
%       o y_est  :  (1 x M_test), a vector with estimated labels y \in {0,...,N_classes}
%                   corresponding to X_test.
%%
M = size(X_test, 2);
N_classes = length(models);
prob_classes = zeros(1, N_classes); % probability calculated for each class, take argmin after
l_alphas = zeros(K,1);
y_est = zeros(1,M);

if ~exist('P_class', 'var')
    P_class = zeros(1, N_classes) + 1;
end

for i=1:M
    point = X_test(:,i);
    
    for n=1:N_classes
        % extract model structs
        model = models(n);
        Priors = model.Priors;
        Mu = model.Mu;
        Sigma = model.Sigma;
        
        % Compute the likelihood of each datapoint
        for k=1:K
            alpha = Priors(k);
            l_alphas(k) = alpha*my_gaussPDF(point,Mu(:,k),Sigma(:,:,k));
        end
        prob_classes(n) = -log(sum(l_alphas)*P_class(n));
        
    end
    
    [~, idx] = min(prob_classes);
    y_est(i) = labels(idx);
end

end