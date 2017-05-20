function [ ll ] = my_gmmLogLik(X, Priors, Mu, Sigma)
%MY_GMMLOGLIK Compute the likelihood of a set of parameters for a GMM
%given a dataset X
%
%   input------------------------------------------------------------------
%
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                    Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%
%   output ----------------------------------------------------------------
%
%      o ll       : (1 x 1) , loglikelihood
%%
[N, M] = size(X);
K = length(Priors);
l_alphas = zeros(K,1);
l_pts = zeros(M,1);

%Compute the likelihood of each datapoint
for i=1:M
    x=X(:,i);
    for k=1:K
        alpha = Priors(k);
        l_alphas(k) = alpha*my_gaussPDF(x,Mu(:,k),Sigma(:,:,k));
    end
    l_pts(i) = log(sum(l_alphas));
end

%Compute the total log likelihood
ll=sum(l_pts);

end

