function [ Priors, Mu, Sigma ] = my_gmmInit(X, K, cov_type, plot_iter)
%MY_GMMINIT Computes initial estimates of the parameters of a GMM 
% to be used for the EM algorithm
%   input------------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of 
%                           dimension N, each column corresponds to a datapoint.
%       o K         : (1 x 1) number K of GMM components.
%       o cov_type  : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%       o plot_iter : (bool)  set to 1 of want to visalize initual Mu's and
%                          Sigma's, works only for N=2
%   output ----------------------------------------------------------------
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                       Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%
%%%%%% STEP 1: Initialization of Priors, Means and Covariances %%%%%%
[N,M]=size(X);

% Initialize Priors as Uniform Probabilities
Priors = zeros(1,K) + 1/K;

% Initialize Means with k-means
[labels,Mu] = my_kmeans(X,K,'random','L2',100,plot_iter);

% Initialize Sigmas with labels and Mu from k-means
Sigma = zeros(N,N,K);
for k=1:K
    pts = X(:,labels==k);
    Sigma(:,:,k) = my_covariance(pts,Mu(:,k),cov_type);
end

%%%%%% Visualize Initial Estimates %%%%%%
if (N==2 && plot_iter==1)
    options.labels      = labels;
    options.class_names = {};
    options.title       = 'Initial Estimates for EM-GMM'; 
    if exist('h0','var') && isvalid(h0), delete(h0);end
    h0 = ml_plot_data(X',options);hold on;
    colors     = hsv(K);
    ml_plot_centroid(Mu',colors);hold on; 
    plot_gmm_contour(gca,Priors,Mu,Sigma,colors);
    grid on; box on;
end

end

