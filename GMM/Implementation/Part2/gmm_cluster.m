function [labels] =  gmm_cluster(X, Priors, Mu, Sigma, type, softThresholds)
%GMM_CLUSTER Computes the cluster labels for the data points given the GMM
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                           Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o type   : string ,{'hard', 'soft'} type of clustering
%
%       o softThresholds: (2 x 1), a vecor for the minimum and maximum of
%                           the threshold for soft clustering in that order
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a M dimensional vector with the label of the
%                             cluster for each datapoint
%                             - For hard clustering, the label is the 
%                             cluster number.
%                             - For soft clustering, the label is 0 for 
%                             data points which do not have high confidnce 
%                             in cluster assignment
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N,M] = size(X);
K = length(Priors);
posteriori_probs = zeros(K,M);
labels = zeros(1,M);

% Find the a posteriori probability for each data point for each cluster,
% code from gmmEM
% 1) Compute probabilities p(x^i|k)
prob_point_over_clusters = zeros(K,M);
for k = 1:K
    prob_point_over_clusters(k,:) = Priors(k) .* my_gaussPDF(X,Mu(:,k),Sigma(:,:,k)); % numerator
end

% denominator is just the sum of the numerator by rows (all clusters),
% this is a row vector.
den = sum(prob_point_over_clusters,1);

% 2) Compute posterior probabilities p(k|x)  %%%
for k = 1:K
    posteriori_probs(k,:) = prob_point_over_clusters(k,:) ./ den;
end

% Use posterior probabilities to assign points to clusters based on
% clustering method 'hard' or 'soft'
for ii = 1:M
    point_probs = posteriori_probs(:,ii);
    [~, label] = max(point_probs);
    labels(ii) = label;
    switch type
        case 'hard'
            % Find the cluster with highest probability - do nothing!
            
    
        case 'soft'
            % Find the cluster with highest probabilty. Unless, the highest
            % and another cluster are in the same range specified by
            % threshold
            if sum(point_probs > min(softThresholds)) == K && sum(point_probs < max(softThresholds)) == K
                labels(ii) = 0;
            end

        otherwise
            fprintf('Invalid type for clustering\n');
            break;
            
    end
end

