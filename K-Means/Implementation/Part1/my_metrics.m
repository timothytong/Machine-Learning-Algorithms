function [RSS, AIC, BIC] =  my_metrics(X, labels, Mu)
%MY_METRICS Computes the metrics for clustering evaluation
%
%   input -----------------------------------------------------------------
%   
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Mu    : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N
%
%   output ----------------------------------------------------------------
%
%       o d      : distance between x_1 and x_2 depending on distance
%                  type {'L1','L2','LInf'}
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
[D, M] = size(X);
[~, K] = size(Mu);


% Compute RSS (Equation 8)
RSS = 0;
for i = 1 : K
    for j = 1 : M
        RSS = RSS + (my_distance(X(:, j), Mu(:, i), 'L2'))^2 * (i == labels(j));
    end
end   
  
% Compute AIC (Equation 9)
B = K * D;
AIC = RSS + 2 * B;

% Compute BIC (Equation 10)
BIC = RSS + B * reallog(M);


end