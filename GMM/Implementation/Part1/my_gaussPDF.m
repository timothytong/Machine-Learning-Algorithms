function prob = my_gaussPDF(X, Mu, Sigma)
%MY_GAUSSPDF computes the Probability Density Function (PDF) of a
% multivariate Gaussian represented by a mean and covariance matrix.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o Mu    : (N x 1), an Nx1 vector corresponding to the mean of the 
%							Gaussian function
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix 
%						   of the Gaussian function
% Outputs ----------------------------------------------------------------
%       o prob  : (1 x M),  a 1xM vector representing the probabilities for each 
%                           M datapoints given Mu and Sigma    
%%

% Compute Probabilities
[N, M] = size(X);
prob = zeros(1,M);
dists = bsxfun(@minus,X,Mu);
for i = 1:M
    dist = dists(:,i);
    prob(i) = 1/((2*pi)^(N/2)*det(Sigma)^(1/2))*exp(-1/2*dist'*inv(Sigma)*dist);
end
