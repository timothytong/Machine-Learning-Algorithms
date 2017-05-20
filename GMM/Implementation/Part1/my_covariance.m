function [ Sigma ] = my_covariance( X, X_bar, type )
%MY_COVARIANCE computes the covariance matrix of X given a covariance type
% and the mean of the dataset X (X_bar).
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o X_bar : (N x 1), an Nx1 matrix corresponding to mean of data X
%       o type  : string , type={'full', 'diag', 'iso'} of Covariance matrix
%
% Outputs ----------------------------------------------------------------
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix of the 
%                          Gaussian function
%%
[N, M] = size(X);
X_demean = bsxfun(@minus, X, X_bar);

if (M == 1)
    Sigma = eye(N);
    return;
else
    Sig_full = 1/(M-1)*X_demean*X_demean' + diag(zeros(1,N)+10^(-5));
end

switch type
    case 'full'
        Sigma = Sig_full;
    case 'diag'
        Sigma = diag(diag(Sig_full));
    case 'iso'
        n = 0;
        for i = 1:M
            n = n + norm(X_demean(:,i))^2;
        end
        iso = zeros(1,N) + 1/(N*M)*n;
        Sigma = diag(iso);
    otherwise
        error 'Wrong type';
end
end

