function [ eig_vec, eig_val_diag, Mu ] = my_pca( X )

%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%
%   output ----------------------------------------------------------------
%
%       o U      : (M x M), Eigenvectors of Covariance Matrix.
%       o L      : (M x M), Eigenvalues of Covariance Matrix
%       o Mu     : (N x 1), Mean Vector of Dataset

% Auxiliary variables
[N, M] = size(X);

% Output variables
eig_vec  = zeros(N,N);
eig_val_diag  = zeros(N,N);
Mu = zeros(N);

% Demean data
Mu = mean(X, 2);
X_demeaned = bsxfun(@minus, X, Mu);

% Compute covariant matrix, using M-1 instead of M because true E{X_demeaned}
% is unknown -> Bessel's correction
C = 1/(M-1) * (X_demeaned * X_demeaned');

[eig_vec, eig_val_diag] = eig(C);

% =================== Sort Eigenvectors wrt. EigenValues ==========
% Sort Eigenvalue and get indices
[L_sort, ind] = sort(diag(eig_val_diag),'descend');

% arrange the columns in this order
eig_vec = eig_vec(:,ind); 

% Vectorize sorted eigenvalues
eig_val_diag = diag(L_sort); 

end

