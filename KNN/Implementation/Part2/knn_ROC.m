function [ TP_rate, FP_rate ] = knn_ROC( X_train, y_train, X_test, y_test, k_range )
%KNN_ROC Implementation of ROC curve for kNN algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_test   : (1 x M_test), a vector with labels y \in {0,1} corresponding to X_test.
%       o k_range  : (1 x K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------

%       o TP_rate  : (1 x K), True Positive Rate computed for each value of k.
%       o FP_rate  : (1 x K), False Positive Rate computed for each value of k.
%        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Populate True Positive Rate and True Negative Rate vectors (for each k-value)
n_k = size(k_range, 2);
TP_rate = zeros(0, n_k);
FP_rate = zeros(0, n_k);

for i = 1:n_k
    y_est = my_knn(X_train, y_train, X_test, k_range(i), 'L2');
    C = confusion_matrix(y_test, y_est);
    TP_rate(i) = C(1, 1) / sum(C(1, :));
    FP_rate(i) = C(2, 1) / sum(C(2, :));
end

end