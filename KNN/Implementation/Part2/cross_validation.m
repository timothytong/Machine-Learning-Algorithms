function [TP_rate_F_fold, FP_rate_F_fold, std_TP_rate_F_fold, std_FP_rate_F_fold] =  cross_validation(X, y, F_fold, tt_ratio, k_range)
%CROSS_VALIDATION Implementation of F-fold cross-validation for kNN algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y         : (1 x M), a vector with labels y \in {0,1} corresponding to X.
%       o F_fold    : (int), the number of folds of cross-validation to compute.
%       o tt_ratio  : (double), Training/Testing Ratio.
%       o k_range   : (1 x K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------
%
%       o TP_rate_F_fold  : (1 x K), True Positive Rate computed for each value of k averaged over the number of folds.
%       o FP_rate_F_fold  : (1 x K), False Positive Rate computed for each value of k averaged over the number of folds.
%       o std_TP_rate_F_fold  : (1 x K), Standard Deviation of True Positive Rate computed for each value of k.
%       o std_FP_rate_F_fold  : (1 x K), Standard Deviation of False Positive Rate computed for each value of k.
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_k = size(k_range, 2);
TP_rate_F_fold = zeros(1, n_k);
FP_rate_F_fold = zeros(1, n_k);
std_TP_rate_F_fold = zeros(1, n_k);
std_FP_rate_F_fold = zeros(1, n_k);

TP_rate_ = zeros(F_fold, 1);
FP_rate_ = zeros(F_fold, 1);

for i = 1:n_k
    k = k_range(i);
    for j = 1:F_fold
        [ X_train, y_train, X_test, y_test ] = split_data(X, y, tt_ratio);
        y_est = my_knn(X_train, y_train, X_test, k, 'L2');
        C = confusion_matrix(y_test, y_est);
        TP_rate_(j) = C(1, 1) / sum(C(1, :));
        FP_rate_(j) = C(2, 1) / sum(C(2, :));
    end
    
    TP_rate_F_fold(i) = mean(TP_rate_);
    FP_rate_F_fold(i) = mean(FP_rate_);
    std_TP_rate_F_fold(i) = std(TP_rate_);
    std_FP_rate_F_fold(i) = std(FP_rate_);
end
end