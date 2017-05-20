function [C] =  confusion_matrix(y_test, y_est)
%CONFUSION_MATRIX Implementation of confusion matrix 
%   for classification results.
%   input -----------------------------------------------------------------
%
%       o y_test    : (1 x M), a vector with true labels y \in {0,1} 
%                        corresponding to X_test.
%       o y_est     : (1 x M), a vector with estimated labels y \in {0,1} 
%                        corresponding to X_test.
%
%   output ----------------------------------------------------------------
%       o C          : (2 x 2), 2x2 matrix of |TP & FN|
%                                             |FP & TN|.
%        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = size(y_est, 2);

tp = 0; fn = 0; fp = 0; tn = 0;
for i = 1:M
    label = y_est(i);
    true_label = y_test(i);
    if ~label && ~true_label
        tp = tp + 1;
    elseif label && ~true_label
        fn = fn + 1;
    elseif ~label && true_label
        fp = fp + 1;
    else
        tn = tn + 1;
    end
end

C = [tp fn; fp tn];
