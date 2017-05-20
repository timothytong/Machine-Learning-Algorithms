function [ p ] = explained_variance( L, Var )
%EXPLAINED_VARIANCE Function that returns the optimal p given a desired
%   explained variance. The student should convert the Eigenvalue matrix 
%   to a vector and visualize the values as a 2D plot.
%   input -----------------------------------------------------------------
%   
%       o L      : (N x N), Diagonal Matrix composed of lambda_i 
%  
%   output ----------------------------------------------------------------
%
%       o p      : optimal principal components wrt. explained variance


% ====================== Implement Eq. 8 Here ====================== 


% ====================== Implement Eq. 9 Here ====================== 
 

% ====================== Implement Eq. 10 Here ====================== 


% Visualize Explained Variance from Eigenvalues
figure;
plot(cum_expl_var, '--r', 'LineWidth', 2) ; hold on;
plot(p,cum_expl_var(p),'or')
title('Explained Variance from EigenValues')
ylabel('% Cumulative Variance Explained')
xlabel('Eigenvector index')
grid on

end

