function [] =  gmm_eval(X, K_range, repeats, cov_type)
%GMM_EVAL Implementation of the GMM Model Fitting with AIC/BIC metrics.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o repeats  : (1 X 1), # times to repeat k-means
%       o K_range  : (1 X K), Range of k-values to evaluate
%       o cov_type : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%
%   output ----------------------------------------------------------------
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


AIC_curve = zeros(1, length(K_range));
BIC_curve = zeros(1, length(K_range));
abic = zeros(repeats, 2);

for i=1:length(K_range)
    k = K_range(i);
    for j=1:repeats
        [ Priors, Mu, Sigma ] = my_gmmEM(X, k, cov_type, 0);
        [ AIC, BIC ] = gmm_metrics(X, Priors, Mu, Sigma, cov_type);
        abic(j,1) = AIC;
        abic(j,2) = BIC;
    end
    abic = mean(abic, 1);
    AIC_curve(i) = abic(1);
    BIC_curve(i) = abic(2);
end

% Plot Metric Curves
figure;
plot(K_range, AIC_curve, '--o', 'LineWidth', 1); hold on;
plot(K_range, BIC_curve, '--o', 'LineWidth', 1); hold on;
xlabel('K')
legend('AIC', 'BIC')
title(sprintf('GMM (%s) Model Fitting Evaluation metrics',cov_type))
grid on


end