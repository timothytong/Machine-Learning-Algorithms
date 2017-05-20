function [ Priors, Mu, Sigma ] = my_gmmEM(X, K, cov_type, plot_iter)
%MY_GMMEM Computes maximum likelihood estimate of the parameters for the 
% given GMM using the EM algorithm and initial parameters
%   input------------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of 
%                           dimension N, each column corresponds to a datapoint.
%       o K         : (1 x 1) number K of GMM components.
%       o cov_type  : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%       o plot_iter : (bool)  set to 1 of want to visalize initual Mu's and
%                          Sigma's, works only for N=2
%       o verb      : (bool)  set to 1 of want to see the convergence output
%   output ----------------------------------------------------------------
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                       Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%
%%%%%% STEP 1: Initialization of Priors, Means and Covariances %%%%%%
[ Priors, Mu, Sigma ] = my_gmmInit(X,K,cov_type,plot_iter);
p_max_prev = -inf;
[N,M] = size(X);
probs = zeros(K,M);
sigma_t1 = zeros(N,N); % Covariance matrix used for the next iteration
max_iter = 500;
iter = 0;

while iter < max_iter

    %%%%%% STEP 2: Expectation Step: Membership probabilities %%%%%%
    
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
        probs(k,:) = prob_point_over_clusters(k,:) ./ den;
    end
    
    %%%%%% STEP 3: Maximization Step: Update Priors, Means and Sigmas %%%%%%    
    sum_probs = sum(probs,2);
    
    % 1) Update Priors
    Priors = 1/M*sum_probs;
    
    % 2) Update Means and Covariance Matrix

    for k=1:K        
        % Update Means
        prob_cluster = probs(k,:);
        
        % interate through each dimension to get the mean.
        for n=1:N
            Mu(n,k) = prob_cluster*transp(X(n,:))/sum_probs(k);
        end

        mu_t1 = Mu(:,k);
        X_demean = bsxfun(@minus, X, mu_t1);
        
        % Update Covariance Matrices 
        if strcmp(cov_type, 'iso') == 1
            iso = 0;
            for i=1:M
                pt_demean = X_demean(:,i);
                iso = iso + prob_cluster(i)*norm(pt_demean)^2;
            end
            sigma_t1 = diag(zeros(1,N)+iso);
            sigma_t1 = sigma_t1/(N*sum_probs(k));
        else
            for i=1:M
                pt_demean = X_demean(:,i);
                sigma_t1 = sigma_t1 + prob_cluster(i)*pt_demean*pt_demean';
            end
            sigma_t1 = sigma_t1/sum_probs(k);
            
            if strcmp(cov_type, 'diag') == 1
                sigma_t1 = diag(diag(sigma_t1));
            end
        end

        % Add a tiny variance to avoid numerical instability
        sigma_t1 = sigma_t1 + diag(zeros(1,N)+10^(-5));
        Sigma(:,:,k) = sigma_t1;
    end    
    
    p_max = my_gmmLogLik(X, Priors, Mu, Sigma);
    %%%%%% Stopping criterion %%%%%%
    diff = abs(p_max_prev-p_max);

    if diff < 10^(-4)
        fprintf('GMM converged after %f\n iters', iter);
        Priors = Priors';
        return;
    end
    p_max_prev = p_max;
    iter = iter + 1;
end

% warning 'max interations exceeded!';

%%%%%% Visualize Final Estimates %%%%%%
if (N==2 && plot_iter==1)
options.labels      = [];
options.class_names = {};
options.plot_figure = false;

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(K);
ml_plot_centroid(Mu',colors);hold on; 
plot_gmm_contour(gca,Priors,Mu,Sigma,colors);
title('Final GMM Parameters');
grid on; box on;

end

