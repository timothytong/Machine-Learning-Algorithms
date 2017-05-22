function [V, L, Mu] = plot_eigenfaces(X, sizeIm)
%PLOT_EIGENFACES Extracts and displays eigenfaces based on dataset X
%   
%   
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%
%   output ----------------------------------------------------------------
%
%       o A_p      : (p x N), Projection Matrix.
%       o Y      : (p x M), Projected data set with N samples each being of dimension k.


% Extract Principal Directions and Components
[V, L, Mu] = my_pca(X);

% Display the first 20 Eigenfaces
N_displayed_images = 20;
Y = zeros(sizeIm * sizeIm, N_displayed_images);

for i = 1:N_displayed_images

        % Extract Eigenface
        % eigenface = reshape(X(:, i),sizeIm,sizeIm);
        
        [A_p, mean_face] = project_pca(X(:, i), Mu, V, 0);
        
        [X_hat]  = reconstruct_pca(mean_face, A_p, Mu);
        % Plot Eigenface
        
        % eigenface = reshape(X_hat,sizeIm,sizeIm);
        Y(:, i) = X_hat;
end

ml_plot_images(Y',[sizeIm sizeIm]);

end