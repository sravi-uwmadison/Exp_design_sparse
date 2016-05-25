function [ grad_ld_term_j, grad_eig_term_j ] = grad_eig_exp( covs, temp_mat, mu, u,j )
% Computes the gradient of the objective function wrt mu for a fixed u

grad_ld_term_j = covs(j,:)*(temp_mat\covs(j,:)');

% Some useful quantities to make the computation efficient
temp_A = temp_mat - mu(j) * covs(j,:)' * covs(j,:);
temp_b = temp_A \ covs(j,:)'; % inv(A) * x_i
temp_1 = (u' * temp_b )^2;

% Denominator
temp_d = covs(j,:) * temp_b; % x_i' * inv(A) * x_i
temp_den =  (1 + mu(j) * temp_d)^2;

grad_eig_term_j =  ( ( 1+mu(j) * temp_d ) * temp_1...
    -mu(j) * temp_1 * temp_d )/ temp_den;

end

