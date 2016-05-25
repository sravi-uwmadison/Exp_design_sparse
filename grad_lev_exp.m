function [ grad_ld_term_j, grad_lev_term_j ] = grad_lev_exp( covs, temp_mat, mu, u,j,lev_ind )
% Computes the gradient of the objective function wrt mu for a fixed u

grad_ld_term_j = covs(j,:)*(temp_mat\covs(j,:)');

% Some useful quantities to make the computation efficient
temp_A = temp_mat - mu(j) * covs(j,:)' * covs(j,:);
temp_b = temp_A \ covs(j,:)'; % inv(A) * x_i
temp_1 = (u' * temp_b )^2;

% Denominator
temp_d = covs(j,:) * temp_b; % x_i' * inv(A) * x_i
temp_den =  (1 + mu(j) * temp_d)^2;

grad_lev_term_j_1 =  ( ( 1+mu(j) * temp_d ) * temp_1...
    -mu(j) * temp_1 * temp_d )/ temp_den;

grad_lev_term_j_1 = - mu(lev_ind) * grad_lev_term_j_1;

grad_lev_term_j_2 = 0;

if (j == lev_ind)
    grad_lev_term_j_2 = u' * (temp_mat \ u);
end

grad_ld_term_j = -grad_ld_term_j;

grad_lev_term_j = grad_lev_term_j_1 + grad_lev_term_j_2;

end

