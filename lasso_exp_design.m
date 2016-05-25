function lasso_exp_design( B)
% clear all,clc
folder_path = pwd;

load('../roisummaries.mat')
y = Ys;
% Remove rows with Nan
[indnanr indnanc] = find(isnan(covs));
indnanr = unique(indnanr);
covs(indnanr,:) = [];
y(indnanr,:) = [];

% Remove columns with the same value, that is, all samples take the same value
ncols = size(covs,2);
for i=1:ncols
    if(size(unique(covs(:,i)),1) == 1)
        covs(:,i) = [];
    end
    if (i > size(covs,2))
        break
    end
end

% Extract the linearly independent columns from covs
[covs] = licols(covs);

r = rank(covs);
n = size(covs,1);
d = size(covs,2);
if ~exist('B','var') error('No B!'); end
% B = 300; % Budget
% Normalize each sample so that the max norm of the dataset is 1
maxNorm = max(sqrt(sum(abs(covs).^2,2)));
covs = covs/maxNorm;

incoherency = 0; % Not used anywhere as of now

initial_sub = 10; % Number of subjects selected randomly before optimization.

rand_init = 0; % To randomly initialize 'initial_sub' subjects set this to 1.

if ((B < initial_sub ) && (rand_init == 1))
    fprintf('Random initialization is not valid or meaningful, make sure initial_sub < B\n');
    return
end

% Tolerances -- not used anywhere as of now
rip_tol = 1;
inf_gain_tol = 0.001;

% sel indicates the indices of the selected subjects
% sub stores the subjects directly

rcovs = covs;  % Another set just for book keeping purposes

if (rand_init == 1)
    randset = randperm(n);
    rand_start = covs(randset(1:initial_sub),:); % Select random 'initial_sub' subjects
    start_mat = eye(d)+ rand_start'*rand_start; % For random set start initialize
    rcovs(randset(1:initial_sub),:) = []; % Pick a random set of 10 subjects
    sel = randset(1:initial_sub)'; % Store the initial indices of subjects
    sub = covs(randset(1:initial_sub),:); % Store the initial subjects
else
    rand_start = [];
    start_mat = eye(d); % Start with the empty set
    sel = [];
    sub = [];
end

[V D] = svds(covs'*covs+eye(d)); % Compute the largest singular vectors

u = zeros(d,1); % Initialize u
mu = zeros(n,1); % Initialize mu

alpha = 1; % Regularization parameter. TODO: Have to tune ofc.
lambda = 1; % Another regularization parameter. TODO: Have to tune ofc.

% Initialize u to be the largest eigenvector
u = V(:,1);

gamma = V(:,1); % Eigenvector corresponding to largest eigenvalue.

if (size(find(u<=0),1) == d) % If all coordinates of u are nonpositive, then multiply by -1. This can be done whenever the covariates are nonnegative, e.g. roisummaries.
    u = -u;
else
    u = u;
end

if (size(find(gamma<=0),1) == d) % If all coordinates of u are nonpositive, then multiply by -1. This can be done whenever the covariates are nonnegative, e.g. roisummaries.
    gamma = -gamma;
else
    gamma = gamma;
end

iter_max = 75; % Maximum iterations of the optimization algorithm


% Run alternating optimization algorithm
% tic
for iter=1:iter_max
    % First optimize over mu fixing u
    for opt_mu = 1:75
        temp_mat = eye(d) + covs' * diag(mu) * covs;
        % Compute gradient. TODO: Write a function to parallize this.
        grad_ld_term = zeros(n,1);
        grad_eig_term = zeros(n,1);
        
        % Compute the gradient wrt mu for a fixed u (in parallel)
	% matlabpool open local 2;
        for j=1:n
            [grad_ld_term(j),grad_eig_term(j)] = grad_eig_exp(covs,temp_mat,mu,u,j);
        end
        % matlabpool close;

        % Missed a negative sign for both of the above terms so multiply by
        % negative one
        
        grad_ld_term = -grad_ld_term;
        grad_eig_term = -grad_eig_term;
        tot_grad = grad_ld_term + alpha * grad_eig_term; % alpha is the reg parameter
        
        step = 0.4; % Step size for projected gradient. TODO: Use line search.
        % Take a gradient step
        mu = mu - step * tot_grad;
        
        % Project using cplex qp solver. TODO: Make this efficient, definitely
        % possible.
%         tic
        mu = cplexqp(2*eye(n),-2*mu,ones(1,n),B,[],[],zeros(n,1),ones(n,1),mu);
%         toc
        
    end
    
    % Now fix mu and optimize u
    
    % Compute the eigenvalue matrix (always positive definite)
    temp_mat = eye(d) + covs' * diag(mu) * covs;
    M = inv(temp_mat); % Note: costly operation
    
    % Use projected gradients
    step = 0.4; % Step size for projected gradients. TODO : Use line search.
    for opt_u = 1:50
        grad_u = ( lambda * eye(d)  * u ) + ( alpha * M * u ) - 2*gamma; % Either solve a linear system or compute inverse once?
        u = u - step * grad_u;
        u = u/norm(u); % This is the projection step
    end
    
end
% toc

mu = round(mu);
n_sel = size(find(mu));
sel = find(mu);
covs_sel = covs(sel,:);
y_sel = y(sel,:);

% tic
% beta_all = inv(covs'*covs+eye(d))*covs'*y(:,1);
% beta_sel = inv(covs_sel'*covs_sel+eye(d))*covs_sel'*y_sel(:,1);
% 
% [beta_all_lasso, fitinfo_all] = lasso(covs,y(:,1)');
[beta_sel_lasso, fitinfo_sel] = lasso(covs_sel,y_sel(:,1)');
% toc

save(sprintf('%s/outs/outputs_%d.mat',folder_path,B),'B','mu','beta_sel_lasso','fitinfo_sel');


end
% 
% fprintf('The number of nonzero elements for a small regularization lasso in the full dataset is %d\n',nnz(beta_all_lasso(:,1)));
% fprintf('The number of nonzero elements for a small regularization lasso in the selected dataset is %d\n',nnz(beta_sel_lasso(:,1)));
% % nnz(beta_all_lasso(:,2))
% % nnz(beta_sel_lasso(:,2))
% fprintf('The number of nonzero elements for a large regularization lasso in the full dataset is %d\n',nnz(beta_all_lasso(:,50)));
% fprintf('The number of nonzero elements for a large regularization lasso in the selected dataset is %d\n',nnz(beta_sel_lasso(:,50)));
% figure;
% plot(beta_all_lasso(:,1),'r.');hold;plot(beta_sel_lasso(:,1),'b+')
% legend('Blue is selected and Red is full dataset');
% figure;
% plot(beta_all_lasso(:,50),'r.');hold;plot(beta_sel_lasso(:,50),'b+')
% legend('Blue is selected and Red is full dataset');



