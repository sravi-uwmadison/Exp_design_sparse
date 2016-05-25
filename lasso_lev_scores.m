function [mu] = lasso_lev_scores( B,lambda)
% clear all,clc
folder_path = pwd;

load('roisummaries.mat')
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
% if ~exist('B','var') error('No B!'); end
% B = 100; % Budget
% Normalize each sample so that the max norm of the dataset is 1
maxNorm = max(sqrt(sum(abs(covs).^2,2)));
covs = covs/maxNorm;

incoherency = 0; % Not used anywhere as of now

initial_sub = 10; % Number of subjects selected randomly before optimization.

rand_init = 0; % To randomly initialize 'initial_sub' subjects set this to 1.

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

iter_max = 2500; % Maximum iterations of the optimization algorithm


% Run alternating optimization algorithm

for iter=1:iter_max
    % First optimize over mu fixing u
%     for opt_mu = 1:1
        temp_mat = eye(d) + covs' * diag(mu) * covs;
        % Compute gradient. TODO: Write a function to parallize this.
        grad_ld_term = zeros(n,1);
        grad_lev_term = zeros(n,1);
        
        % Compute the component f_i that is maximum
        hat_matrix = covs*(temp_mat \ covs');
        lev_scores = diag(hat_matrix);
        lev_function = mu.* lev_scores;
        [lev_max lev_ind] = max(lev_function);
        u = covs(lev_ind,:);
        u = u';
        
        % Compute the gradient wrt mu for a fixed u (in parallel)
        %matlabpool open local 3;
        parfor j=1:n
            [grad_ld_term(j),grad_lev_term(j)] = grad_lev_exp(covs,temp_mat,mu,u,j,lev_ind);
        end
        %matlabpool close;
       
        tot_grad = grad_ld_term + alpha * grad_lev_term; % alpha is the reg parameter
        
        step = 0.4; % Step size for projected gradient. TODO: Use line search.
        % Take a gradient step
        mu = mu - step * tot_grad;
        
        % Project using cplex qp solver. TODO: Make this efficient, definitely
        % possible.
%         tic
        mu = cplexqp(2*eye(n),-2*mu,ones(1,n),B,[],[],zeros(n,1),ones(n,1),mu);
%         toc
        
%     end
end

mu = round(mu);
end