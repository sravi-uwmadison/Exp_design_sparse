function [ lambdas,fval ] = d_opt( X,Budgets )
% This function selects budget number of samples. X is the data matrix with
% rows as features and columns as samples. This assumes that the constant
% coefficient is included in the matrix, that is, the last row is all 1s.

d = size(X,1); % Number of features
n = size(X,2); % Number of samples

% normalize data so that the 2-norm of each sample is atmost 1.

maxNorm = max(sqrt(sum(abs(X).^2,1)));
X = X/maxNorm;
c = ones(n,1); % Just the cardinality constraint for now.

% Maximum iterations for each budget
maxiter = 50;
scaling = 10500; % This constant makes sure that the logdet is not infinity.

lambda = zeros(n,1);
idn = eye(n);
idd = eye(d);
% Store function values for plotting purposes
fval = zeros(maxiter,size(Budgets,2));
% Store all lambdas
lambdas = zeros(n,size(Budgets,2));
b = 1;
tic
for B=Budgets
%     tic
    lambda = zeros(n,1);
    fprintf('Budget = %d\n',B);
    for i=1:maxiter
%         tic
        if (mod(i,10) == 0)
            fprintf('Iteration = %d\n',i);
        end
        % Compute gradient
%         tic
        Xhat = X*diag(lambda);
        temp = idd - Xhat*inv(idn+X'*Xhat)*X';
        grad = -diag(X'*temp*X);
%         toc
        
        % Solve subproblem
        lambdabar = cplexlp(grad,[],[],ones(n,1)',B,zeros(n,1),ones(n,1));
        
        if (grad'*(lambdabar - lambda) == 0)
            break;
        end
        eta = 2/(i+2);
        lambda = lambda + eta *(lambdabar - lambda);
%         tic
%         fval(i,1) = log(det(eye(d)+X*diag(lambda)*X'));
        fval(i,b) = log(det(eye(n) + X'*X*diag(lambda)));
%         toc
    end    
    lambdas(:,b) = lambda;
    b = b+1;
end
toc
% save('lambdas','lambdas','fval')

end

