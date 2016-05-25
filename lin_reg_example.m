clear all,clc

load processed_lin_reg.txt
X = processed_lin_reg(:,3:6);
y = processed_lin_reg(:,7);

X = X';
n = size(X,2); % Number of samples
X = [X;ones(1,n)];
d = size(X,1); % Number of features

Budgets = [1:82];
[lambdas,fvals] = d_opt(X,Budgets);
round_lambdas = pipage(lambdas,X);
round_lambds = round(round_lambdas);
 
X = X';
 
beta_all = (eye(d) + X'*X)\(X'*y);


m = size(round_lambdas,2);
betas = zeros(size(X,2),m);
tic
for i=1:m
    lambda = round_lambdas(:,i);
    ind = find(lambda);
    Xtemp = X(ind,:);    
    ytemp = y(ind,:);
    betas(:,i) = (eye(d) + Xtemp'*Xtemp)\(Xtemp'*ytemp);
end
toc

Betas = [betas,beta_all];

