clear all,clc

load lars.txt
B = 50:10:220;
nB = size(B,2);
X = lars(:,1:size(lars,2)-1);
y = lars(:,size(lars,2));

n = size(X,1);
% X = [2 3; 4 5; 5 6];

epsilon = 0.9;
muX = mean(X);
sigmaX_norm = norm(std(X));

maxiter = 10;
filtered = zeros(size(X,1),size(X,2)+1,maxiter); % This stores the filtered samples at each stage.
num_filtered_iter = zeros(maxiter,1); % This stores the number of samples filtered at each stage.

for iter=1:maxiter
   
   % Construct a ball of radius sigmaX around muX and delete points inside
   % the ball   
   X_minus_mu = bsxfun(@minus, X,muX); 
   rownorm = sum(sqrt(sum(X_minus_mu'.^2,1)),1)'; % These are the distance of all the points from the mean
   
   % Now we will pick the samples outside the ball
   rownorm(rownorm < epsilon*sigmaX_norm) = 0;
   rownorm(rownorm >= epsilon*sigmaX_norm) = 1;
   
   num_filtered_iter(iter) = size(find(rownorm == 0),1); % This stores the number of samples filtered at each stage.
   
   % rownorm now consits of subjects that are selected to the next round
   filtered(1:size(rownorm,1)-size(find(rownorm),1),:,iter) = [X(find(rownorm == 0),:), y(find(rownorm == 0),:)]; % Y consists of samples that were rejected at this state
   
   % Update X, muX, sigmaX_norm
   X = X(find(rownorm),:);
   y = y(find(rownorm),:);
   
   if size(X,1) == 1
       Y(1,:,iter+1) = [X, y];
       iter
       break;
   elseif size(X,1) == 0       
       iter
       break;
   else       
       muX = mean(X);
       sigmaX_norm = norm(std(X));
   end
    
end
% sum(num_filtered_iter)

num_to_pick = zeros(size(num_filtered_iter,1) ,nB);

for i=1:nB
    num_to_pick(:,i) = round((num_filtered_iter/n) * B(i));
end

num_to_pick

























