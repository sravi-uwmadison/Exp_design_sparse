clear all,clc 
folder_path = pwd;
load centered_covs % Load the centered data in the folder, contains just X and y
load y_2
load y_1
cd ./outs; % Go to the folder containing (only) the output files

files = dir; % Get the names of the files in the folder
% files(1) = []; files(1) = []; files(1) = []; files(end) = []; 
nfiles = double(size(files));
for i=1:1:nfiles
   if (files(i).bytes >0) % Pick the files that are useful
     fname = files(i).name;
     load(fname); % Load the file
     mu = round(mu); % Making sure that the entries are either 1 or 0
     X = X(find(mu),:);
     Y = y1(find(mu),:);
     [beta_sel_lasso1 fitinfo_sel1] = lasso(X,Y);
     clear Y; Y = y2(find(mu),:);
     [beta_sel_lasso2 fitinfo_sel2] = lasso(X,Y);     
     %cd ./correct_beta/; % Move to the folder to save
     if strcmp(files(i).name(13:14),'10')
         save(sprintf('correct_beta/beta_%s_%s.mat',files(i).name(9:11),files(i).name(13:14)),...
             'B','mu','beta_sel_lasso1','fitinfo_sel1','beta_sel_lasso2','fitinfo_sel2'); % Save betas
     else
         save(sprintf('correct_beta/beta_%s_%s.mat',files(i).name(9:11),files(i).name(13)),...
             'B','mu','beta_sel_lasso1','fitinfo_sel1','beta_sel_lasso2','fitinfo_sel2'); % Save betas
     end
     cd(folder_path); % Go to the pwd
     load centered_covs % Load the centered data in the folder, contains just X and y
     load y_2
     load y_1
     cd ./outs/;     % Outs to load mu, that is, the selected subjects
   end
end
cd(folder_path);
 