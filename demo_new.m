clear;
clc;
warning off;
% addpath('/data/home/u21120291/iccv2023/data/')
% addpath('/data/home/u21120291/iccv2023/functions/')
% addpath('/data/home/u21120291/iccv2023/measure/')
%resultdir1 = '/data/home/u21120291/iccv2023/';
% c = parcluster('local');
% c.NumWorkers = 48;
% parpool(c, c.NumWorkers);
addpath('measure')

%% dataset
ds = {'Caltech101-20'};

dsPath = '';
resPath = '';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};

for dsi = 1
    % load data & make folder
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
    %Y=y0(:,1);
    k = length(unique(Y));
    %X=X(4);
%     for i=1:length(X)
%         X{i}=X{i}';
%     end
    matpath = strcat(resPath,dataName);
    %txtpath = strcat(resPath,strcat(dataName,'.txt'));
%     if (~exist(matpath,'file'))
%         mkdir(matpath);
%         addpath(genpath(matpath));
%     end
    %dlmwrite(txtpath, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');
    
    %% para setting
%     lambda = [0.0001,0.001,0.01,0.1,1];
%     lambda1 = [0.0001,0.001,0.01,0.1,1];
% %     lambda = [0.0001,0.001,0.01,0.1,1,10,100,1000];
% %     lambda1 = [0.0001,0.001,0.01,0.1,1,10,100,1000];
%      anchor = [k,2*k,3*k,4*k];
   lambda = [1e-3];%% cal20 1e-3 1e-2 k cal7 1e-1 1e-4 4k %nus% 1e-4 1e-2 3*k
    lambda1 = [1e-2];
    anchor = [1*k];
    d = (1)*k ;
    temp=[];
    for ilambda = 1:length(lambda)
        for ilambda1 = 1:length(lambda1)
            for ichor = 1:length(anchor)
                tic;
                %[U,V,A,W,C,iter,obj] = algo_qp(X,Y,lambda(ilambda),d,anchor(ichor)) ;
                [U,V,A,Z,iter,obj,alpha] = algo_7(X',Y,lambda(ilambda),lambda1(ilambda1),d,anchor(ichor)); % X,Y,lambda,d,numanchor
                [res std] = myNMIACCwithmean(U,Y,k); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
                temp=[temp;res,std];
                time  = toc;
                fprintf('lambda:%d \t lambda2:%d \t Anchor:%d \t Res:%12.6f %12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[lambda(ilambda) lambda1(ilambda1) anchor(ichor) res(1) res(2) res(3) res(4) res(7) time]);

            end
        end
    end
    clear X Y 
end
 max(temp,[],1)
%save('cal20_result.mat')

