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

%% dataset
ds = {'toydata_5'};

dsPath = '';
resPath = '';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};

for dsi = 1
    % load data & make folder
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
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
    lambda = [0.0001];
    anchor = [2*k];
%     lambda = 0.0001;
%     anchor = 2*k ;
    d = (1)*k ;
    
    %%
    for ilambda = 1:length(lambda)
        for ichor = 1:length(anchor)
            tic;
            %[A,W,C,G,F,iter,obj,alpha] = algo_OMSC(X',Y,lambda(ilambda),d,anchor(ichor));  [~,idx]=max(F);
            [U,V,A,C,iter,obj] = algo_4(X,Y,lambda(ilambda),d,anchor(ichor)) ;
            %[U,V,A,Z,iter,obj] = algo_4(X',Y,lambda(ilambda),d,anchor(ichor)); % X,Y,lambda,d,numanchor
            %[result]= mycluster_DB(U,X,k)
            res = myNMIACCwithmean(U,Y,k); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
            timer(ilambda,ichor)  = toc;
            fprintf('lambda:%d \t Anchor:%d \t Dimension:%d\t Res:%12.6f %12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[lambda(ilambda) anchor(ichor) d res(1) res(2) res(3) res(4) res(7) timer(ilambda,ichor)]);
            
            %resall{ilambda,ichor} = res;
            %objall{ilambda,ichor} = obj;
            %dlmwrite(txtpath, [anchor(ichor) d(id) res timer(ichor,id)],'-append','delimiter','\t','newline','pc');
            matname = ['_Anch_','_lambda _',num2str(lambda(ilambda)),num2str(anchor(ichor)),'.mat'];

            %save([matpath,'/',matname],'C','A','U','objall','resall');
            
            
            %save([matpath,'/',matname],'P');
            % save all res and obj in one mat
            %%save([resPath,'All_',dataName,'.mat'],'resall','objall','metric');
        end
    end
    clear X Y 
 end
%save('youtube.mat')

