function [UU,V,A,C,iter,obj,alpha] = algo_7(X,Y,lambda,lambda2,d,numanchor)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di
%% 对U进行正交约束,且有正则项
%% initialize
maxIter = 50 ; % the number of iterations
lambda=0;
%lambda2=0;
m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);
A = cell(1,numview);
U = cell(1,numview);
B = cell(1,numview);

%W = cell(numview,1);            % di * d
%A = zeros(d,m);         % d  * m
C = zeros(m,numsample); % m  * n
for i = 1:numview
   X{i} = mapstd(X{i}',0,1); % turn into d*n
   di = size(X{i},1); 
   %W{i} = zeros(di,d);
   A{i} = randi(di,m);
   U{i} = randi(m,m)/m;
   B{i} = randi(m,m);
   Lb{i} = randi(m,m);
end
%C(:,1:m) = eye(m);


alpha = ones(1,numview)/numview;
opt.disp = 0;

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    %% optimize A
    sumAlpha = 0;
    part1 = 0;
    for ia = 1:numview
        al2 = alpha(ia)^2;
        sumAlpha = sumAlpha + al2;
        part1 = al2 * (X{ia}) * C'* U{ia}';
    
    [Unew,~,Vnew] = svd(part1,'econ');
%     A = (part1/sumAlpha) * inv(Z*Z');
    A{ia} = Unew*Vnew';
    end
    
  %% optimize U
for iu = 1:numview
    Ub = A{iu}'*(X{iu})*C';
    [W,~,V] = svd(Ub,'econ');
    U{iu} = W*V';
 end 
    %% optimize C
    temp_c = 0;temp_c11=0;temp_c22=0;
    for ia=1:numview
        temp_c1{ia} = (alpha(ia)^2)*eye(m,m)+lambda2*Lb{ia}*eye(m,m)-lambda*eye(m,m);
        temp_c11=temp_c11+temp_c1{ia};
        temp_c2{ia} = U{ia}'*A{ia}'*X{ia};
        temp_c22 = temp_c22+temp_c2{ia};
    end
    C=temp_c+inv(temp_c11)*temp_c22;

    for ii = 1:size(C,2)
        C(:,ii) = EProjSimplex(C(:,ii));
    end  
        %% optimize Lb
        for iv = 1:numview
            B{iv} = constructW_PKN(A{iv}, 3);
            Db = diag(sum(B{iv},1)+eps);
            Lb{iv} = eye(m,m)-Db^-0.5*B{iv}*Db^-0.5;
        end
    %% optimize alpha
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm( X{iv} - A{iv} * U{iv} * C,'fro')^2;
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;

    %%
    term1 = 0; term2 =0; term3=0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - A{iv} * U{iv} * C,'fro')^2;
        term3 = term3 + lambda2 * trace_large(C,Lb{iv});
    end
    term2 = lambda * norm(C,'fro')^2;
    obj(iter) = term1 - term2 + term3;
    %obj(iter) = term1;
    
    %if  iter>=maxIter 
    if (iter>29) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,V]=svd(C','econ');
        flag = 0;
    end
end
end
  function [traceValue]=trace_large(C,Lb)
        lbSparse = sparse(Lb);
        [m, n] = size(C);
        chunkSize = 1000; % 分块大小
        traceValue = 0;
        for j = 1:chunkSize:n
            endIdx = min(j + chunkSize - 1, n);
            C_chunk = C(:, j:endIdx);
            traceValue = traceValue + trace(C_chunk' * lbSparse * C_chunk);
        end
    end        
         
    
