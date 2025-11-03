function [UU,V,A,C,iter,obj,alpha] = algo_4(X,Y,lambda,d,numanchor)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di
%% 对U进行正交约束,且有正则项
%% initialize
maxIter = 50 ; % the number of iterations

m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);
A = cell(1,numview);
U = cell(1,numview);
%W = cell(numview,1);            % di * d
%A = zeros(d,m);         % d  * m
C = zeros(m,numsample); % m  * n

for i = 1:numview
   X{i} = mapstd(X{i}',0,1); % turn into d*n
   di = size(X{i},1); 
   %W{i} = zeros(di,d);
   A{i} = zeros(di,m);
   U{i} = ones(m,m)/m;
end
C(:,1:m) = eye(m);


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
        part1 = al2 * X{ia} * C'* U{ia}';
    
    [Unew,~,Vnew] = svd(part1,'econ');
%     A = (part1/sumAlpha) * inv(Z*Z');
    A{ia} = Unew*Vnew';
    end
    % Replace inv(A)*b with A\b
    % Replace b*inv(A) with b/A
  %% optimize U
for iu = 1:numview
    Ub = A{iu}'*X{iu}*C';
    [W,~,V] = svd(Ub,'econ');
    U{iu} = W*V';
 end 
    %% optimize C
options = optimset( 'Algorithm','interior-point-convex','Display','off'); % Algorithm 默认为 interior-point-convex

for ji=1:numsample
    ff=0;H=0;
    for j=1:numview
        HH =(alpha(ia)^2)* U{j}'*U{j};
        H = H + 2 * HH - 2*lambda*eye(m);
        H = (H+H')/2;
        CC =  A{j} * U{j};
        ff = ff - 2*sumAlpha* X{j}(:,ji)'*CC;
    end
    C(:,ji) = quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
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
    term1 = 0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - A{iv} * U{iv} * C,'fro')^2;
    end
    term2 = lambda * norm(C,'fro')^2;
    obj(iter) = term1 - term2;
    %obj(iter) = term1;
    
    %if  iter>=maxIter 
    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,V]=svd(C','econ');
        flag = 0;
    end
end
         
         
    
