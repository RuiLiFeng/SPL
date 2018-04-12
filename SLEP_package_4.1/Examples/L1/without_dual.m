clear;
clc;
cd ..
cd ..

root=cd;
addpath(genpath([root '/SLEP']));
                     % add the functions in the folder SLEP to the path
                   
% change to the original folder
cd Examples/L1;

m=200;  n=10000;     % The data matrix is of size m x n

randNum=2;           % a random number

% ---------------------- generate random data ----------------------
randn('state',(randNum-1)*3+1);
A=randn(m,n);        % the data matrix

randn('state',(randNum-1)*3+2);
xOrin=randn(n,1);

randn('state',(randNum-1)*3+3);
y=[ones(m/4,1);...
    -ones(3*m/4, 1)];  % the response

rho=0.1;             % the regularization parameter
                     % it is a ratio between (0,1), if .rFlag=1

%----------------------- Set optional items -----------------------
opts=[];

% Starting point
opts.init=2;        % starting from a zero point

% Termination 
opts.tFlag=5000;       % run .maxIter iterations
opts.maxIter=5000;    % maximum number of iterations

% Normalization
opts.nFlag=0;       % without normalization

% Regularization
opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)
%opts.rsL2=0.01;     % the squared two norm term

% Group Property
opts.sWeight=[1,1]; % set the weight for positive and negative samples

%----------------------- Run the code LeastR -----------------------
N=100;%siez of questions
Y=0.5*y+0.5;
p=mean(Y)*ones(m,1);
lambdaMax=norm(A'*(Y-p),inf);% ÇólambdaMax
[a1,b1,c1]=LogisticR(A, y, 1, opts);

NowAct=zeros(1,n);
lamList=1:-1/N:0;
betaList=zeros(n,N);
cList=zeros(1,N);
cList(1)=b1;
funValList=zeros(1,N);
funValList(1)=c1(end);
actsize=zeros(1,N);

strong_rule_time=tic;

metric = max(A'*A, [], 2);

for i=1:N-1
    p=1./(1+exp(-cList(i)-A*betaList(:,i)));
    for j=1:n
       if abs(transpose(A(:,j))*(Y-p))+0.5*metric(i+1)*norm(betaList(:,i),1)>=lamList(i+1)*lambdaMax
           NowAct(j)=j;
       end
        
    end
    [aaa,actsize(i+1)]=size(find(NowAct~=0));
    k=1;
    ActA=zeros(m,actsize(i+1));
    for j=1:n
        if NowAct(j)~=0
            ActA(:,k)=A(:,j);
            k=k+1;
        end
    end
    [x,cList(i+1),val]=LogisticR(ActA, y, lamList(i+1), opts);
    funValList(i+1)=val(end);
    k=1;
    for j=1:n
        if NowAct(j)~=0
            betaList(j,i+1)=x(k);
            k=k+1;
        end
    end
            

end
    
    

disp("strong rule running time:");
toc(strong_rule_time);
funCheck=zeros(1,N);
betaListCheck=zeros(n,N);
cListCheck=zeros(1,N);
without_screen_time=tic;
for i=1:N
    [a,b,c]=LogisticR(A,y,lamList(i),opts);
    funCheck(i)=c(end);
    betaListCheck(:,i)=a;
    cListCheck(i)=b;
end
disp("origin algorithm running time");
toc(without_screen_time);
is_no_change=norm(betaList-betaListCheck,2)+norm(funValList-funCheck,2)+norm(cList-cListCheck,2);
strong_rule_aberror=is_no_change;
is_no_change=is_no_change/(norm(betaListCheck,2)+norm(funCheck,2)+norm(cListCheck,2));
if is_no_change>=0.0001
    disp("the solution is changed under strong rule!");
    disp(strong_rule_aberror);
else
    disp("the solution of strong rule is right, the absolute error is:");
    disp(strong_rule_aberror);
end
strong_rule_funVal=funValList;
strong_rule_betaList=betaList;
strong_rule_cList=cList;
strong_rule_actsize=actsize;
strong_rule_is_no_change=is_no_change;

