clear;
clc;
cd ..
cd ..

root=cd;
addpath(genpath([root '/SLEP']));
                     % add the functions in the folder SLEP to the path
                   
% change to the original folder
cd Examples/L1;

m=1000;  n=1000;     % The data matrix is of size m x n

randNum=2;           % a random number

% ---------------------- generate random data ----------------------
randn('state',(randNum-1)*3+1);
A=randn(m,n);        % the data matrix

randn('state',(randNum-1)*3+2);
xOrin=randn(n,1);

randn('state',(randNum-1)*3+3);
y=[ones(2*m/5,1);...
    -ones(3*m/5, 1)];  % the response

rho=0.1;             % the regularization parameter
                     % it is a ratio between (0,1), if .rFlag=1

%----------------------- Set optional items -----------------------
opts=[];

% Starting point
opts.init=2;        % starting from a zero point

% Termination 
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=10;    % maximum number of iterations

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
funValList(1)=c1;
actsize=zeros(1,N);

tic;



for i=1:N-1
    p=1./(1+exp(-cList(i)-A*betaList(:,i)));
    for j=1:n
       if abs(transpose(A(:,j))*(Y-p))>=(2*lamList(i+1)-lamList(i))*lambdaMax
            NowAct(j)=j;
        end
        
    end
    [aaa,actsize(i+1)]=size(NowAct~=0);
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
    

end
    
    


toc;

% Now we use rung-kuta algorithm
