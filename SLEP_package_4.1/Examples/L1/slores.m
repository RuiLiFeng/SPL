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

[a1,b1,c1]=LogisticR(A, y, 1, opts);

NowAct=zeros(1,n);
lamList=1:-1/N:0;
betaList=zeros(n,N);
cList=zeros(1,N);
cList(1)=b1;
funValList=zeros(1,N);
funValList(1)=c1(end);
thetaList=zeros(m,N);
actsize=zeros(1,N);
for i=1:m
    if(y(i)==1)
        thetaList(i,1)=1-mean(Y);
    else
        thetaList(i,1)=mean(Y);
    end
end
X=diag(y)*A;
r=zeros(N,1);
lambdaMax=norm(X'*thetaList(:,1),inf)/m;
tic;

for i=1:N-1
    nabla=log(thetaList(:,i)./(1-thetaList(:,i)))/m;
    r(i)=sqrt(0.25*m*(dual((lamList(i+1)/lamList(i))*thetaList(:,i))-dual(thetaList(:,i))+nabla'*thetaList(:,i)*(1-lamList(i+1)/lamList(i))));  
    NowAct=zeros(1,n);
    for j=1:n
        if 0.2*norm(X(:,j),2)*r(i)+abs(X(:,j)'*thetaList(:,i))>=m*lamList(i+1)*lambdaMax
            NowAct(j)=1;
        end
    end
    [aa,actsize(i+1)]=size(find(NowAct~=0));
     ActA=zeros(m,actsize(i+1));
    k=1;
    for j=1:n
        if NowAct(j)~=0
        ActA(:,k)=A(:,j);
        k=k+1;
        end
    end
    
    [x,cList(i+1),val]=LogisticR(ActA,y,lamList(i+1),opts);
    funValList(i+1)=val(end);
    k=1;
    for j=1:n
        if NowAct(j)~=0
            betaList(j,i+1)=x(k);
            k=k+1;
        end
    end
    thetaList(:,i+1)=exp(-X*betaList(:,i+1)-cList(i+1)*y)./(1+exp(-X*betaList(:,i+1)-cList(i+1)*y));
end
disp("slore running time:");
toc;
funCheck=zeros(1,N);
betaListCheck=zeros(n,N);
cListCheck=zeros(1,N);
tic;
for i=1:N
    [a,b,c]=LogisticR(A,y,lamList(i),opts);
    funCheck(i)=c(end);
    betaListCheck(:,i)=a;
    cListCheck(i)=b;
end
disp("origin algorithm running time");
toc;
is_no_change=norm(betaList-betaListCheck,'fro')+norm(funValList-funCheck,'fro')+norm(cList-cListCheck,'fro');
%is_no_change=is_no_change/(norm(betaListCheck,1)+norm(funCheck,1)+norm(cListCheck,1));
if is_no_change>=0.00001%relative error exceeds 1% then we believe it has some trouble
    disp("the solution is changed under slore!");
else
    disp("the solution is right");
end
disp(is_no_change);

    
