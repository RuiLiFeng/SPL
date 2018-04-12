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
    -ones(3*m/4,1)];  % the response

rho=0.1;             % the regularization parameter
                     % it is a ratio between (0,1), if .rFlag=1

%----------------------- Set optional items -----------------------
opts=[];

% Starting point
opts.init=2;        % starting from a zero point

% Termination 
opts.tFlag=20;       % run .maxIter iterations
opts.maxIter=200;    % maximum number of iterations

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
funValList(1)=c1;
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
lambdaMax=norm(X'*thetaList(:,1),inf)/m;
ss=find(thetaList(:,1)'*X==m*lambdaMax|thetaList(:,1)'*X==-m*lambdaMax);
tic;
nabla=zeros(m,1);
for i=1:m
    nabla(i)=log(thetaList(i,1)/(1-thetaList(i,1)))/m;
end
for i=1:2
 p=1./(1+exp(-cList(i)-A*betaList(:,i)));
    for j=1:n
       if abs(transpose(A(:,j))*(Y-p))>=(2*lamList(i+1)-lamList(i))*lambdaMax*m
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
    for j=1:m
        thetaList(j,i+1)=exp((-X(j,:)*betaList(:,i+1)-y(j)*cList(i+1))/m)/(1+exp((-X(j,:)*betaList(:,i+1)-y(j)*cList(i+1))/m));
    end
end




for i=3:N-1
    
  
   r=sqrt(0.25*m*(dualfunction(thetaList(:,i-1),m)-2*dualfunction(thetaList(:,i),m)+dualfunction((lamList(i+1)/lamList(i))*thetaList(:,i),m)));
    dd=find(NowAct~=0);
    
  
 %{    r=sqrt(0.5*m*(dualfunction((lamList(i+1)/lamList(i))*thetaList(:,i),m)-dualfunction(thetaList(:,i),m)+nabla'*thetaList(:,1)*(1-lamList(i+1)/lamList(i))));
     if i==1
         dd=[ss,1];
     else
         dd=find(NowAct~=0);
     end
    %} 
     
     
     
    NowAct=zeros(1,n);
    for j=1:n
        T1=0;
        T2=0;
        Px=X(:,j)-y'*X(:,j)*y/(y'*y);
        x0=sign(transpose(thetaList(:,i))*X(:,dd(1)))*X(:,dd(1));
       
        Px0=x0-y'*x0*y/(y'*y);
        
        
        
        if Px~=0
            d=m*(lamList(i)-lamList(i+1))*lambdaMax/(r*norm(Px0,2));
            if Px'*Px0>=d*norm(Px,2)*norm(Px0,2)
                T1=r*norm(Px,2)-thetaList(:,i)'*X(:,j);
                
            else
                a0=(Px'*Px0)^2-d^2*norm(Px,2)^2*norm(Px0,2)^2;
                a1=2*Px'*Px0*norm(Px0,2)^2*(1-d^2);
                a2=norm(Px0,2)^4*(1-d^2);
                deta=a1^2-4*a0*a2;
                u2=(-a1+sqrt(deta))/(2*a2);
                T1=r*norm(Px+u2*Px0,2)-u2*m*lambdaMax*(lamList(i)-lamList(i+1))-thetaList(:,i)'*X(:,j);
                
            end
            Px=-Px;
            if Px'*Px0>=d*norm(Px,2)*norm(Px0,2)
                T2=r*norm(Px,2)+thetaList(:,i)'*X(:,j);
                
            else
                a0=(Px'*Px0)^2-d^2*Px'*Px*Px0'*Px0;
                a1=2*Px'*Px0*Px0'*Px0*(1-d^2);
                a2=(Px0'*Px0)^2*(d-d^2);
                deta=a1^2-4*a0*a2;
                u2=(-a1+sqrt(deta))/(2*a2);
                T2=r*norm(Px+u2*Px0,2)-u2*m*lambdaMax*(lamList(i)-lamList(i+1))+thetaList(:,i)'*X(:,j);
            end
            if max([T1,T2])>=m*lamList(i+1)*lambdaMax
                NowAct(j)=j;
            end
        end
    end
     [temp,actsize(i+1)]=size(find(NowAct~=0));
    if actsize(i+1)~=0
    ActA=zeros(m,actsize(i+1));
    k=1;
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
    else
        funValList(i+1)=c1;
        cList(i+1)=b1;
    end
    for j=1:m
        thetaList(j,i+1)=exp((-X(j,:)*betaList(:,i+1)-y(j)*cList(i+1))/m)/(1+exp((-X(j,:)*betaList(:,i+1)-y(j)*cList(i+1))/m));
    end
   
end
toc;
    
    
    
    
    
    