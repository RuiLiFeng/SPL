function[value]=dual(theta)
% input must be a colum vector which has m elements
[m,~]=size(theta);
value=(theta'*log(theta)+(1-theta)'*log(1-theta))/m;
