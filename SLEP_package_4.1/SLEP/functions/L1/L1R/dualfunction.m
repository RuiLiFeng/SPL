function g=dualfunction(x,m)
if size(x)~=[m,1]
    error('\n Check the length of x!\n');
end
g=0;
for i=1:m
    g=g+x(i)*log(x(i))+(1-x(i))*log(1-x(i));
end
g=g/m;