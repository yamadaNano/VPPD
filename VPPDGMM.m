% Perform VPPD on a GMM

m1 = [1; -1];
m2 = [-1; 1];
S1 = [2 0; 0 1];
S2 = [3 0; 0 1];

p = 0.2;

e1 = randn(2, 100*p);
e2 = randn(2, 100*(1-p));

y1 = bsxfun(@plus,S1*e1,m1);
y2 = bsxfun(@plus,S2*e2,m2);

fig = scatter(y1[0,:],y1[1,:])
axis([-2, 2, -2, 2])