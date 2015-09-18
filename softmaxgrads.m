N = 1000;
a = linspace(1,20,1000);

for n = 10:5:N
    z = (a.^2).*((n-1)./(exp(a) + n - 1)).^2 + (n-1)*(1./((n-1).*exp(a) + 1)).^2;
    plot(a,z)
    print n
end