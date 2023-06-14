function h = han(n)
k=(1:n).';
h = (sin(pi*(k-1)./(n-1))).^2;
end