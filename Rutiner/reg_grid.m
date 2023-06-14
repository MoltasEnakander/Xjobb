function X = reg_grid(x,y,z)
[X1,X2] =  meshgrid(y,x);
nx = max(size(x));
ny = max(size(y));
nz = max(size(z));
n = nx*ny*nz;
X1 = reshape(X1,1,nx*ny);
X2 = reshape(X2,1,nx*ny);
X = zeros(3,n);
X(1,1:nx*ny) = X2;
X(2,1:nx*ny) = X1;
X(3,1:nx*ny) = z(1);
if nz > 1
    for i = 2:nz
        X(1,nx*ny*(i-1)+1:nx*ny*i) = X2;
        X(2,nx*ny*(i-1)+1:nx*ny*i) = X1;
        X(3,nx*ny*(i-1)+1:nx*ny*i) = z(i);
    end
end
end