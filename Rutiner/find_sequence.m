function [ind_nr_out,x_out] = find_sequence(x,xmin,xmax)
amin = min(x);
amax = max(x);
awidth = amax-amin;
nx = max(max(size(x)));
dx = awidth./(nx-1);
x_out = x;
x_dep = x;
ind_nr = find(x);
ind_nr_out = find(x);
while xmin < amin
    amin = amin-awidth-dx;
    x_dep = x_dep-awidth-dx;
    ind_nr_out = [ind_nr ind_nr_out];
    x_out = [x_dep x_out];
end
x_dep = x;
while xmin > amax
    amax = amax+awidth+dx;
    x_dep = x_dep+awidth+dx;
    ind_nr_out = [ind_nr_out ind_nr];
    x_out = [x_out x_dep];
end
while xmax > amax
    amax = amax+awidth+dx;
    x_dep = x_dep+awidth+dx;
    ind_nr_out = [ind_nr_out ind_nr];
    x_out = [x_out x_dep];
end
% Pruning array to the useful part 
imin= find_nearest_min(x_out,xmin);
imax= find_nearest_max(x_out,xmax);
ind_nr_out = ind_nr_out(imin:imax);
x_out = x_out(imin:imax);
end