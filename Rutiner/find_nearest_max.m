function i = find_nearest_max(x,y)
[y_min,i] = min(abs(x-y));
ny = max(max(size(y)));
if i<ny
    i = i+1;
end
end