function i = find_nearest_min(x,y)
[y_min,i] = min(abs(x-y));
if i>1
    i=i-1;
end
end