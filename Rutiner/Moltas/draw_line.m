function [im] = draw_line(im, x, y, intensity)
    if max(y) > size(im, 1) % out of bounds
               if y(1) > y(end) % remove values at the front
                   ind_y = find(y > size(im, 1));
                   ind_y = max(ind_y);
                   y = y(ind_y + 1:end);
                   x = x(ind_y + 1:end);
               else % remove values at the end
                   ind_y = find(y > size(im, 1));
                   ind_y = min(ind_y);
                   y = y(1:end-ind_y);
                   x = x(1:end-ind_y);
               end
               
               
    end
    
    if max(x) > size(im, 2) % out of bounds
               if x(1) > x(end) % remove values at the front
                   ind_x = find(x > size(im, 1));
                   ind_x = max(ind_x);
                   x = x(ind_x + 1:end);
                   y = y(ind_x + 1:end);
               else % remove values at the end
                   ind_x = find(x > size(im, 1));
                   ind_x = min(ind_x);
                   x = x(1:end-ind_x);
                   y = y(1:end-ind_x);
               end
    end
    
    im(sub2ind(size(im), y, x)) = im(sub2ind(size(im), y, x)) + intensity;
end