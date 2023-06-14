function [im] = generateScene(r_shape, c_shape, max_shape)
        im = zeros(r_shape, c_shape);
        n_shape = randi(max_shape); % nr of shapes to create, maximum max_shape

        for i = 1:n_shape
            %shape_type = randi(5);
            shape_type = 1;
            %shape_type = shape;

            %row = randi(r_shape-5); % generate row position
            %col = randi(c_shape-5); % generate col position
            %l = randi([10, floor(min(r_shape, c_shape)/2)]);
            row = 50;
            col = 50;
            l = 20;
            % boundary condition
            if row + l > r_shape
               row = row - l; 
            end

            % boundary condition
            if col + l > c_shape
               col = col - l; 
            end

            % randomise intensity inside the unit circle
            radius = rand(); % random radius 0-1
            phi = linspace(0, 2*pi, 1000);
            phi = phi(randi(size(phi, 2))); % random angle 0-2pi

            real_part = radius * cos(phi);
            imag_part = radius * sin(phi);
            intensity = real_part + imag_part * 1j; % random complex intensity
            intensity = 1 + 0j;

            if shape_type == 1
                % rectangle
%                 im(row, col:col+l) = im(row, col:col+l) + intensity;
%                 im(row:row+l, col) = im(row:row+l, col) + intensity;
%                 im(row+l, col:col+l) = im(row+l, col:col+l) + intensity;
%                 im(row:row+l, col+l) = im(row:row+l, col+l) + intensity;
                im(row, col:col+l) = intensity;
                im(row:row+l, col) = intensity;
                im(row+l, col:col+l) = intensity;
                im(row:row+l, col+l) =  intensity;

            elseif shape_type == 2
                % circle
                [rowsInImage, columnsInImage] = meshgrid(1:r_shape, 1:c_shape);
                circle1 = (rowsInImage - row).^2 ...
                + (columnsInImage - col).^2 <= l.^2;
                circle2 = (rowsInImage - row).^2 ...
                + (columnsInImage - col).^2 <= (l-1).^2;
                circle3 = int8(circle1) - int8(circle2);
                circle3 = double(circle3) * intensity;
                im = im + circle3';

            elseif shape_type == 3
                % triangle
                p1 = [row, row + floor(col/2)]; % top of triangle
                p2 = [row + l, col];            % bottom left
                p3 = [row + l, col + l];        % bottom right
                [x, y] = bresenham(p1(2),p1(1),p2(2),p2(1)); %top-botleft
                im = draw_line(im, x, y, intensity);
                [x, y] = bresenham(p2(2),p2(1),p3(2),p3(1)); %botleft-botright
                im = draw_line(im, x, y, intensity);
                [x, y] = bresenham(p1(2),p1(1),p3(2),p3(1)); %top-botright
                im = draw_line(im, x, y, intensity);

            elseif shape_type == 4
                % line between 2 random points            
                p1 = [row, col];
                p2 = [randi(r_shape), randi(c_shape)];
                %p2 = [row + randi(40), col+ randi(15)];
                [x, y] = bresenham(p1(2),p1(1),p2(2),p2(1)); %top-botleft
                im = draw_line(im, x, y, intensity);

            else
                % draw random number of points
                nr_of_points = randi(10);

                for j = 1:nr_of_points
                    % random position
                    row = randi(r_shape); % generate row position
                    col = randi(c_shape); % generate col position
                    % random intensity
                    radius = rand(); % random radius 0-1
                    phi = linspace(0, 2*pi, 1000);
                    phi = phi(randi(size(phi, 2))); % random angle 0-2pi

                    real_part = radius * cos(phi);
                    imag_part = radius * sin(phi);
                    intensity = real_part + imag_part * 1j; % random complex intensity
                    im(row, col) = im(row, col) + intensity;
                end            
            end
        end
end 