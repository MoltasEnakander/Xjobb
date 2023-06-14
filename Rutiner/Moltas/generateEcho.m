function [rcs] = generateEcho(scene, calrange, ff, f, phi, xrange, yrange)
%generateEcho(scene, calrange, ff, f, phi, xrange, yrange, r_shape, c_shape)
    [y_ind, x_ind] = find(abs(scene) > 0);
    %disp([x_ind, y_ind])
    xp = xrange(x_ind);
    yp = yrange(y_ind);
    %disp([xp, yp])
    amp = zeros(size(x_ind, 1), 1);
    for i = 1:max(size(x_ind))
        amp(i) = scene(y_ind(i), x_ind(i)); % complex intensity values
    end
    
%     xp = ceil(find(abs(scene) > 0) / r_shape); % col ind of non-zero intensity
%     yp = mod(find(abs(scene) > 0), r_shape); % row ind of non-zero intensity
%     yp(yp == 0) = r_shape; % intensities at the bottom results in modulo = 0
%     amp = zeros(size(xp, 1), 1);
%     for i = 1:size(xp, 1)
%         amp(i) = scene(yp(i), xp(i)); % complex intensity values
%     end
    
    % convert array positions to real positions
%     xp = xp - 1;
%     yp = yp - 1;
%     xp = xp / (c_shape/xrange); % value between 0 and xmax
%     xp = xp - xrange/2;
%     yp = yp / (r_shape/yrange); % value between 0 and xmax
%     yp = yp - yrange/2;
    
    %Christers kod (flyttat till python)
    %hanning_flag = hanning_flag;
    %elev_angle = elev_angle;
    %calrange = cal_range;
    %ff = ff;
    %fstart = f_start;
    %fstop = f_stop;
    %nf = nf;
    %f = linspace(fstart,fstop,nf);
    %B = fstop - fstart;
    %fc = (fstart+fstop)/2.;
    % Upplösning i y-led är c/(2B). Matcha i x-led (x-ledsupplösning är
    % c/(2*fc*sin(theta_tot))
    %theta_tot = asin(B/fc).*180./pi; % Konvertera till grader!
    % ntheta = ntheta;
    %phi = linspace(-theta_tot/2,theta_tot/2,ntheta);
    
    rcs=ptsource(xp,yp,amp,f,phi,calrange,ff);

end