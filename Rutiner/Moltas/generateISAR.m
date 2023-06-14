function [isar] = generateISAR(rcs, f, phi, hanning_flag, elev_angle, calrange, ff, x, y)
    %xmin = -1;
    %xmax = 1;
    %nx = 401;
    %ymin = -1;
    %ymax = 1;
    %ny = 401;
    %x = linspace(xmin,xmax,nx);
    %y = linspace(ymin,ymax,ny);
    % Entydighet ges av upplösning*antal frekvens/vinkelpunkter
    isar = calculate_image_MP(rcs,f,phi,calrange,x,y,ff,hanning_flag,elev_angle);
end