addpath("Moltas\");
addpath("Moltas\bresenham\");
close all;
r_shape = 64;
c_shape = 64;
max_shape = 1;
im = generateScene(r_shape, c_shape, max_shape);

figure(1);
imagesc(abs(im));
colormap inferno;
colorbar;
set(gca,'YDir','normal')
xp = ceil(find(abs(im) > 0) / r_shape); % col ind of non-zero intensity
yp = mod(find(abs(im) > 0), r_shape); % row ind of non-zero intensity
yp(yp == 0) = r_shape; % intensities at the bottom results in modulo = 0
amp = zeros(size(xp, 1), 1);
for i = 1:size(xp, 1)
    amp(i) = im(yp(i), xp(i)); % complex intensity values
end

% convert array positions to real positions
xp = xp / (c_shape/2); % value between 0 and 2
xp = xp - 1;
yp = yp / (r_shape/2);
yp = yp - 1;

%Christers kod
hanning_flag = 1;
elev_angle = 0;
calrange = 7.5;
ff = 0;
fstart = 8;
fstop = 12;
nf = 256;
f = linspace(fstart,fstop,nf);
B = fstop - fstart;
fc = (fstart+fstop)/2.;
% Upplösning i y-led är c/(2B). Matcha i x-led (x-ledsupplösning är
% c/(2*fc*sin(theta_tot))
theta_tot = asin(B/fc).*180./pi; % Konvertera till grader!
ntheta = 256;
phi = linspace(-theta_tot/2,theta_tot/2,ntheta);
rcs=ptsource(xp,yp,amp,f,phi,calrange,ff);

figure(2);
imagesc(abs(rcs));
colormap inferno;
colorbar;

xmin = -1;
xmax = 1;
nx = 256;
ymin = -1;
ymax = 1;
ny = 256;
x = linspace(xmin,xmax,nx);
y = linspace(ymin,ymax,ny);
% Entydighet ges av upplösning*antal frekvens/vinkelpunkter
isar = calculate_image_MP(rcs,f,phi,calrange,x,y,ff,hanning_flag,elev_angle);
cmax = max(max(20.*log10(abs(isar))));
cmin = cmax-50;
plotcut_dB_in(isar,x,y,'test',cmin,cmax);


%Christers kod
% hanning_flag = 1;
% elev_angle = 0;
% calrange = 7.5;
% ff = 0;
% fstart = 8;
% fstop = 12;
% nf = 64;
% f = linspace(fstart,fstop,nf);
% B = fstop - fstart;
% fc = (fstart+fstop)/2.;
% % Upplösning i y-led är c/(2B). Matcha i x-led (x-ledsupplösning är
% % c/(2*fc*sin(theta_tot))
% theta_tot = asin(B/fc).*180./pi; % Konvertera till grader!
% ntheta = 64*8;
% phi2 = linspace(-theta_tot/2,theta_tot/2,ntheta);
% rcs2=ptsource(xp,yp,amp,f,phi2,calrange,ff);
% 
% 
% xmin = -1;
% xmax = 1;
% nx = 500;
% ymin = -1;
% ymax = 1;
% ny = 500;
% x = linspace(xmin,xmax,nx);
% y = linspace(ymin,ymax,ny);
% % Entydighet ges av upplösning*antal frekvens/vinkelpunkter
% isar2 = calculate_image_MP(rcs2,f,phi2,calrange,x,y,ff,hanning_flag,elev_angle);
% cmax = max(max(20.*log10(abs(isar))));
% cmin = cmax-50;
% plotcut_dB_in(isar2,x,y,'test',cmin,cmax);
