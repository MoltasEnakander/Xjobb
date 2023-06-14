xp = [0];
yp = [0];
amp = [1];
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
ntheta = nf;
phi = linspace(-theta_tot/2,theta_tot/2,ntheta);
rcs=ptsource(xp,yp,amp,f,phi,calrange,ff);
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
plotcut_dB_in(isar,x,y,'',cmin,cmax);
