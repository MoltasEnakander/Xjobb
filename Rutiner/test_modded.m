close all
xp = [0];
yp = [0];
amp = [1];
hanning_flag = 0;
elev_angle = 0;
calrange = 7.5;
ff = 0;
fstart = 8;
fstop = 12;
nf = 100;
f = linspace(fstart,fstop,nf);
B = fstop - fstart;
fc = (fstart+fstop)/2.;
% Upplösning i y-led är c/(2B). Matcha i x-led (x-ledsupplösning är
% c/(2*fc*sin(theta_tot))
theta_tot = asin(B/fc).*180./pi; % Konvertera till grader!
ntheta = nf;
phi = linspace(-theta_tot/2,theta_tot/2,ntheta);
rcs=ptsource(xp,yp,amp,f,phi,calrange,ff);
cmax = max(max(20.*log10(abs(rcs))));
cmin = cmax-5;
plotcut_RCS_dB_in(rcs,phi,f,'RCS from point at (0,0)',cmin,cmax);

disp(sum(sum(abs(rcs).^2)))

xmin = -1;
xmax = 1;
nx = 256;
ymin = -1;
ymax = 1;
ny = 256;
x = linspace(xmin,xmax,nx);
y = linspace(ymin,ymax,ny);
z = 0;
% Entydighet ges av upplösning*antal frekvens/vinkelpunkter
isar = calculate_image_MP(rcs,f,phi,calrange,x,y,ff,hanning_flag,elev_angle);

disp(sum(sum(abs(isar).^2)))

cmax = max(max(20.*log10(abs(isar))));
cmin = cmax-50;
plotcut_dB_in(isar,x,y,'test',cmin,cmax);
% Ändrar formatet på input av koordinater och amplitud för att kunna göra
% framåtpropagering
X = reg_grid(x,y,z);
pts = reshape(isar,nx*ny,1);
% Rutin för framåtpropagering
rcs_fw =  calculate_image_MP_inv(pts,X,f,phi,calrange,ff);
disp(sum(sum(abs(rcs_fw).^2)))
cmax = max(max(20.*log10(abs(rcs_fw))));
cmin = min(min(20.*log10(abs(rcs_fw))));
plotcut_RCS_dB_in(rcs_fw,phi,f,'RCS forward from BP',cmin,cmax);
% Normaliseringen blir inte rätt men du kan bestämma normaliseringen genom
% att köra en puntkälla som här med amplituden 1
normalizing_constant1 = 1./sqrt(sum(sum(abs(rcs).^2)))
normalizing_constant = 1./sqrt(sum(sum(abs(rcs_fw).^2)))
norm_con = normalizing_constant1/normalizing_constant
rcs_fw2 = rcs_fw / norm_con;
cmax2 = max(max(20.*log10(abs(rcs_fw2))));
cmin2 = min(min(20.*log10(abs(rcs_fw2))));
plotcut_RCS_dB_in(rcs_fw2,phi,f,'RCS forward from BP after norm',cmin2,cmax2);
