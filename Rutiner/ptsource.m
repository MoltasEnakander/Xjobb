function rcs=ptsource(x,y,amp,f,theta,calrange,ff)
% ff=0 betyder närfält, ff=1 betyder fjärrfält
% f vektor med frekvenser i GHz
% theta är vektor med vinkel i grader
c=0.299792458;
im=complex(0,1);
nf = max(size(f));
ntheta = max(size(theta));
n = max(size(amp));
if ff==0
    xr=calrange.*sin(theta.*pi./180);
    yr=-calrange.*cos(theta.*pi./180);
    rcs = zeros(nf,ntheta);
    for i=1:n
        rd=-calrange+sqrt((x(i)-xr).^2+(y(i)-yr).^2);
        [RD,F]=meshgrid(rd,f);
        rcs=rcs+amp(i).*exp(-im.*4.*pi.*RD.*F./c).*(calrange./(calrange+RD)).^2;
    end
else
    rcs = zeros(nf,ntheta);
    for i=1:n
        rd=-x(i).*sin(theta.*pi./180)+y(i).*cos(theta.*pi./180);normalizing_constant1
        [RD,F]=meshgrid(rd,f);
        rcs=rcs+amp(i).*exp(-im.*4.*pi.*RD.*F./c);
    end
end
end