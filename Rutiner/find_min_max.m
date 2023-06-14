function [rmin,rmax] = find_min_max(calrange,phi,x,y,rd)
xmin = x(1);
xmax = x(end);
ymin = y(1);
ymax = y(end);
xr=calrange.*sin(phi.*pi./180.0);
yr=-calrange.*cos(phi.*pi./180.0);
xc=(xmin+xmax)./2.0;
yc=(ymin+ymax)./2.0;
rw=sqrt((xmax-xmin).^2+(ymax-ymin).^2);
rw=rw.*1.001;
rc=sqrt((xc-xr).^2+(yc-yr).^2)-calrange;
rmin=min(rc-rw./2.0)-abs(rd);
rmax=max(rc+rw./2.0)+abs(rd);
end