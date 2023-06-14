function plotcut_dB_in(isar,x,y,figtitle,cmin,cmax)
DY=(20.*log10(abs(isar))).';
fighandle=figure;
jetmap=inferno(2048);
% dx = (x(2)-x(1))/2.;
% dy = (y(2)-y(1))/2.;
dx = 0;
dy = 0;
line_col=[0 0 0];
colormap(jetmap);
pcolor(x-dx,y-dy,DY);caxis([cmin cmax]);shading interp;
set(gca,'TickDir','in')
axis([min(x)-dx max(x)-dx min(y)-dy max(y)-dy]);
axis('square');
% load('C:\Home\Matlab\Columbus_new\Pareto\bmw\bmw_ov_adj.mat');
% antal_ritningar=max(max(size(ritning)));
% fiover=fiover.*pi./180;
% for n=1:antal_ritningar
%     ovpicture=ritning{n};
%     x=ovpicture(:,1)*cos(fiover)+ovpicture(:,2)*sin(fiover);
%     y=-ovpicture(:,1)*sin(fiover)+ovpicture(:,2)*cos(fiover);
%     x=x+xover;
%     y=y+yover;
%     line(x,y,...
%         'Color','w',...
%         'LineWidth',[1],...
%         'EraseMode','normal');
% end;
xlabel('Cross-range (m)','Color',line_col);
ylabel('Down-range (m)','Color',line_col);
title(strrep(figtitle,'_',' '));
%*** Skapa figuren for amplitudskalning ***
h=colorbar;
label_h = ylabel(h, 'Amplitude (Rel.A.U.)', 'Rotation', 270);
label_h.Position(1) = 4;
label_h.Position(2) = -25;
set(h,'TickDir','out','Ycolor',line_col);
drawnow;
end