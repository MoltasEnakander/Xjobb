function plotcut_RCS_dB_in(rcs,x,y,figtitle,cmin,cmax)
    DY=(20.*log10(abs(rcs)));
    fighandle=figure;
    jetmap=inferno(2048);
    line_col=[0 0 0];
    dx = (x(2)-x(1))/2.;
    dy = (y(2)-y(1))/2.;
    colormap(jetmap);
    pcolor(x-dx,y-dy,DY);caxis([cmin cmax]);shading interp;
    set(gca,'TickDir','in')
    axis([min(x) max(x) min(y) max(y)]);
    axis('square');
    xlabel('Angle (°)','Color', line_col);
    ylabel('Frequency (GHz)','Color', line_col);
    title(strrep(figtitle,'_',' '));
    % *** Skapa figuren for amplitudskalning ***
    h=colorbar;
    set(h,'TickDir','out','Ycolor',line_col);
    drawnow;
end