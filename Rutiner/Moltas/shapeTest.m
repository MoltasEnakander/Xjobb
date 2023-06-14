im = generateScene(256, 256, 3);
im2 = rescale(abs(im), 0,255);
image(im2);
colormap inferno
c = colorbar;
%set(c, 'ylim', [0 1])