function [amp] = calculate_image_MP(rcs,f,phi,calrange,x,y,ff,hanning_flag,elev_angle)
z = 0.0;
fstart = f(1);
fstop = f(end);
nf = max(size(f));
xmin = x(1);
xmax = x(end);
nx = max(size(x));
ymin = y(1);
ymax = y(end);
ny = max(size(y));
nphi_tot = length(phi);
rd = 0.0;
[rmin,rmax] = find_min_max(calrange,phi,x,y,rd);
if ff == 0
   [dr,r_fft,~,~,~,~] = calculate_dr_fft(calrange,rcs,f,rmin,rmax,hanning_flag); 
else
    [dr,r_fft,~,~,~,~] = calculate_dr_fft_ff(rcs,f,rmin,rmax,hanning_flag);
end

nprocessors = getenv('NUMBER_OF_PROCESSORS');
nim = str2num(nprocessors);
if nim>12
    nim = 12;
end
if (nphi_tot<nim*2)
    nim = 1;
end

% disp(['Running ' num2str(nim) ' parallel processes'])
if (nim==1)
    rmin = r_fft(1);
    rmax = r_fft(end);
    nr = length(r_fft);
    phistart = phi(1);
    phistop = phi(end);
    nphi = length(phi);
    parr(1) = calrange;
    parr(2) = elev_angle;
    parr(3) = phistart;
    parr(4) = phistop;
    parr(5) = fstart;
    parr(6) = fstop;
    parr(7) = rmin;
    parr(8) = rmax;
    parr(9) = xmin;
    parr(10) = xmax;
    parr(11) = ymin;
    parr(12) = ymax;
    parr(13) = z;
    pari(1) = nphi;
    pari(2) = nf;
    pari(3) = nx;
    pari(4) = ny;
    pari(5) = nr;
    pari(6) = ff;
    amp = isar_image(dr,parr,pari);
    
elseif (nim == 4)
    parr(1) = calrange;
    parr(2) = elev_angle;
    parr(3) = fstart;
    parr(4) = fstop;
    parr(5) = xmin;
    parr(6) = xmax;
    parr(7) = ymin;
    parr(8) = ymax;
    parr(9) = z;
    
    pari(1) = nf;
    pari(2) = nx;
    pari(3) = ny;
    pari(4) = ff;
    rmin = ones(1,nim).*r_fft(1);
    rmax = ones(1,nim).*r_fft(end);
    nr = ones(1,nim).*length(r_fft);
    na = length(phi);
    nl = floor(na/4);
    nstart = [1 1+nl 1+2*nl 1+3*nl];
    nstop = [nl 2*nl 3*nl na];
    
    dr0 = dr(:,nstart(1):nstop(1));
    nphi(1) = nstop(1)-nstart(1)+1;
    phistart(1) = phi(nstart(1));
    phistop(1) = phi(nstop(1));
    dr1 = dr(:,nstart(2):nstop(2));
    nphi(2) = nstop(2)-nstart(2)+1;
    phistart(2) = phi(nstart(2));
    phistop(2) = phi(nstop(2));
    dr2 = dr(:,nstart(3):nstop(3));
    nphi(3) = nstop(3)-nstart(3)+1;
    phistart(3) = phi(nstart(3));
    phistop(3) = phi(nstop(3));
    dr3 = dr(:,nstart(4):nstop(4));
    nphi(4) = nstop(4)-nstart(4)+1;
    phistart(4) = phi(nstart(4));
    phistop(4) = phi(nstop(4));
    [amp0,amp1,amp2,amp3] = isar_image_MP4(dr0,dr1,dr2,dr3,phistart,phistop,nphi,rmin,rmax,nr,parr,pari);
    amp = (amp0.*nphi(1)+amp1.*nphi(2)+amp2.*nphi(3)+amp3.*nphi(4))./nphi_tot;
    
elseif (nim == 8)
    parr(1) = calrange;
    parr(2) = elev_angle;
    parr(3) = fstart;
    parr(4) = fstop;
    parr(5) = xmin;
    parr(6) = xmax;
    parr(7) = ymin;
    parr(8) = ymax;
    parr(9) = z;
    
    pari(1) = nf;
    pari(2) = nx;
    pari(3) = ny;
    pari(4) = ff;
    rmin = ones(1,nim).*r_fft(1);
    rmax = ones(1,nim).*r_fft(end);
    nr = ones(1,nim).*length(r_fft);
    na = length(phi);
    nl = floor(na/nim);
    nstart = [1 1+nl 1+2*nl 1+3*nl 1+4*nl 1+5*nl 1+6*nl 1+7*nl];
    nstop = [nl 2*nl 3*nl 4*nl 5*nl 6*nl 7*nl na];
    
    dr0 = dr(:,nstart(1):nstop(1));
    nphi(1) = nstop(1)-nstart(1)+1;
    phistart(1) = phi(nstart(1));
    phistop(1) = phi(nstop(1));
    dr1 = dr(:,nstart(2):nstop(2));
    nphi(2) = nstop(2)-nstart(2)+1;
    phistart(2) = phi(nstart(2));
    phistop(2) = phi(nstop(2));
    dr2 = dr(:,nstart(3):nstop(3));
    nphi(3) = nstop(3)-nstart(3)+1;
    phistart(3) = phi(nstart(3));
    phistop(3) = phi(nstop(3));
    dr3 = dr(:,nstart(4):nstop(4));
    nphi(4) = nstop(4)-nstart(4)+1;
    phistart(4) = phi(nstart(4));
    phistop(4) = phi(nstop(4));
    dr4 = dr(:,nstart(5):nstop(5));
    nphi(5) = nstop(5)-nstart(5)+1;
    phistart(5) = phi(nstart(5));
    phistop(5) = phi(nstop(5));
    dr5 = dr(:,nstart(6):nstop(6));
    nphi(6) = nstop(6)-nstart(6)+1;
    phistart(6) = phi(nstart(6));
    phistop(6) = phi(nstop(6));
    dr6 = dr(:,nstart(7):nstop(7));
    nphi(7) = nstop(7)-nstart(7)+1;
    phistart(7) = phi(nstart(7));
    phistop(7) = phi(nstop(7));
    dr7 = dr(:,nstart(8):nstop(8));
    nphi(8) = nstop(8)-nstart(8)+1;
    phistart(8) = phi(nstart(8));
    phistop(8) = phi(nstop(8));
    [amp0,amp1,amp2,amp3,amp4,amp5,amp6,amp7] = isar_image_MP8(dr0,dr1,dr2,dr3,dr4,dr5,dr6,dr7,phistart,phistop,nphi,rmin,rmax,nr,parr,pari);
    amp = (amp0.*nphi(1)+amp1.*nphi(2)+amp2.*nphi(3)+amp3.*nphi(4)+amp4.*nphi(5)+amp5.*nphi(6)+amp6.*nphi(7)+amp7.*nphi(8))./nphi_tot;

elseif (nim == 12)
    parr(1) = calrange;
    parr(2) = elev_angle;
    parr(3) = fstart;
    parr(4) = fstop;
    parr(5) = xmin;
    parr(6) = xmax;
    parr(7) = ymin;
    parr(8) = ymax;
    parr(9) = z;
    
    pari(1) = nf;
    pari(2) = nx;
    pari(3) = ny;
    pari(4) = ff;
    rmin = ones(1,nim).*r_fft(1);
    rmax = ones(1,nim).*r_fft(end);
    nr = ones(1,nim).*length(r_fft);
    na = length(phi);
    nl = floor(na/nim);
    nstart = [1 1+nl 1+2*nl 1+3*nl 1+4*nl 1+5*nl 1+6*nl 1+7*nl 1+8*nl 1+9*nl 1+10*nl 1+11*nl];
    nstop = [nl 2*nl 3*nl 4*nl 5*nl 6*nl 7*nl 8*nl 9*nl 10*nl 11*nl na];
    
    dr0 = dr(:,nstart(1):nstop(1));
    nphi(1) = nstop(1)-nstart(1)+1;
    phistart(1) = phi(nstart(1));
    phistop(1) = phi(nstop(1));
    dr1 = dr(:,nstart(2):nstop(2));
    nphi(2) = nstop(2)-nstart(2)+1;
    phistart(2) = phi(nstart(2));
    phistop(2) = phi(nstop(2));
    dr2 = dr(:,nstart(3):nstop(3));
    nphi(3) = nstop(3)-nstart(3)+1;
    phistart(3) = phi(nstart(3));
    phistop(3) = phi(nstop(3));
    dr3 = dr(:,nstart(4):nstop(4));
    nphi(4) = nstop(4)-nstart(4)+1;
    phistart(4) = phi(nstart(4));
    phistop(4) = phi(nstop(4));
    dr4 = dr(:,nstart(5):nstop(5));
    nphi(5) = nstop(5)-nstart(5)+1;
    phistart(5) = phi(nstart(5));
    phistop(5) = phi(nstop(5));
    dr5 = dr(:,nstart(6):nstop(6));
    nphi(6) = nstop(6)-nstart(6)+1;
    phistart(6) = phi(nstart(6));
    phistop(6) = phi(nstop(6));
    dr6 = dr(:,nstart(7):nstop(7));
    nphi(7) = nstop(7)-nstart(7)+1;
    phistart(7) = phi(nstart(7));
    phistop(7) = phi(nstop(7));
    dr7 = dr(:,nstart(8):nstop(8));
    nphi(8) = nstop(8)-nstart(8)+1;
    phistart(8) = phi(nstart(8));
    phistop(8) = phi(nstop(8));  
    dr8 = dr(:,nstart(9):nstop(9));
    nphi(9) = nstop(9)-nstart(9)+1;
    phistart(9) = phi(nstart(9));
    phistop(9) = phi(nstop(9));
    dr9 = dr(:,nstart(10):nstop(10));
    nphi(10) = nstop(10)-nstart(10)+1;
    phistart(10) = phi(nstart(10));
    phistop(10) = phi(nstop(10));
    dr10 = dr(:,nstart(11):nstop(11));
    nphi(11) = nstop(11)-nstart(11)+1;
    phistart(11) = phi(nstart(11));
    phistop(11) = phi(nstop(11));
    dr11 = dr(:,nstart(12):nstop(12));
    nphi(12) = nstop(12)-nstart(12)+1;
    phistart(12) = phi(nstart(12));
    phistop(12) = phi(nstop(12));
    
    [amp0,amp1,amp2,amp3,amp4,amp5,amp6,amp7,amp8,amp9,amp10,amp11] = isar_image_MP12(dr0,dr1,dr2,dr3,dr4,dr5,dr6,dr7,dr8,dr9,dr10,dr11,phistart,phistop,nphi,rmin,rmax,nr,parr,pari);
    amp = (amp0.*nphi(1)+amp1.*nphi(2)+amp2.*nphi(3)+amp3.*nphi(4)+amp4.*nphi(5)+amp5.*nphi(6)+amp6.*nphi(7)+amp7.*nphi(8)+amp8.*nphi(9)+amp9.*nphi(10)+amp10.*nphi(11)+amp11.*nphi(12))./nphi_tot;
   
end
end