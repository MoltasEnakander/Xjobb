function [dr_fft,r_fft,fft_npad,fft_par1,fft_par2,fft_ind] = calculate_dr_fft(calrange,rcs,f,rmin,rmax,hanning_flag)
fstart =f(1);
fstop = f(end);
over_sampling = 16;
im=complex(0,1);
c = 0.299792458;
[nf,nphi] = size(rcs);
if hanning_flag==1
    [D,H] = meshgrid(han(nphi),han(nf));
    hanwin = D.*H;
elseif hanning_flag==2
    [D,H] = meshgrid(nuttall(nphi),nuttall(nf));
    hanwin = D.*H;
else
    hanwin = ones(nf,nphi);
end
wincorr = sum(sum(ones(nf,nphi)))/sum(sum(hanwin));
rcs = rcs.*hanwin.*wincorr;
B = fstop-fstart;
dr = c./B./2.0;
amb_range = dr.*(nf-1);
amin = -amb_range./2.0;
amax = amb_range./2.0;
fft_npad = (ceil((amax-amin)*over_sampling/dr));
% fft_npad = 2^nextpow2(fft_npad);
% npad = find_npad(nf);
dr_ff = dr.*(nf-1)./(fft_npad-1);
r_fft = linspace(amin,amax-dr_ff,fft_npad);
fc = (fstart+fstop)./2.0;
[F,] = meshgrid(f,linspace(1,nphi,nphi));
F=F.';
fft_par1 = (F./fc).*exp(+im.*4.*pi.*F.*((amax-amin)./2.0)./c);
dr_fft = ifft(fft_par1.*rcs,fft_npad);

[fft_ind,r_out] = find_sequence(r_fft,rmin,rmax);

r_fft = r_out;
dr_fft = dr_fft(fft_ind,:);
[R_FFT,~] = meshgrid(r_fft,linspace(1,nphi,nphi));
R_FFT = R_FFT.'; 
R_FACTOR = ((calrange+R_FFT)./(calrange)).^2;
fft_par2 = (fft_npad.*R_FACTOR./(nf)).*exp(-im.*2.0.*pi.*B.*R_FFT./c);
dr_fft=dr_fft.*fft_par2;
end