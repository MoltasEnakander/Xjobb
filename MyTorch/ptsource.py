import numpy as np
import torch


# Code from Christer rewritten in python
""" Calculate RCS measurements for angle-frequency """
def ptsource(x, y, amp, f, phi, calrange, ff, device, method):
    # x contains x-positions of pts
    # y contains y-positions of pts
    # amp contains the amplitudes of the point sources
    # f contains frequencies (GHz)
    # phi contains angles (degrees)
    # calrange is the distance to the center of the scene
    # ff indicates near-field (f=0) or far-field (f=1)
    
    c = 0.299792458
    nf = np.max(f.shape)
    nphi = np.max(phi.shape)
    n = np.max(amp.shape)
    rcs = torch.zeros((nf, nphi)).to(device).type(torch.complex64)
    if not ff: # near-field
        xr = calrange * torch.sin(phi * np.pi / 180)
        yr = -calrange * torch.cos(phi * np.pi / 180)

        if method == 0: # everything at once
            xdiff = x.reshape(-1, 1) - xr
            ydiff = y.reshape(-1, 1) - yr
            rd = -calrange + torch.sqrt(xdiff**2 + ydiff**2)
            RD = rd.repeat(1, nf)
            RD = torch.reshape(RD, (n, nf, nphi))
            F = f[:, None].repeat(1, nphi)
            amp = amp.view(n, 1, 1)

            # calc rcs for n layers and then combine them
            rcs = amp * torch.exp(-4j*np.pi*RD*F/c) * (calrange/(calrange + RD))**2
            rcs = torch.sum(rcs, dim=0)

        elif method == 1: # compute batches and combine
            step = 1024 # nr of layers to combine
            F = f[:, None].repeat(1, nphi)
            for i in range(0, n, step):
                if n - i >= step: # it is possible to extract a complete batch of layers
                    xdiff = x[i:i+step].reshape(-1, 1) - xr
                    ydiff = y[i:i+step].reshape(-1, 1) - yr
                    amp_batch = amp[0:step].view(step, 1, 1)
                    amp = amp[step:]

                else: # compute the remaining layers
                    step = n % step
                    xdiff = x[-step:].reshape(-1, 1) - xr
                    ydiff = y[-step:].reshape(-1, 1) - yr
                    amp_batch = amp.view(step, 1, 1)

                rd = -calrange + torch.sqrt(xdiff**2 + ydiff**2)
                RD = rd.repeat(1, nf)
                RD = torch.reshape(RD, (step, nf, nphi))

                # calculate rcs for the batch
                M = torch.exp(-4j*np.pi*RD*F/c) * (calrange/(calrange + RD))**2
                rcs_batch = amp_batch * M
                rcs_batch = torch.sum(rcs_batch, dim=0)
                rcs = rcs_batch + rcs
                torch.cuda.empty_cache()

        elif method == 2: # one layer at a time
            for i in range(n):
                rd = -calrange + torch.sqrt((x[i] - xr)**2 + (y[i] - yr)**2)
                [RD, F] = torch.meshgrid(rd, f, indexing="xy") # "xy" gives the correct grid, otherwise the resulting rcs needs to be transposed
                rcs += amp[i] * torch.exp(-4j*np.pi*RD*F/c) * (calrange/(calrange + RD))**2        

    elif ff: # far-field
        if method == 0: # everything at once
            rd = -x.reshape(-1, 1) * torch.sin(phi * np.pi / 180) + y.reshape(-1, 1) * torch.cos(phi * np.pi / 180)                
            RD = rd.repeat(1, nf)
            RD = torch.reshape(RD, (n, nf, nphi))
            F = f[:, None].repeat(1, nphi)
            amp = amp.view(n, 1, 1)

            # calc rcs for n layers and then combine them
            rcs = amp * torch.exp(-4j*np.pi*RD*F/c)
            rcs = torch.sum(rcs, dim=0)            

        elif method == 1: # compute batches and combine
            step = 5000 # nr of layers to combine
            F = f[:, None].repeat(1, nphi)
            for i in range(0, n, step):
                if n - i >= step: # it is possible to extract a complete batch of layers
                    amp_batch = amp[i:i+step].view(step, 1, 1)
                    rd = -x[i:i+step] * torch.sin(phi * np.pi / 180) + y[i:i+step] * torch.cos(phi * np.pi / 180)

                else: # compute the remaining layers
                    step = n % step
                    amp_batch = amp[-step:].view(step, 1, 1)
                    rd = -x[-step:] * torch.sin(phi * np.pi / 180) + y[-step:] * torch.cos(phi * np.pi / 180)

                RD = rd.repeat(1, nf)
                RD = torch.reshape(RD, (step, nf, nphi))

                # calculate rcs for the batch
                rcs_batch = amp_batch * torch.exp(-4j*np.pi*RD*F/c)
                rcs_batch = torch.sum(rcs_batch, dim=0)
                rcs += rcs_batch


        elif method == 2: # one layer at a time
            for i in range(n):
                rd = -x[i] * torch.sin(phi * np.pi / 180) + y[i] * torch.cos(phi * np.pi / 180)
                [RD, F] = torch.meshgrid(rd, f, indexing="xy")
                rcs += amp[i] * torch.exp(-4j*np.pi*RD*F/c)

    return rcs


""" Calculate nbins number of random RCS measurements for angle-frequency """
def ptsource_sparse(x, y, amp, f, phi, calrange, ff, nbins, device):
    # x contains x-positions of pts
    # y contains y-positions of pts
    # amp contains the amplitudes of the point sources
    # f contains frequencies (GHz)
    # phi contains angles (degrees)
    # calrange is the distance to the center of the scene
    # ff indicates near-field (f=0) or far-field (f=1)
    # nbins denotes the nr of measurements in the sparse rcs
    
    c = 0.299792458
    n = np.max(amp.shape)    
    rcs = torch.zeros(nbins).to(device).type(torch.complex64)

    if not ff: # near-field
        xr = calrange * torch.sin(phi * np.pi / 180)
        yr = -calrange * torch.cos(phi * np.pi / 180)

        step = 1024 # nr of layers to combine            
        for i in range(0, n, step):
            if n - i >= step: # it is possible to extract a complete batch of layers
                xdiff = x[i:i+step].reshape(-1, 1) - xr
                ydiff = y[i:i+step].reshape(-1, 1) - yr
                amp_batch = amp[i:i+step].view(step, 1)

            else: # compute the remaining layers
                step = n % step
                xdiff = x[-step:].reshape(-1, 1) - xr
                ydiff = y[-step:].reshape(-1, 1) - yr
                amp_batch = amp[-step:].view(step, 1)

            rd = -calrange + torch.sqrt(xdiff**2 + ydiff**2)

            # calculate rcs for the batch
            M = torch.exp(-4j*np.pi*rd*f/c) * (calrange/(calrange + rd))**2
            rcs_batch = amp_batch * M
            rcs_batch = torch.sum(rcs_batch, dim=0)
            rcs = rcs_batch + rcs
            torch.cuda.empty_cache()

    elif ff:
        step = 1024 # nr of layers to combine
        for i in range(0, n, step):
            if n - i >= step: # it is possible to extract a complete batch of layers
                amp_batch = amp[i:i+step].view(step, 1, 1)
                rd = -x[i:i+step] * torch.sin(phi * np.pi / 180) + y[i:i+step] * torch.cos(phi * np.pi / 180)

            else: # compute the remaining layers
                step = n % step
                amp_batch = amp[-step:].view(step, 1, 1)
                rd = -x[-step:] * torch.sin(phi * np.pi / 180) + y[-step:] * torch.cos(phi * np.pi / 180)

            # calculate rcs for the batch
            rcs_batch = amp_batch * torch.exp(-4j*np.pi*rd*f/c)
            rcs_batch = torch.sum(rcs_batch, dim=0)
            rcs += rcs_batch
            torch.cuda.empty_cache()

    return rcs


""" Calculate nbins number of random RCS measurements for angle-frequency, same as ptsource_sparse
    but this function handles multiple channels """
def ptsource_sparse2(nscenes, x, y, amp, f, phi, calrange, ff, nbins, device):
    # nscenes denotes the nr of scenes to calculate sparse rcs
    # x contains x-positions of pts
    # y contains y-positions of pts
    # amp contains the amplitudes of the point sources
    # f contains frequencies (GHz)
    # phi contains angles (degrees)
    # calrange is the distance to the center of the scene
    # ff indicates near-field (f=0) or far-field (f=1)
    # nbins denotes the nr of measurements in the sparse rcs

    c = 0.299792458
    n = np.max(amp.shape)    
    rcs = torch.zeros(nscenes, nbins).to(device).type(torch.complex64)

    if not ff: # near-field
        xr = calrange * torch.sin(phi * np.pi / 180)
        yr = -calrange * torch.cos(phi * np.pi / 180)

        step = 1024 # nr of layers to combine            
        for i in range(0, n, step):
            if n - i >= step: # it is possible to extract a complete batch of layers
                xdiff = x[i:i+step].reshape(-1, 1) - xr
                ydiff = y[i:i+step].reshape(-1, 1) - yr                    
                amp_batch = amp[i:i+step].view(step, 1)

            else: # compute the remaining layers
                step = n % step
                xdiff = x[-step:].reshape(-1, 1) - xr
                ydiff = y[-step:].reshape(-1, 1) - yr                    
                amp_batch = amp[-step:].view(step, 1)

            rd = -calrange + torch.sqrt(xdiff**2 + ydiff**2)

            # calculate rcs for the batch
            M = torch.exp(-4j*np.pi*rd*f/c) * (calrange/(calrange + rd))**2
            rcs_batch = amp_batch * M
            rcs_batch = rcs_batch.view(nscenes, step//nscenes, nbins)
            rcs_batch = torch.sum(rcs_batch, dim=1)
            rcs = rcs_batch + rcs
            torch.cuda.empty_cache()

    elif ff:
        step = 1024 # nr of layers to combine
        for i in range(0, n, step):
            if n - i >= step: # it is possible to extract a complete batch of layers
                amp_batch = amp[i:i+step].view(step, 1, 1)
                rd = -x[i:i+step] * torch.sin(phi * np.pi / 180) + y[i:i+step] * torch.cos(phi * np.pi / 180)

            else: # compute the remaining layers
                step = n % step
                amp_batch = amp[-step:].view(step, 1, 1)
                rd = -x[-step:] * torch.sin(phi * np.pi / 180) + y[-step:] * torch.cos(phi * np.pi / 180)

            # calculate rcs for the batch
            rcs_batch = amp_batch * torch.exp(-4j*np.pi*rd*f/c)
            rcs_batch = rcs_batch.view(nscenes, step//nscenes, 1)
            rcs_batch = torch.sum(rcs_batch, dim=1)
            rcs += rcs_batch
            torch.cuda.empty_cache()

    return rcs


""" Calculate RCS measurements for angle-time """
def ptsource_time(x, y, amp, f, phi, calrange, ff, device, method):
    # x contains x-positions of pts
    # y contains y-positions of pts
    # amp contains the amplitudes of the point sources
    # f contains frequencies (GHz)
    # phi contains angles (degrees)
    # calrange is the distance to the center of the scene
    # ff indicates near-field (f=0) or far-field (f=1)
    
    c = 0.299792458
    nf = np.max(f.shape)
    nphi = np.max(phi.shape)
    nt = nf
    ts = 1/(torch.max(f) - torch.min(f))
    BW = torch.max(f) - torch.min(f) # bandwidth
    fc = (torch.max(f) + torch.min(f)) / 2
    n = np.max(amp.shape)    
    ts = 1/(torch.max(f) - torch.min(f))
    t_mid = (2*calrange)/c
    t_max = t_mid + ts * nt/2
    t_min = t_mid - ts * (nt/2 - 1)
    t_grid = torch.arange(nt).to(device)
    t_grid = t_min + ts * t_grid # array containing values from t_min -> t_max with step size ts
    nsinc_comps = 5 # nr of sinc components to add
    sinc_comps = torch.linspace(-np.floor(nsinc_comps/2), np.floor(nsinc_comps/2), nsinc_comps).type(torch.LongTensor).to(device)
    rcs = torch.zeros((nt, nphi)).to(device).type(torch.complex64)
    if not ff: # near-field
        xr = calrange * torch.sin(phi * np.pi / 180)
        yr = -calrange * torch.cos(phi * np.pi / 180)

        if method == 1:
            step = 2048 # nr of layers to combine
            for i in range(0, n, step):
                if n - i >= step: # it is possible to extract a complete batch of layers
                    xdiff = x[i:i+step].reshape(-1, 1) - xr
                    ydiff = y[i:i+step].reshape(-1, 1) - yr
                    amp_batch = amp[0:step].view(step, 1, 1)
                    amp = amp[step:]
                    
                else: # compute the remaining layers
                    step = n % step
                    xdiff = x[-step:].reshape(-1, 1) - xr
                    ydiff = y[-step:].reshape(-1, 1) - yr
                    amp_batch = amp.view(step, 1, 1)

                rd = -calrange + torch.sqrt(xdiff**2 + ydiff**2)
                t = 2 * (rd + calrange) / c # time taken for radar waves to travel (antenna -> point scatterer -> antenna) for a certain angle

                # fit times into bins t_min, t_min + ts, t_min + 2*ts, ..., t_max - ts, t_max
                t_bins = torch.round((t - t_min) / ts).type(torch.LongTensor).view(t.shape[0], 1, t.shape[-1]).to(device)
                t_bins = t_bins.repeat(1, nsinc_comps, 1)
                t_bins = t_bins + sinc_comps.reshape(-1, 1) # col 0 will contain time_bins for angle 0, where time_bins = [t-2, t-1, t, t+1, t+2]
                                
                row_bins = torch.transpose(t_bins, dim0=-2, dim1=-1).flatten() # start_dim=-2 tror den här inte ska med
                col_bins = torch.arange(nphi).repeat(step, nsinc_comps, 1).transpose(dim0=-2, dim1=-1).flatten() # start_dim=-2 samma här
                point_bins = torch.arange(np.max(col_bins.shape)) % step
                point_bins = point_bins.view(np.max(point_bins.shape)//step, step).transpose(dim0=-2, dim1=-1).flatten()

                M = torch.zeros((step, nt, nphi)).to(device).type(torch.complex64)

                # compute sinc components
                sincs = torch.sinc(BW * (t[:, None, :] - t_grid[t_bins])).transpose(dim0=-2, dim1=-1).flatten()

                # compute phase
                phases = torch.exp(-2j*np.pi*(t[:, None, :] - t_grid[t_bins])*fc).transpose(dim0=-2, dim1=-1).flatten()

                rd = rd[:, None, :].repeat(1, nsinc_comps, 1).transpose(dim0=-2, dim1=-1).flatten()

                M[point_bins, row_bins, col_bins] = sincs * phases * (calrange/(calrange + rd))**2 #* amp_batch.repeat(1, sincs.shape[0]//step, 1).flatten()

                rcs_batch = amp_batch * M
                rcs_batch = torch.sum(rcs_batch, dim=0)
                rcs = rcs + rcs_batch
        
        elif method == 2:
            for i in range(n):
                rd = -calrange + torch.sqrt((x[i] - xr)**2 + (y[i] - yr)**2)
                t = 2 * (rd + calrange) / c # time taken for radar waves to travel (antenna -> point scatterer -> antenna) for a certain angle

                # fit times into bins t_min, t_min + ts, t_min + 2*ts, ..., t_max - ts, t_max
                t_bins = torch.round((t - t_min) / ts).type(torch.LongTensor).to(device)
                t_bins = t_bins.repeat(nsinc_comps, 1)
                t_bins = t_bins + sinc_comps.reshape(-1, 1) # col 0 will contain time_bins for angle 0, where time_bins = [t-2, t-1, t, t+1, t+2]

                # compute rows and columns of non-zero components
                row_bins = torch.transpose(t_bins, dim0=-2, dim1=-1).flatten(start_dim=-2) 
                col_bins = torch.arange(nphi).repeat(nsinc_comps, 1).transpose(dim0=-2, dim1=-1).flatten()
    
                M = torch.zeros((nt, nphi)).to(device).type(torch.complex64)

                # compute sinc components
                sincs = torch.sinc(BW * (t - t_grid[t_bins]))
                
                # compute phases
                phase = torch.exp(-2j*np.pi*(t - t_grid[t_bins])*fc).transpose(dim0=-2, dim1=-1).flatten()

                rd = rd.repeat(nsinc_comps, 1).transpose(dim0=-2, dim1=-1).flatten()

                M[row_bins, col_bins] = sincs.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2) * phase * (calrange/(calrange + rd))**2
                
                rcs += amp[i] * M

        return rcs