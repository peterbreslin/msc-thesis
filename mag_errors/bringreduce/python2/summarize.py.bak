# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:49:42 2017

@author: talens
"""

import os
import glob

import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'
rcParams['axes.labelsize'] = 'x-large'
rcParams['image.interpolation'] = 'none'
rcParams['image.origin'] = 'lower'
rcParams['axes.titlesize'] = 'xx-large'

color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

from . import creds, astrometry, bringio

def send_mail(subject, message, images):  
    
    import smtplib
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText
    from email.MIMEImage import MIMEImage  

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = 'bring@strw.leidenuniv.nl'
    msg['To'] = ', '.join(creds.recipients)      
    
    text = MIMEText(message)
    msg.attach(text)
    
    for filename in images:
        
        fp = open(filename, 'rb')
        img = MIMEImage(fp.read())
        fp.close()

        img.add_header('Content-Disposition', 'attachment', filename=os.path.split(filename)[1])
        msg.attach(img)
            
    try:
        
        smtpserver = smtplib.SMTP("smtp.strw.leidenuniv.nl", 587)
        smtpserver.ehlo()
        smtpserver.starttls()
        smtpserver.login(creds.user, creds.pwd)    
        smtpserver.sendmail(creds.user, creds.recipients, msg.as_string())
        smtpserver.close()
        
    except:
        
        print 'The following email was not sent \n'
        print message     
        
    return

###############################################################################
### Functions to send a reduction e-mail.
###############################################################################

def reduction_summary(directory):

    head, tail = os.path.split(directory)
    head, tail = os.path.split(head)

    fastphot = os.path.join(directory, 'fast_'+tail+'.hdf5')    
    slowphot = os.path.join(directory, 'slow_'+tail+'.hdf5')

    subject = 'Reduction Summary {}'.format(tail)
    message = []
    images = []   

    # Create diagnostic figures.
    figname = os.path.join(directory, tail+'astro.png')
    
    try:
        nsols = fig_astrometry(fastphot, figname)
    except:
        nsols = 0
    else:
        images.append(figname)
    
    figname = os.path.join(directory, tail+'phot.png')
    
    try:
        nimages = fig_photometry(fastphot, slowphot, figname)
    except:
        nimages = 0
    else:
        images.append(figname)    
    
    # Create the text message.
    message.append('Hi,\n\n')
    message.append('{} science frames were processed.\n'.format(nimages))
    message.append('The astrometry was solved {} times.\n'.format(nsols))
    message.append('\nCheers,\nbRing')
    message = ''.join(message)
    
    send_mail(subject, message, images)
    
    return 

def fig_astrometry(fastphot, figname): 

    import datetime

    # Read the astrometry.
    with h5py.File(fastphot, 'r') as f:
        
        grp = f['header']
        siteinfo = dict()
        siteinfo['lon'] = grp.attrs['lon'] 
        siteinfo['lat'] = grp.attrs['lat']
        siteinfo['height'] = grp.attrs['alt']       
        
        grp = f['astrometry']
        
        lstseq = grp['lstseq'].value
        fstars = grp['fstars'].value
        ustars = grp['ustars'].value
        dr = grp['dr'].value
        
        ra0 = grp['ra0'].value
        ra1 = grp['ra1'].value
        ra2 = grp['ra2'].value
        ra3 = grp['ra3'].value
        ra4 = grp['ra4'].value
        
        dec0 = grp['dec0'].value
        dec1 = grp['dec1'].value
        dec2 = grp['dec2'].value
        dec3 = grp['dec3'].value
        dec4 = grp['dec4'].value
        
        grp = f['station']
        lstseq0 = grp['lstseq'].value
        lst = grp['lst'].value
        jd = grp['jd'].value
        year = grp['year'].value
        month = grp['month'].value
        day = grp['day'].value
        hour = grp['hour'].value
        mn = grp['min'].value
        sec = grp['sec'].value
     
    utc = np.array([datetime.datetime(year[i], month[i], day[i], hour[i], mn[i], sec[i]) for i in range(len(lst))])   
     
    # Normalize the julian date.
    offset = np.floor(jd[0]) 
    jd = jd - offset     
      
    idx = np.searchsorted(lstseq0, lstseq)
      
    # Convert to relative ha, dec.
    ha0 = np.mod(lst[idx]*15 - ra0, 360)     
    ha1 = np.mod(lst[idx]*15 - ra1, 360)    
    ha2 = np.mod(lst[idx]*15 - ra2, 360)    
    ha3 = np.mod(lst[idx]*15 - ra3, 360)    
    ha4 = np.mod(lst[idx]*15 - ra4, 360)
         
    ha0 = ha0 - np.mean(ha0)     
    ha1 = ha1 - np.mean(ha1)  
    ha2 = ha2 - np.mean(ha2)  
    ha3 = ha3 - np.mean(ha3)  
    ha4 = ha4 - np.mean(ha4)  
    
    dec0 = dec0 - np.mean(dec0)     
    dec1 = dec1 - np.mean(dec1) 
    dec2 = dec2 - np.mean(dec2) 
    dec3 = dec3 - np.mean(dec3) 
    dec4 = dec4 - np.mean(dec4)  
    
    fig = plt.figure(figsize=(14,9))
    
    gs = gridspec.GridSpec(3, 3)
    
    majorLocator = MultipleLocator(500)
    
    dt = datetime.timedelta(minutes=10)
    sunset = astrometry.last_sunset(siteinfo, utc[0]+dt)
    sunrise = astrometry.next_sunrise(siteinfo, utc[-1]-dt)   

    # Add stars used.
    ax = plt.subplot(gs[0,0:2])
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))  
    ax.yaxis.set_major_locator(majorLocator)
    plt.plot(utc[idx], fstars, 'k.')
    plt.plot(utc[idx], ustars, '.', c=(146./255,0,0))
    plt.xlim(sunset-dt, sunrise+dt)
    plt.axvline(sunset, c=(0./255, 109./255, 219./255), lw=2)
    plt.axvline(sunrise, c=(0./255, 109./255, 219./255), lw=2)
    plt.xlabel('Time [UTC]')
    plt.ylabel('# stars')
    
    # Add residual RMS. 
    ax = plt.subplot(gs[1,0:2])
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    plt.plot(utc[idx], dr, 'k.')
    plt.xlim(sunset-dt, sunrise+dt)
    plt.axvline(sunset, c=(0./255, 109./255, 219./255), lw=2)
    plt.axvline(sunrise, c=(0./255, 109./255, 219./255), lw=2)
    plt.ylim(0, 2)
    plt.xlabel('Time [UTC]')
    plt.ylabel('RMS [pix]')
    
    # Add change in position.
    plt.subplot(gs[2,0], aspect='equal')
    plt.annotate('(1002, 668)', (0, 1), xytext=(10, -10), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', size='x-large')
    plt.plot(ha0*60, dec0*60, 'k.')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("$\Delta$HA [']")
    plt.ylabel("$\Delta$Dec [']")
    
    plt.subplot(gs[2,1], aspect='equal')
    plt.annotate('(1002, 2004)', (0, 1), xytext=(10, -10), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', size='x-large')
    plt.plot(ha1*60, dec1*60, 'k.')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("$\Delta$HA [']")
    plt.ylabel("$\Delta$Dec [']")
    
    plt.subplot(gs[2,2], aspect='equal')
    plt.annotate('(2004, 1336)', (0, 1), xytext=(10, -10), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', size='x-large')
    plt.plot(ha2*60, dec2*60, 'k.')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("$\Delta$HA [']")
    plt.ylabel("$\Delta$Dec [']")
    
    plt.subplot(gs[1,2], aspect='equal')
    plt.annotate('(3006, 668)', (0, 1), xytext=(10, -10), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', size='x-large')
    plt.plot(ha3*60, dec3*60, 'k.')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("$\Delta$HA [']")
    plt.ylabel("$\Delta$Dec [']")
    
    plt.subplot(gs[0,2], aspect='equal')
    plt.annotate('(3006, 2004)', (0, 1), xytext=(10, -10), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', size='x-large')
    plt.plot(ha4*60, dec4*60, 'k.')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("$\Delta$HA [']")
    plt.ylabel("$\Delta$Dec [']")
    
    plt.tight_layout()
    
    plt.savefig(figname, dpi=100)
    plt.close()    
    
    return len(lstseq)
    
def fig_photometry(fastphot, slowphot, figname):
    
    import datetime      
        
    fig = plt.figure(figsize=(14,9))    
    
    ax = plt.subplot(111)    
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))    
    
    with h5py.File(fastphot, 'r') as f, h5py.File(slowphot, 'r') as g:
        
        grp = f['header']
        siteinfo = dict()
        siteinfo['lon'] = grp.attrs['lon'] 
        siteinfo['lat'] = grp.attrs['lat']
        siteinfo['height'] = grp.attrs['alt']               
        
        # Select stars in a small magnitude range.
        grp = f['stars']        
        ascc = grp['ascc'].value
        vmag = grp['vmag'].value
        
        mask = (vmag < 4.8) & (vmag > 4.75)
        ascc = ascc[mask]

        # Read the station fields of both files.
        grp = f['station']
        lstseq1 = grp['lstseq'].value
        jd1 = grp['jd'].value
        exptime1 = grp['exptime'].value
        
        year = grp['year'].value
        month = grp['month'].value
        day = grp['day'].value
        hour = grp['hour'].value
        mn = grp['min'].value
        sec = grp['sec'].value
     
        utc1 = np.array([datetime.datetime(year[i], month[i], day[i], hour[i], mn[i], sec[i]) for i in range(len(jd1))])         
        
        grp = g['station']
        lstseq2 = grp['lstseq'].value
        jd2 = grp['jdmid'].value
        exptime2 = grp['exptime'].value
        
        # Normalize the julian date.
        offset = np.floor(jd1[0])        
        jd1 = jd1 - offset
        jd2 = jd2 - offset      
        
        grp1 = f['lightcurves']
        grp2 = g['lightcurves']        
        
        lines = []
        for i in range(len(ascc)):
            
            # Add fast photometry.
            lc = grp1[ascc[i]].value            
            idx = np.searchsorted(lstseq1, lc['lstseq'])
            
            if np.mean(lc['flux0']/exptime1[idx]) > 30000:
                continue
            
            l, = ax.plot(utc1[idx], lc['flux0']/exptime1[idx], '.', label=ascc[i], c=color_sequence[i], zorder=1)
            lines.append(l)            
            
            # Add slow photometry.
            try:
                lc = grp2[ascc[i]]   
            except:
                continue
            else:
                idx = np.searchsorted(lstseq1, lc['lstseq'])
                idx2 = np.searchsorted(lstseq2, lc['lstseq'])
                ax.scatter(utc1[idx], lc['flux0']/exptime2[idx2], marker='o', c='w', zorder=2)
                
    dt = datetime.timedelta(minutes=10) 
    sunset = astrometry.last_sunset(siteinfo, utc1[0]+dt)
    sunrise = astrometry.next_sunrise(siteinfo, utc1[-1]-dt)  
                           
    plt.xlim(sunset-dt, sunrise+dt)
    plt.axvline(sunset, c=(0./255, 109./255, 219./255), lw=2)
    plt.axvline(sunrise, c=(0./255, 109./255, 219./255), lw=2)

    fig.legend(lines, ascc)    
    plt.xlabel('Time [UTC]')
    plt.ylabel('Flux [counts/sec]')    

    plt.tight_layout()

    plt.savefig(figname, dpi=100)
    plt.close()
    
    return len(lstseq1)
    
###############################################################################
### Functions to send a calibration e-mail.
###############################################################################

def calibration_summary(directory, astromaster):

    head, tail = os.path.split(directory)
    head, tail = os.path.split(head)

    subject = 'Calibration Summary {}'.format(tail)
    message = []
    images = [] 

    # Get the systematics file to summarize.
    sysfile = os.path.join(directory, 'sys0_vmag_'+tail+'.hdf5')

    head, tail = os.path.split(sysfile)
    tail = tail.rsplit('.')[0].rsplit('_')[-1]

    # Create diagnostic figures.
    figname = os.path.join(head, tail+'trans.png')
    try:
        fig_transmission(sysfile, astromaster, figname)
    except:
        pass
    else:
        images.append(figname)
    
    figname = os.path.join(head, tail+'ipx.png')
    try:
        fig_intrapix(sysfile, astromaster, figname)
    except:
        pass
    else:
        images.append(figname)
    
    figname = os.path.join(head, tail+'clouds.png')
    try:
        fig_clouds(sysfile, figname)
    except:
        pass
    else:
        images.append(figname)
    
    # Create the text message.
    message.append('Hi,\n\n')
    message.append('The daily calibration is finished.\n')
    message.append('\nCheers,\nbRing')
    message = ''.join(message)
    
    send_mail(subject, message, images)
    
    return 
    
def wcsgrid(wcspars):

    # Add lines of constant declination.
    ha = np.linspace(0, 360, 360)
    dec = np.linspace(-80, 80, 17)
    ha, dec = np.meshgrid(ha, dec)
    
    tmp = ha.shape
    
    ha, dec = ha.ravel(), dec.ravel()
    
    x, y = astrometry.world2wcs(wcspars, ha, dec, 0.)
    x, y = x.reshape(tmp), y.reshape(tmp)
    
    here = (x > -50) & (x < 4008+50) & (y > -50) & (y < 2672+50)
    x[~here] = np.nan
    y[~here] = np.nan
    
    plt.plot(x.T, y.T, c='k')
    
    # Add lines of constant hour angle.
    ha = np.linspace(0, 345, 24)
    dec = np.linspace(-80, 80, 160)
    ha, dec = np.meshgrid(ha, dec)
    
    tmp = ha.shape
    
    ha, dec = ha.ravel(), dec.ravel()
    
    x, y = astrometry.world2wcs(wcspars, ha, dec, 0.)
    x, y = x.reshape(tmp), y.reshape(tmp)
    
    here = (x > -50) & (x < 4008+50) & (y > -50) & (y < 2672+50)
    x[~here] = np.nan
    y[~here] = np.nan
    
    plt.plot(x, y, c='k')
    
    return

def plot_polar(grid, data, wcspars, **kwargs):
    
    data = data[1:-1,1:-1]    
    data = np.ma.array(data, mask=np.isnan(data))    
    
    ha, dec = grid.xedges, grid.yedges
    ha, dec = np.meshgrid(ha, dec)
    x, y = astrometry.world2wcs(wcspars, ha, dec, 0)
    
    im = plt.pcolormesh(x, y, data.T, **kwargs)
    wcsgrid(wcspars)

    return im
    
def fig_transmission(filename, astromaster, figname):
    
    # Read the transmission map.
    f = bringio.SysFile(filename)
    pg, nobs, trans = f.read_transmission()
    
    wcspars, polpars = bringio.read_astromaster(astromaster)    
    
    # Plot the transmission map.
    fig = plt.figure(figsize=(14,9))
    
    plt.suptitle('Transmission', size='xx-large')
    
    gs = gridspec.GridSpec(2, 2, width_ratios = [15,.5], height_ratios = [1,10])
    
    plt.subplot(gs[1,0], aspect='equal')
    
    vmin = np.nanpercentile(trans, .1)
    vmax = np.nanpercentile(trans, 99.9)
    im = plot_polar(pg, trans, wcspars, cmap=plt.cm.viridis, vmin=vmin, vmax=vmax)
    
    plt.xlim(0, 4008)
    plt.ylim(0, 2672)
    
    cax = plt.subplot(gs[1,1])
    cb = plt.colorbar(im, cax = cax)
    cb.ax.invert_yaxis()
    cb.set_label('Magnitude')
    
    plt.tight_layout()
    
    plt.savefig(figname, dpi=100)
    plt.close()
    
    return
    
def fig_intrapix(filename, astromaster, figname):
    
    # Read the intrapixel amplitudes.
    f = bringio.SysFile(filename) 
    pg, nobs, amplitudes = f.read_intrapix()     
    
    wcspars, polpars = bringio.read_astromaster(astromaster)        
    
    # Plot the amplitudes.
    fig = plt.figure(figsize=(16, 10))
    
    plt.suptitle('Intrapixel Amplitudes', size='xx-large')
    
    gs = gridspec.GridSpec(3, 3, width_ratios = [15,15,.5], height_ratios = [1,10,10])
    
    plt.subplot(gs[1,0], aspect='equal')
    plt.title(r'$\sin(2\pi x)$')
   
    im = plot_polar(pg, amplitudes[:,:,0], wcspars, cmap=plt.cm.viridis, vmin=-.05, vmax=.05)
   
    plt.xlim(0, 4008)
    plt.ylim(0, 2672)
    
    plt.subplot(gs[1,1], aspect='equal')
    plt.title(r'$\cos(2\pi x)$')
    
    im = plot_polar(pg, amplitudes[:,:,1], wcspars, cmap=plt.cm.viridis, vmin=-.05, vmax=.05)   
    
    plt.xlim(0, 4008)
    plt.ylim(0, 2672)
    
    plt.subplot(gs[2,0], aspect='equal')
    plt.title(r'$\sin(2\pi y)$')
    
    im = plot_polar(pg, amplitudes[:,:,2], wcspars, cmap=plt.cm.viridis, vmin=-.05, vmax=.05)  
    
    plt.xlim(0, 4008)
    plt.ylim(0, 2672)
    
    plt.subplot(gs[2,1], aspect='equal')
    plt.title(r'$\cos(2\pi y)$')
    
    im = plot_polar(pg, amplitudes[:,:,3], wcspars, cmap=plt.cm.viridis, vmin=-.05, vmax=.05)   
    
    plt.xlim(0, 4008)
    plt.ylim(0, 2672)
    
    cax = plt.subplot(gs[1:,2])
    cb = plt.colorbar(im, cax = cax)
    cb.set_label('Amplitude')
    
    plt.tight_layout()
    
    plt.savefig(figname, dpi=100)
    plt.close()  
    
    return
    
def fig_clouds(filename, figname):
    
    # Read the intrapixel amplitudes.
    f = bringio.SysFile(filename)     
    hg, nobs, clouds, sigma, lstmin, lstmax = f.read_clouds() 
    
    # Plot the clouds.
    fig = plt.figure(figsize=(10, 16))
    
    plt.suptitle('Clouds', size='xx-large')
    
    gs = gridspec.GridSpec(2, 2, width_ratios = [15,.5], height_ratios = [1,20])
    
    plt.subplot(gs[1,0], xticks=[], yticks=[])
    
    mask = np.isfinite(clouds)
    idx1, = np.where(np.any(mask, axis=1))
    idx2, = np.where(np.any(mask, axis=0))
    clouds = clouds[idx1][:,idx2]    
    im = plt.imshow(clouds.T, interpolation='None', aspect='auto', cmap=plt.cm.viridis, vmin=-.5, vmax=.5)
    
    idx, = np.where(np.diff(idx2) > 1)
    for i in idx:
        plt.axhline(i+1, xmax=.1, c='k', lw=2)
    
    plt.ylabel('Time')
    plt.xlabel('Sky Patch')
    
    cax = plt.subplot(gs[1,1])
    cb = plt.colorbar(im, cax = cax)
    cb.ax.invert_yaxis()
    cb.set_label('Magnitude')
    
    plt.tight_layout()
    
    plt.savefig(figname, dpi=100)
    plt.close()
    
    return
    
def main():

    return    
    
if __name__ == '__main__':
    main()
