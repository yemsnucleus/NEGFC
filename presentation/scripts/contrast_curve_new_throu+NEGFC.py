#!/usr/bin/python
# -*- coding: latin-1 -*-
from matplotlib.pyplot import *
from pylab import *
import numpy as np
import copy
import vip_hci as vip
import scipy
from vip_hci.negfc import *
from vip_hci.preproc import *
from vip_hci.var import *
from vip_hci.phot import *
from vip_hci.pca import *
import pandas as pd
from scipy.optimize import leastsq
from scipy.stats import t
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma, sigma_clipped_stats
from scipy.optimize import minimize
from skimage.draw import circle
plots = vip.var.pp_subplots
plt.interactive(False)


def chisquare_mod ( modelParameters,sourcex, sourcey, frame, ang, plsc, psf_norma, fwhm, fmerit,
              svd_mode='lapack'):
        try:
            r, theta, flux = modelParameters
        except TypeError:
            print('paraVector must be a tuple, {} was given'.format(type(modelParameters)))
        frame_negfc = inject_fcs_cube_mod(frame, psf_norma, ang, -flux, pixel, r, theta,n_branches=1)
        centy_fr, centx_fr = frame_center(frame_negfc)
        posy = r * np.sin(np.deg2rad(theta)-np.deg2rad(ang)) + centy_fr
        posx = r * np.cos(np.deg2rad(theta)-np.deg2rad(ang)) + centx_fr
        indices = circle(posy, posx, radius=2*fwhm)
        yy, xx = indices
        values = frame_negfc[yy, xx].ravel()    
        # Function of merit
        if fmerit == 'sum':
            values = np.abs(values)
            chi2 = np.sum(values[values > 0])
            N = len(values[values > 0])
            return chi2 / (N-3)
        elif fmerit == 'stddev':
            return np.std(values[values != 0])
        else:
            raise RuntimeError('`fmerit` choice not recognized')


    

def inject_fcs_cube_mod(array, psf_template, angle_list, flevel, plsc, rad_dists, 
                    theta,n_branches=1, imlib='opencv', verbose=True):
    
    
    ceny, cenx = frame_center(array)
    size_fc = psf_template.shape[0]
    fc_fr = np.zeros_like(array, dtype=np.float64)  # TODO: why float64?
    w = int(np.floor(size_fc/2.))
    # fcomp in the center of a zeros frame
    fc_fr[int(ceny-w):int(ceny+w+1), int(cenx-w):int(cenx+w+1)] = psf_template

    array_out = np.zeros_like(array)
    tmp = np.zeros_like(array)
    for branch in range(n_branches):
        ang = (branch * 2 * np.pi / n_branches) + np.deg2rad(theta)
        rad = rad_dists
        y = rad * np.sin(ang - np.deg2rad(angle_list))
        x = rad * np.cos(ang - np.deg2rad(angle_list))
        tmp += frame_shift(fc_fr, y, x, imlib=imlib)*flevel
    array_out = array + tmp
        
    return array_out   
    




#iniziamo con aprire il cubo (center_im), la psf (median_unsat) e gli angoli di rotazione (rotnth)
print('Digita il nome del tuo sistema')
name = raw_input()
print('Hai un cubo sorted?y/n')
answ=raw_input()
if answ=='y':
    cube_in='./%s/center_im_nosec.fits'%name    #rinomina il cubo
    cube1,hdr= vip.fits.open_fits(cube_in,header=True) #salva i valori del file nella variabile cube1 (occhio che inver8te le variabili, controlla con .shape per vederne le dimensioni)
    rot='./%s/rotnth_sorted_binned.fits'%name  #rinomina gli angoli
    rot1,hdr= vip.fits.open_fits(rot,header=True) #salva i valori del file nella variabile rot1
else :
    cube_in='./%s/center_im.fits'%name    #rinomina il cubo
    cube1,hdr= vip.fits.open_fits(cube_in,header=True) #salva i valori del file nella variabile cube1 (occhio che inver8te le variabili, controlla con .shape per vederne le dimensioni)
    rot='./%s/rotnth.fits'%name  #rinomina gli angoli
    rot1,hdr= vip.fits.open_fits(rot,header=True) #salva i valori del file nella variabile rot1
print('Decidi che lunghezza vuoi (H2=0  H3=1) e se vuoi la psf iniziale (0)o finale (1)')
lam=input()
w=input()

#cube=cube1[lam,:,:,:]  
cube=cube1[:,:,:]
cube_shifted=np.zeros((len(cube),len(cube[0,:,0]),len(cube[0,:,0])))
for i in range (len(cube)):
       cube_shifted[i,:,:]=frame_shift(cube[i,:,:], -1, -1)  
#vip.fits.write_fits('./%s/cube_shifted.fits'%name, cube_shifted) 
ycube_center,xcube_center=frame_center(cube[0,:,:])
if len(cube[1,:,1])%2==0:
       ycen=ycube_center-0.5
       xcen=xcube_center-0.5
       newdim=len(cube[1,:,1])-1
       cube_crop=cube_crop_frames(cube_shifted,newdim,xy=[int(ycen),int(xcen)], force=True)    #tagliamo l'immagine'''
       #vip.fits.write_fits('./%s/cube_crop.fits'%name,cube_crop) 
else :
       cube_crop=cube

ycube_center,xcube_center=frame_center(cube_crop[0,:,:])
dim1=len(cube_crop)
dim2=len(cube_crop[1,:,1])
r=10
ones=np.ones((dim2,dim2))
maschera_invert=get_circle(ones,radius=r)
maschera=ones-maschera_invert
for i in range (len(cube)):
        cube_crop[i,:,:]=cube_crop[i,:,:]*maschera
 

#se vuoi considerare solo una parte dei frame e quindi degli angoli (vedi  sotto)
        
#cube_crop2=np.zeros((int(dim1/3),dim2,dim2))
#for i in range (int(dim1/3)):
#        cube_crop2[i]=cube_crop[i+2*int(dim1)/3]
#cube_crop=cube_crop2


psf='./%s/median_unsat.fits'%name  #rinomina la psf
psf1,hdr= vip.fits.open_fits(psf,header=True) #salva i valori del file nella variabile psf1 (occhio che inverte le variabili, controlla con .shape per vederne le dimensioni)
rot=-rot1  #usiamo la convenzione con gli angoli positivi (rot avrebbe angoli negativi


#rot2=np.zeros((int(dim1/3)))
#for i in range (int(dim1/3)):
#        rot2[i]=rot[i+2*int(dim1)/3]
#rot=rot2


psf2=psf1[lam,:,:-1,:-1]   #faccio una nuova psf di dimensione dispari (63x63) e a lambda fissata (vip vuole sempre cubi dispari per lavorare)    #psf H2
#psf2=psf1[lam,:-1,:-1]   #quando non hai psf iniziale e finale usa questa riga
#psf2=psf1[lam,:,:]        #se la psf è gia dispari
fit = vip.var.fit_2dgaussian(psf2[w,:,:], crop=True, cropsize=30, debug=True, full_output=True) #da qua leggi i valori di FWHM per x e y della psf e 
#fit = vip.var.fit_2dgaussian(psf2, crop=True, cropsize=30, debug=True, full_output=True)  #quando non hai psf iniziale e finale usa questa riga
fwhm_sphere = np.mean([fit.fwhm_y,fit.fwhm_x])  #ne fai una media
pixel=0.01225
y_cent, x_cent = frame_center(psf2[w])  #cerchiamo il nuovo centro dell'immagine
#y_cent, x_cent = frame_center(psf2)     #quando non hai psf iniziale e finale usa questa riga
y_c=int(y_cent)
x_c=int(x_cent)
psf_center, y0_sh, x0_sh = cube_recenter_2dfit(psf2, (y_c, x_c), fwhm_sphere,model='gauss',nproc=1, subi_size=7, negative=False,
                                            full_output=True, debug=False)  
#psf3=np.zeros((2,len(psf2),len(psf2)))      #quando non hai psf iniziale e finale usa queste righe
#psf3[0,:,:]=psf2
#psf3[1,:,:]=psf2
#psf_center, y0_sh, x0_sh = cube_recenter_2dfit(psf3, (y_c, x_c), fwhm_sphere,model='gauss',
#                                             nproc=1, subi_size=5, negative=False,
#                                             full_output=True, debug=False)


psf_center_in=psf_center[w,:,:]
psf_norm, fwhm_flux_in=normalize_psf(psf_center_in, fwhm=fwhm_sphere, size=None, threshold=None, mask_core=None,
             full_output=True, verbose=True)  #psf normalizzata ad 1

#vip.fits.write_fits('./%s/median_unsat_norm.fits'%name, psf_norm)    #per salvare l'immagine


#generiamo un'immagine con la tecnica adi
fr_adi = vip.madi.adi(cube_crop, rot, mode='fullfr')
#contrast_curve(cube_crop,rot,psf_norm, fwhm_sphere, pixel, fwhm_flux_in[0],vip.madi.adi)
vip.phot.detection(fr_adi, psf_norm, debug=False, mode='log', snr_thresh=7, 
                           bkg_sigma=3, matched_filter=False)
#ds9 = vip.Ds9Window()
#ds9.display(fr_adi)
print('1 frame by frame')
print('2 full cube')
rout=input()
if rout==1:
    print('Inserisci le coordinate del pianeta (ricorda che ds9 da un pixel di piu)')
    sourcex=input()
    sourcey=input()
    print('Inserisci un guess per il flusso')
    f0=input()
    print('Decidi se vuoi somma (sum) o standard deviation (stddev) e che FWHM vuoi')
    mode=raw_input()
    nfwhm=input()
    fwhma=nfwhm*fwhm_sphere
    cube_emp=np.zeros((int(dim1), dim2,dim2))
    f_0_comp=np.zeros((dim1))
    r_0_comp=np.zeros((dim1))
    theta_0_comp=np.zeros((dim1))
    for i in range (int(dim1)):
        x=sourcex-xcube_center
        y=sourcey-ycube_center
        cnx1=int(np.cos(-rot[i]*np.pi/180.)*x-np.sin(-rot[i]*np.pi/180.)*y+xcube_center)
        cny1=int(np.sin(-rot[i]*np.pi/180.)*x+np.cos(-rot[i]*np.pi/180.)*y+ycube_center)
        frame=cube_crop[i,:,:]
        r0= np.sqrt(x**2+y**2)
        theta0 = np.mod(np.arctan2(y,x)/np.pi*180,360)
        p=(r0,theta0,f0)       
        solu = minimize(chisquare_mod, p, args=(sourcex, sourcey, frame, rot[i], pixel, psf_norm, fwhma, mode),
                    method = 'Nelder-Mead')
        r_0, theta_0, f_0 = solu.x
        f_0_comp[i]=f_0
        r_0_comp[i]=r_0
        theta_0_comp[i]=theta_0
        print(i,r_0,theta_0, f_0)
        frame_emp=inject_fcs_cube_mod(frame, psf_norm, rot[i], -f_0, pixel, r_0, theta_0, n_branches=1)
        cube_emp[i,:,:]=frame_emp

else :
    print('Inserisci le coordinate del pianeta (ricorda che ds9 da un pixel di più)')
    sourcex=input()
    sourcey=input()
    print('1 se hai già i valori precisi')
    print('2 se vuoi una stima precisa')
    choice=input()
    if choice==1:
        print('r_0= ')
        r_0=input()
        print('theta_0=')
        theta_0=input()
        print('f_0=')
        f_0=input()
        print('Somma (sum) o standard deviation (stddev)')
        mode=raw_input()
        print('Numero di FWHM')
        nfwhm=input()
        fwhm=nfwhm*fwhm_sphere
        source_xy =[(sourcex,sourcey)]
        flx_min=f_0-5
        flx_max=f_0+5
        plpar = [(r_0, theta_0, f_0)]
        cube_emp = cube_planet_free(plpar, cube_crop, rot, psf_norm, pixel)
    else:
        print('In base al flusso trovato per il pianeta, dai un limite sup e inf')
        flx_min=input()
        flx_max=input()
        print('Decidi se vuoi somma (sum) o standard deviation (stddev) e quante FWHM vuoi')
        mode=raw_input()
        nfwhm=input()
        fwhm=nfwhm*fwhm_sphere
        source_xy =[(sourcex,sourcey)]   
        r0, theta0, f0 = firstguess(cube_crop, rot, psf_norm, annulus_width=1, aperture_radius=1,ncomp=1,plsc=pixel, fmerit=mode,planets_xy_coord=source_xy, fwhm=fwhm,f_range=np.linspace(flx_min,flx_max,10))
        print(r0, theta0, f0)
        r_0=float(r0)
        theta_0=float(theta0)
        f_0=float(f0)
        plpar = [(r_0, theta_0, f_0)]   #parametri del pianeta
        cube_emp = cube_planet_free(plpar, cube_crop, rot, psf_norm, pixel)  #planet subtraction


plcny=sourcey
plcnx=sourcex
fr_adi_res = vip.madi.adi(cube_emp, rot, mode='fullfr')
#fr_adi_res = vip.fits.open_fits('%s/fr_adi_res_RDI.fits'%name)
vip.fits.write_fits('./%s/fr_adi_res.fits'%name,fr_adi_res)
offset=9
if rout==1:
    r_0=np.mean(r_0_comp)
    theta_0=np.mean(theta_0_comp)
    f_0=np.mean(f_0_comp)

plpar = [(r_0, theta_0, f_0)]   #parametri del pianeta

if 0<theta_0<=90 or 270<theta_0<=360:
    xp=511.+r_0/np.sqrt(1+(np.tan(theta_0*np.pi/180.))**2)
    yp=511.+r_0*np.tan(theta_0*np.pi/180.)/np.sqrt(1+(np.tan(theta_0*np.pi/180.))**2)
else :
    xp=511.-r_0/np.sqrt(1+(np.tan(theta_0*np.pi/180.))**2)
    yp=511.-r_0*np.tan(theta_0*np.pi/180.)/np.sqrt(1+(np.tan(theta_0*np.pi/180.))**2)

frame_square=get_square(fr_adi_res,offset*2+1,yp,xp)
rad_dist=[]
noise=[]
for i in range(int(round(fwhm_sphere)),int(offset+2),int(round(fwhm_sphere))):
    annulus=get_annulus(frame_square,i,round(fwhm_sphere))
    stdev=np.std(annulus[np.where(annulus !=0.)])
    rad_dist.append(i)
    noise.append(stdev)


new_psf_size = 3 * int(round(fwhm_sphere))
if new_psf_size % 2 == 0:
    new_psf_size += 1
if cube_crop.ndim == 3:
    psf_norm_crop = normalize_psf(psf_norm, fwhm=fwhm_sphere,size=min(new_psf_size,psf_norm.shape[1]))
    

xfc=[]
yfc=[]
ffc=[]
i=0
for r in range(int(round(fwhm_sphere)),int(np.max(rad_dist)),int(round(fwhm_sphere))):
    for theta in range(0,360,int(np.arctan(round(fwhm_sphere)/r)*180/np.pi)):
        if 0<=theta<=90:
            x=r/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
            y=r*np.tan(theta*np.pi/180.)/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
        elif 90<theta<=180:
            x=-r/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
            y=-r*np.tan(theta*np.pi/180.)/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
        elif 180<theta<=270:
            x=-r/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
            y=-r*np.tan(theta*np.pi/180.)/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
        else :
            x=r/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
            y=r*np.tan(theta*np.pi/180.)/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)


        flux=noise[i]*10.
        xfc.append(x)
        yfc.append(y)
        ffc.append(flux)
    i=i+1
xfc=np.asarray(xfc)
yfc=np.asarray(yfc)
ffc=np.asarray(ffc)


throughput=[]
cube_crop2=np.zeros((len(cube_crop),len(cube_crop[0,:,0]),len(cube_crop[0,:,0])))
cube_crop3=np.ones((len(cube_crop),len(cube_crop[0,:,0]),len(cube_crop[0,:,0])))*10**(-6)
cube_planet_fc=np.zeros((len(cube_crop),len(cube_crop[0,:,0]),len(cube_crop[0,:,0])))
cube_emp_pl=np.zeros((len(cube_crop),len(cube_crop[0,:,0]),len(cube_crop[0,:,0])))

for k in range(len(xfc)):
    cube_crop2=cube_crop
    rfc=np.sqrt((xfc[k]+xp-511.)**2+(yfc[k]+yp-511.)**2)
    if (xfc[k]+xp)>=511. and (yfc[k]+yp)>=511.:
        thetafc=np.arctan((yfc[k]+yp-511.)/(xfc[k]+xp-511.))*180/np.pi
    elif ((xfc[k]+xp)<511. and (yfc[k]+yp)>511.) or ((xfc[k]+xp)<511. and (yfc[k]+yp)<511.):
        thetafc=np.arctan((yfc[k]+yp-511.)/(xfc[k]+xp-511.))*180/np.pi+180
    else:
        thetafc=np.arctan((yfc[k]+yp-511.)/(xfc[k]+xp-511.))*180/np.pi+360
    print(xfc[k],yfc[k],rfc,thetafc)

    cube_planet_fc=cube_inject_companions(cube_crop2,psf_norm_crop,rot,ffc[k],pixel,rfc,theta=thetafc)

    if rout==1:
        for i in range (int(dim1)):
            x=sourcex-xcube_center
            y=sourcey-ycube_center
            cnx1=int(np.cos(-rot[i]*np.pi/180.)*x-np.sin(-rot[i]*np.pi/180.)*y+xcube_center)
            cny1=int(np.sin(-rot[i]*np.pi/180.)*x+np.cos(-rot[i]*np.pi/180.)*y+ycube_center)
            frame=cube_planet_fc[i,:,:]
            r0= np.sqrt(x**2+y**2)
            theta0 = np.mod(np.arctan2(y,x)/np.pi*180,360)
            p=(r0,theta0,f0)       
            solu = minimize(chisquare_mod, p, args=(sourcex, sourcey, frame, rot[i], pixel, psf_norm, fwhma, mode),
                    method = 'Nelder-Mead')
            r_0, theta_0, f_0 = solu.x
            print(i,r_0,theta_0, f_0)
            frame_emp=inject_fcs_cube_mod(frame, psf_norm, rot[i], -f_0, pixel, r_0, theta_0, n_branches=1)
            cube_emp_pl[i,:,:]=frame_emp

    else :
        r_0, theta_0, f_0 = firstguess(cube_planet_fc, rot, psf_norm, annulus_width=1, aperture_radius=1,ncomp=1,plsc=pixel, fmerit=mode,planets_xy_coord=source_xy, fwhm=fwhm,f_range=np.linspace(flx_min,flx_max,10))
        plpar = [(r_0, theta_0, f_0)]
        cube_emp_pl=cube_planet_free(plpar, cube_planet_fc, rot, psf_norm, pixel, imlib='opencv',interpolation='lanczos4')
    
    fr_adi_fc=vip.madi.adi(cube_emp_pl, rot, mode='fullfr') 
    

    cube_planet_fc_map=cube_inject_companions(cube_crop3,psf_norm_crop,rot,ffc[k],pixel,rfc,theta=thetafc)
    fr_adi_map=vip.madi.adi(cube_planet_fc_map, rot, mode='fullfr') 
    

    injected_flux = aperture_flux(fr_adi_map, [(yfc[k]+yp)], [(xfc[k]+xp)], fwhm_sphere,
                                              ap_factor=1, mean=False)
    recovered_flux = aperture_flux((fr_adi_fc-fr_adi_res), [(yfc[k]+yp)],
                                               [(xfc[k]+xp)], fwhm_sphere, ap_factor=1,
                                               mean=False)
    thruput = recovered_flux[0] / injected_flux[0]
    throughput.append([xfc[k],yfc[k],thruput])

throughput=np.asarray(throughput)
rad_dist=np.asarray(rad_dist)
i=0
j=0
throughput_sum=0
throughput_mean=[]
somma=0
while i<=len(throughput):
    if i==len(throughput):
        throughput_mean.append([throughput_sum/somma,rad_dist[j]])
    else :
        r=np.sqrt((throughput[i,0])**2+(throughput[i,1])**2)
        if r<=rad_dist[j]:
            throughput_sum=throughput_sum+throughput[i,2]
            somma=somma+1
        else:
            throughput_mean.append([throughput_sum/somma,rad_dist[j]])
            throughput_sum=0
            somma=0
            throughput_sum=throughput_sum+throughput[i,2]
            somma=somma+1
            j=j+1

    i=i+1

throughput_mean=np.asarray(throughput_mean)
res=fr_adi_res[(plcny-offset):(plcny+offset),(plcnx-offset):(plcnx+offset)]
r=fwhm_sphere/2.
out_file = open("./%s/%s_contrastcurves_corr_full_trp_NEGFC.txt"%(name,name),"w")
dist=[]
cont=[]

j=0
while r<offset:
    ann=get_annulus(res,r,fwhm_sphere/2.)
    ann_abs=np.abs(ann)
    mean=np.mean(ann_abs[ann_abs != 0])
    stddev=np.std(ann_abs[ann_abs != 0])
    contrast=5*stddev/np.max(psf_center_in)
    tcorr=t.ppf(contrast,2*np.pi*r-1)
    tcorr=-tcorr
    contrastcorr=(tcorr*stddev*np.sqrt(1+1/(2*np.pi*r))+mean)/np.max(psf_center_in)
    if (r<=throughput_mean[j,1]) or ((j+1)>=len(throughput_mean[:,0])):
        contrastcorr_trp=contrastcorr*1/throughput_mean[j,0]
        print(r,throughput_mean[j,0],throughput_mean[j,1])
    else :
        j=j+1
        contrastcorr_trp=contrastcorr*1/throughput_mean[j,0]
        print(r,throughput_mean[j,0],throughput_mean[j,1])
    out_file.write("%s,"%(str(r)))
    out_file.write("%s\n"%(str(contrastcorr_trp)))
    dist.append(r)
    cont.append(contrastcorr_trp)
    r=r+1

out_file.close()

