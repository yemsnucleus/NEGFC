# Negative Fake Companion Technique (NEGFC) pipeline

**Neg**ative **F**ake **C**ompanion (NegFC) is a direct imagining technique used to subtract the contribution of a companion in raw data.
<p align="center">
<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/cube.gif?raw=true" 
     width="500" 
     height="300" />
</p>
<p align="center">
<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/final.png?raw=true" 
     width="400" 
     height="300" />
</p>

From an educational perspective, some functions, such as `fit_2dgaussian` and  `cube_recenter_2dfit`, were re-written from scratch to make explicit the steps behind the blackbox. We followed the same logic as the official implementations (see Section [Used Functions](#used-functions) )

## Pipeline Description
### Preprocessing
1. Crop and shift a cube (even dimensions)
2. Find the center of the PSF
3. Cut the PSF focusing on the star 

<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/pipeline_1.png?raw=true" 
     width="=300" 
     height="200" />
     
4. Fit a 2-dimensional gaussian to the PSF

<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/pipeline_2.png?raw=true" 
     width="=300" 
     height="200" />
     
5. From the gaussian model we obtain the center of the star (mean) and FWHM (std)
6. We shift the center to the original dimensions (since we cutted it on step 3)
7. Average the x/y FWHM assuming an sphere
8. Using a reference PSF frame (FWHM sphere and x/y center), we center the sequence of PSF (cube). For each frame:
    1. we cut a sub-image using the center of the reference PSF
    2. fit a gaussian starting with the fwhm of the sphere
    3. with the new center coordinates, we shift the frame to be aligned with the reference PSF
9. Normalize a PSF to have flux in 1xFWHM aperture equal to one.

<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/pipeline_3.png?raw=true" 
     width="=300" 
     height="200" />

At the end of this step, we got a normalized reference PSF, FWHM aperture, and FWHM

### Detection
1.- For each frame within the cube we adjust a PCA
... in progress...

## Used Functions

- [`normalize_psf`](https://vip.readthedocs.io/en/latest/vip_hci.fm.html#vip_hci.fm.fakecomp.normalize_psf)
- [`fit_2dgaussian`](https://vip.readthedocs.io/en/latest/vip_hci.var.html?highlight=fit_2dgaussian#vip_hci.var.fit_2d.fit_2dgaussian)
- [`cube_recenter_2dfit`](https://vip.readthedocs.io/en/latest/vip_hci.preproc.html?highlight=cube_recenter_2dfit#vip_hci.preproc.recentering.cube_recenter_2dfit)
- [`get_square`](https://vip.readthedocs.io/en/latest/vip_hci.var.html?highlight=get_square#vip_hci.var.shapes.get_square)
- [`Gaussian2D`](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Gaussian2D.html)
- [`frame_shift`](https://vip.readthedocs.io/en/latest/vip_hci.preproc.html?highlight=frame_shift#vip_hci.preproc.recentering.frame_shift)
- [`pca`](https://vip.readthedocs.io/en/latest/_modules/vip_hci/psfsub/pca_fullfr.html#pca)
	+ [`prepare_matrix`](https://vip.readthedocs.io/en/latest/vip_hci.var.html?highlight=prepare_matrix#vip_hci.var.shapes.prepare_matrix)
	+ [`pca_grid`](https://vip.readthedocs.io/en/latest/vip_hci.psfsub.html?highlight=pca_grid#vip_hci.psfsub.utils_pca.pca_grid)
	+ [`svd_wrapper`](https://vip.readthedocs.io/en/latest/_modules/vip_hci/psfsub/svd.html?highlight=svd_wrapper)
	+ [`cube_derotate`](https://vip.readthedocs.io/en/latest/_modules/vip_hci/preproc/derotation.html#cube_derotate)
	+ [`frame_rotate`](https://vip.readthedocs.io/en/latest/vip_hci.preproc.html?highlight=vip_hci.preproc.frame_rotate%60#vip_hci.preproc.derotation.frame_rotate)
- [`sigma_clipped_stats`](https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html)
	+ [`sigma_clip`](https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipping.SigmaClip.html#astropy.stats.SigmaClip)
- [`peak_local_max`](https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/feature/peak.py#L119-L326)
- [`snr`](https://vip.readthedocs.io/en/latest/_modules/vip_hci/metrics/snr_source.html#snr)

## To-do
- ~cosmetics~
- ~moon detection~
- visualize residuals
- estimate parameters of moons
