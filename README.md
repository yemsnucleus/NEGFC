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

## Requirements
The following packages were used to create and test the pipeline. It is highly recommended to have the same versions. However, most of the python packages are compatible with newer versions.

- `Python==3.8.16`
- `vip-hci==1.3.6`
- `tqdm`
- `pandas`
- `numpy`
- `astropy`
- `skimage`
- `scipy==1.10`
- `matplotlib`

## Running the script
To run the script with default parameters:
```
python main.py
```

## Directory tree 
```
 📂 NEGFC: Root directory
│   📜 .gitignore: Files to ignore when pushing on GitHub
│   📜 README.md: Markdown readme (what you are reading now)    
│
└─── 📂 data: Images folder
│   │
│   └─── 📂 HCI
│       │   📜 center_im.fits: cube of frames
│       │   📜 median_unsat.fits: PSFs
│       │   📜 rotnth.fits: Rotational angles
│   
└─── 📂 figures: Figures used in the presentation and the markdown readme
│ 
└─── 📜 detection.py: Function to recognize companion candidates 
│ 
└─── 📜 loss.py: Defines the loss function and optimization logic
│ 
└─── 📜 pca.py: Perform principal component analysis to calculate residuals
│ 
└─── 📜 plottools.py: Useful function to display partial outputs
```


## Pipeline Description
[Look at the slides on this link!](https://docs.google.com/presentation/d/1-jH5DRscOK33Ga0WSnYmIGjU-SO7Sh1tINMAqQLfwWI/edit?usp=sharing)

The pipeline consists on 4 steps
- Preprocesing: We normalize the PSF image. This will be used as a mold to create the fake companion
<p align="center">
<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/pipeline_preprocessing.png?raw=true" 
     width="600" 
     height="380" />
</p>

- Residuals calculation: In this step, we use PCA to remove star gain and other invariant patterns.
<p align="center">
<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/pipeline_residual.png?raw=true" 
     width="600" 
     height="380" />
</p>

- Detection: Identification of potential companion. Here we assume background thresholds and gaussian distributed companion
<p align="center">
<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/pipeline_detection.png?raw=true" 
     width="600" 
     height="380" />
</p>

- Optimization: We adjust a normalized PSF to the imprecise coordinates and flux obtained in the Detection step. Here try to reproduce the distribution of the companion by minimizing a chisquare loss. 
<p align="center">
<img src="https://github.com/yemsnucleus/NEGFC/blob/main/figures/pipeline_optimize.png?raw=true" 
     width="600" 
     height="380" />
</p>


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

