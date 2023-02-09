# Negative Fake Companion Technique (NEGFC) pipeline

**Neg**ative **F**ake **C**ompanion (NegFC) is a direct imagining technique used to subtract the contribution of a companion in raw data.

From an educational perspective, some functions, such as `fit_2dgaussian` and  `cube_recenter_2dfit`, were re-written from scratch to make explicit the steps behind the blackbox. We followed the same logic as the official implementations (see Section [Used Functions](#used-functions) )


## Pipeline Description
In progress...

## Used Functions

- [`normalize_psf`](https://vip.readthedocs.io/en/latest/vip_hci.fm.html#vip_hci.fm.fakecomp.normalize_psf)
- [`fit_2dgaussian`](https://vip.readthedocs.io/en/latest/vip_hci.var.html?highlight=fit_2dgaussian#vip_hci.var.fit_2d.fit_2dgaussian)
- [`cube_recenter_2dfit`](https://vip.readthedocs.io/en/latest/vip_hci.preproc.html?highlight=cube_recenter_2dfit#vip_hci.preproc.recentering.cube_recenter_2dfit)
- [`get_square`](https://vip.readthedocs.io/en/latest/vip_hci.var.html?highlight=get_square#vip_hci.var.shapes.get_square)
- [`Gaussian2D`](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Gaussian2D.html)
- [`frame_shift`](https://vip.readthedocs.io/en/latest/vip_hci.preproc.html?highlight=frame_shift#vip_hci.preproc.recentering.frame_shift)

## To-do
- cosmetics
