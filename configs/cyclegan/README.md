# CycleGAN: Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks

## Introduction

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html">CycleGAN (ICCV'2017)</a></summary>

```bibtex
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017},
  url={https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html},
}
```
</details>

## Results and Models
<div align="center">
  <b> Results from CycleGAN trained by MMGeneration</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/114303527-108ed200-9b01-11eb-978c-274392e4d8e0.PNG" width="800"/>
</div>

We use `FID` and `IS` metrics to evaluate the generation performance of CycleGAN.<sup>1</sup>

| Models |      Dataset      |   FID    |  IS   |                                                                  Config                                                                  |                                                                                                                 Download                                                                                                                  |
| :----: | :---------------: | :------: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Ours  |      facades      | 124.8033 | 1.792 |      [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_resnet_in_facades_b1x1_80k.py)       |             [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_80k_facades_20210902_165905-5e2c0876.pth?versionId=CAEQMhiBgICA5rCs3RciIDNmNDdjYzE1YTBiYjRiOTQ4NTI2ZjgwYzMxMDZmZWNk) \| [log](https://download.openmmlab.com/mmgen/cyclegan/cyclegan_lsgan_resnet_in_1x1_80k_facades_20210317_160938.log.json) <sup>2</sup>|
|  Ours  |    facades-id0    | 125.1694 | 1.905 |    [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_id0_resnet_in_facades_b1x1_80k.py)     |     [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_80k_facades_convert-bgr_20210902_164411-d8e72b45.pth?versionId=CAEQMhiBgMCZ3rCs3RciIDk0NWIwMmZjNzRhMjRkMTdiMjEyNTdhYTBkMmU4MmRi)      |
|  Ours  |   summer2winter   | 83.7177  | 2.771 |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_resnet_in_summer2winter_b1x1_250k.py)   |   [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_convert-bgr_20210902_165932-fcf08dc1.pth?versionId=CAEQMhiBgIDT37Cs3RciIDNhYzQ3ZWU3MzZjNTQ1ZmJiZmMyZGZiMTc1NzUyZDM1)   |
|  Ours  | summer2winter-id0 | 83.1418  | 2.720 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_id0_resnet_in_summer2winter_b1x1_250k.py) | [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_convert-bgr_20210902_165640-8b825581.pth?versionId=CAEQMhiBgICe3rCs3RciIGNiM2JmNjViNmQ5ZTRhMTQ4YWI0YjFkOTdmNTE3MzFi) |
|  Ours  |   winter2summer   | 72.8025  | 3.129 |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_resnet_in_winter2summer_b1x1_250k.py)   |   [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_convert-bgr_20210902_165932-fcf08dc1.pth?versionId=CAEQMhiBgIDT37Cs3RciIDNhYzQ3ZWU3MzZjNTQ1ZmJiZmMyZGZiMTc1NzUyZDM1)   |
|  Ours  | winter2summer-id0 | 73.5001  | 3.107 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_id0_resnet_in_winter2summer_b1x1_250k.py) | [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_convert-bgr_20210902_165640-8b825581.pth?versionId=CAEQMhiBgICe3rCs3RciIGNiM2JmNjViNmQ5ZTRhMTQ4YWI0YjFkOTdmNTE3MzFi) |
|  Ours  |    horse2zebra    | 64.5225  | 1.418 |    [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_resnet_in_horse2zebra_b1x1_270k.py)    |    [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_convert-bgr_20210902_170004-a32c733a.pth?versionId=CAEQMhiBgMD327Cs3RciIDhkMjhhZDJkYjliYTQyM2M5MzU5ZDYxZGNhZGI5Njc4)    |
|  Ours  |  horse2zebra-id0  | 74.7770  | 1.542 |  [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_id0_resnet_in_horse2zebra_b1x1_270k.py)  |  [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_convert-bgr_20210902_165724-77c9c806.pth?versionId=CAEQMhiBgICF4rCs3RciIDA1YzcxZDI3ZmQwNjRhYTBiZjgzMGJmZWY3MmVhNDZj)  |
|  Ours  |    zebra2horse    | 141.1517 | 3.154 |    [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_resnet_in_zebra2horse_b1x1_270k.py)    |    [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_convert-bgr_20210902_170004-a32c733a.pth?versionId=CAEQMhiBgMD327Cs3RciIDhkMjhhZDJkYjliYTQyM2M5MzU5ZDYxZGNhZGI5Njc4)    |
|  Ours  |  zebra2horse-id0  | 134.3728 | 3.091 |  [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/cyclegan/cyclegan_lsgan_id0_resnet_in_zebra2horse_b1x1_270k.py)  |  [model](https://download.openmmlab.com/mmgen/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_convert-bgr_20210902_165724-77c9c806.pth?versionId=CAEQMhiBgICF4rCs3RciIDA1YzcxZDI3ZmQwNjRhYTBiZjgzMGJmZWY3MmVhNDZj)  |

`FID` comparison with official:

| Dataset  |   facades   | facades-id0 | summer2winter | summer2winter-id0 | winter2summer | winter2summer-id0 | horse2zebra | horse2zebra-id0 | zebra2horse | zebra2horse-id0 |  average   |
| :------: | :---------: | :---------: | :-----------: | :---------------: | :-----------: | :---------------: | :---------: | :-------------: | :---------: | :-------------: | :--------: |
| official | **123.626** | **119.726** |  **77.342**   |    **76.773**     |  **72.631**   |      74.239       | **62.111**  |     77.202      | **138.646** |   **137.050**   | **95.935** |
|   ours   |  124.8033   |  125.1694   |    83.7177    |      83.1418      |    72.8025    |    **73.5001**    |   64.5225   |   **74.7770**   |  141.1571   |  **134.3728**   |   97.79    |

`IS` comparison with evaluation:

| Dataset  |  facades  | facades-id0 | summer2winter | summer2winter-id0 | winter2summer | winter2summer-id0 | horse2zebra | horse2zebra-id0 | zebra2horse | zebra2horse-id0 |  average  |
| :------: | :-------: | :---------: | :-----------: | :---------------: | :-----------: | :---------------: | :---------: | :-------------: | :---------: | :-------------: | :-------: |
| official |   1.638   |    1.697    |     2.762     |     **2.750**     |   **3.293**   |     **3.110**     |    1.375    |    **1.584**    |  **3.186**  |      3.047      |   2.444   |
|   ours   | **1.792** |  **1.905**  |   **2.771**   |       2.720       |     3.129     |       3.107       |  **1.418**  |      1.542      |    3.154    |    **3.091**    | **2.462** |

Note:
1. With a larger identity loss, the image-to-image translation becomes more conservative, which makes less changes. The original authors did not say what is the best weight for identity loss. Thus, in addition to the default setting, we also set the weight of identity loss to 0 (denoting `id0`) to make a more comprehensive comparison.
2. This is the training log before refactoring. Updated logs will be released soon.
