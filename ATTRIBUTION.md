# Attribution & Provenance

This project builds on the research code accompanying:

> Yeh, C., Perez, A., Driscoll, A., Azzari, G., Tang, Z., Lobell, D., Ermon, S.,
> & Burke, M. (2020). *Using publicly available satellite imagery and deep
> learning to understand economic well-being in Africa.* **Nature Communications**
> 11, 2583. — code: https://github.com/sustainlab-group/africa_poverty
> (MIT License, © 2022 Christopher Yeh — retained in [LICENSE](LICENSE)).

The DHS cluster wealth-index labels and the African satellite imagery derive
from that project and the underlying [DHS Program](https://dhsprogram.com/)
surveys. Sentinel-2 imagery is accessed via Google Earth Engine.

## What is original to this repository

- **Full PyTorch reimplementation** (`src/acp/`, `scripts/`): data pipeline,
  model factory (timm), training/evaluation engine with proper metrics and early
  stopping, Grad-CAM, and a multi-backbone comparison. The original research
  code was TensorFlow 1.x; none of it remains here.
- **Africa → China zero-shot transfer study.** A purpose-built dataset of 20
  geocoded locations in Guizhou, China
  ([`china/china_coordinates.csv`](china/china_coordinates.csv)) with an
  adversarial design — poverty-alleviation relocation sites that visually mimic
  wealthy suburbs, and cave dwellings invisible to optical satellites — plus a
  domain analysis report
  ([`china/guizhou_dataset_report.md`](china/guizhou_dataset_report.md)).
- **Transfer-learning ablation** isolating the effect of ImageNet pretraining
  vs. from-scratch training on this small (~2k image) regression task.
- **Data-prep utilities** (`scripts/data_prep/`): Google Earth Engine image
  download and DHS survey cleaning.

## License

This repository is released under the MIT License (see [LICENSE](LICENSE)).
The original `africa_poverty` code is © 2022 Christopher Yeh; extensions and the
China study in this repository are by Jiaming Wei.
