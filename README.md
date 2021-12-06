## Methods we use:
1. 3D single person pose estimation:
    We use code from https://github.com/mkocabas/EpipolarPose , thanks the authors for publishing their code.
2. 2D multi-person pose estimation:
    We use code from https://github.com/HRNet/DEKR for 2d pose estimation, thanks the authors for sharing the code. 


## Installation
### Common
```
git clone --recurse-submodules git@github.com:xieshuqin/16822-project.git
pip install -r requirements.txt
pushd CrowdPose/crowdpose-api/PythonAPI
python setup.py install --user
popd
```
### EpipolarPose
```
Follow the steps in https://github.com/mkocabas/EpipolarPose to setup the environment and the related data.
```
### Multi-person Pipeline
```
1. Download pose_dekr_hrnetw32_coco.pth from https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EmoNwNpq4L1FgUsC9KbWezABSotd3BGOlcWCdkBi91l50g?e=HWuluh and put it into ./dekr_models
2. python multiperson_reconstruction.py
3. Note that a small subset of CMU Panoptic Dataset is included in this repo. Feel free to follow the instructions in ./panoptic-toolbox to get more data.
```
