## Installation
### Common
```
# Please use Python 3.
git clone --recurse-submodules git@github.com:xieshuqin/16822-project.git
cd 16822-project
pip install -r requirements.txt
pushd CrowdPose/crowdpose-api/PythonAPI
python setup.py install
popd
sudo apt-get install python3-tk
```
### EpipolarPose
```
1. Follow the steps in https://github.com/mkocabas/EpipolarPose to setup the environment and the related data.
```
### Multi-person Pipeline
```
1. Download model/pose_coco/pose_dekr_hrnetw32_coco.pth from https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EmoNwNpq4L1FgUsC9KbWezABSotd3BGOlcWCdkBi91l50g?e=HWuluh and put it into ./dekr_models
2. python multiperson_reconstruction.py # Run the whole algorithm.
3. Note that a small subset of CMU Panoptic Dataset is included in this repo. Feel free to follow the instructions in ./panoptic-toolbox to get more data.
```
## Code We Reuse:
### 3D Single Person Pose Estimation Training Pipeline
We use code from https://github.com/mkocabas/EpipolarPose. Thanks the authors for publishing their code.
### 2D Multi-Person Pose Estimator
We use code from https://github.com/HRNet/DEKR for 2d pose estimation. Thanks the authors for sharing the code.
### Person Re-Id
We use code from https://github.com/KaiyangZhou/deep-person-reid for person re-id. Thank the authors for publishing their code.
### PanopticStudio Toolbox
We use code from https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox to retrieve CMU Panoptic Dataset. We thank the authors for their code.
