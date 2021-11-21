## Methods we use:
1. 3D single person pose estimation:
    We use code from https://github.com/mkocabas/EpipolarPose , thanks the authors for publishing their code.
2. 2D multi-person pose estimation:
    We use code from https://github.com/HRNet/DEKR for 2d pose estimation, thanks the authors for sharing the code. 


## Installation
```
git clone --recurse-submodules git@github.com:xieshuqin/16822-project.git
pip install -r requirements.txt
pushd CrowdPose/crowdpose-api/PythonAPI
python setup.py install --user
popd

# Installation for EpipolarPose
Follow the steps in https://github.com/mkocabas/EpipolarPose to setup the environment and the related data.
pip install h5py
```
