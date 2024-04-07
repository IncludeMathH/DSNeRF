# For DSNeRF
```powershell
conda create -n dsnerf-mamba
conda activate dsnerf-mamba
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
# For Mamba

First, note that these two commands should produce matching CUDA versions:

```powershell
python3 -c 'import torch; print(torch.version.cuda)'
nvcc --version
```
since nvcc will be used during the build of causal_conv1d. If they don't, you might need to do:

```powershell
sudo update-alternatives --config cuda
```

and then set the CUDA alternatives version to the version reported by torch.version.cuda

Then:
```powershell
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal_conv1d
git checkout v1.2.0.post2
# edit setup.py to add the lines here:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_60,code=sm_60")
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
```

```powershell
git clone git@github.com:state-spaces/mamba.git
cd mamba
git checkout v1.2.0.post1
# edit setup.py to add the lines here:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_60,code=sm_60")
MAMBA_FORCE_BUILD=TRUE pip install .
```