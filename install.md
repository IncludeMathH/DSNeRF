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
git checkout v1.2.0  # this is the highest compatible version allowed by Mamba
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
```

```powershell
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post1/causal_conv1d-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

wget https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```