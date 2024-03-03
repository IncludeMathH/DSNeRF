Install Colmap [linux上跑通COLMAP](https://zhuanlan.zhihu.com/p/526135749)

```powershell
conda create -n dsnerf python=3.8 -y
conda activate dsnerf
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```