# VM setup (Google Cloud, Linux + NVIDIA GPU)

Create the VM with the GPU attached (GPU quota needed; T4/L4/V100 on N1/G2, A100 on A2).
## 1. NVIDIA driver
Check first: `nvidia-smi`. If it fails, install the driver. On a Deep Learning VM image: `sudo /opt/deeplearning/install-driver.sh`.
Otherwise:
```bash
curl -L https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py -o install_gpu_driver.py
sudo python3 install_gpu_driver.py
sudo reboot
```
After reboot, `nvidia-smi` should list the GPU (see Google's "Install GPU drivers" docs if the script has moved).

## 2. Conda + env
The system Python (3.13) is not used; the env has its own Python 3.10.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)" && conda init

conda create -n raster2seq python=3.10 -y
conda activate raster2seq
which python   # ~/miniconda3/envs/raster2seq/bin/python
```

## 3. Repos (side by side in ~)
```bash
cd ~
git clone -b osm-changes https://github.com/jennahgosciak/Raster2Seq.git
git clone <vectorizing-raster-maps repo URL>
```

## 4. Packages
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r ~/vectorizing-raster-maps/requirements.txt
sudo apt-get install -y libcairo2          # for CairoSVG / render_omt.py
nvcc --version || conda install -c nvidia cuda-toolkit=12.1 -y   # CUDA compiler for step 5
pip check
```

## 5. Attention op (needs the GPU visible)
```bash
pip install ninja   # the build fails without it
cd ~/Raster2Seq/models/ops && sh make.sh && cd ~
python -c "import torch, MultiScaleDeformableAttention; print(torch.cuda.is_available(), torch.version.cuda)"   # True 12.1
```

## 6. wandb
```bash
wandb login   # paste the API key
```
`main_ddp.py` saves the key to `wandb_keys/<user>_wandb.key` in the folder it runs from; keep `wandb_keys/` in `.gitignore`.

## 7. Data
Data is not in git. From the Mac:
```bash
gcloud compute scp data/coco_osm_mini.zip <vm-name>:~/vectorizing-raster-maps/data/ --zone <zone>
```
Then `unzip` it on the VM.

Optional: `plotly_get_chrome` to save `plot_floor.py --compute_stats` histograms.
