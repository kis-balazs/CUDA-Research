# System Setup Instructions - Native

⚠️ Linux/Ubuntu/WSL Setup Steps!

1. first do a `sudo apt update && sudo apt upgrade -y && sudo apt autoremove` then pop over to [downloads](https://developer.nvidia.com/cuda-downloads)
2. fill in the following settings that match the device you'll be doing this course on: Operating System
   - Architecture
   - Distribution
   - Version
   - Installer Type
3. you'll have to run a command very similar to the one below in the "runfile section"

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run
```

4. in the end, you should be able to run `nvcc --version` and get info about the nvidia cuda compiler (version and such).
   also run `nvidia-smi` to ensure nvidia recognizes your cuda version and connected GPU

5. If `nvcc` doesn't work, run `echo $SHELL`. If it says bin/bash, add the following lines to the ~/.bashrc rile. If it says bin/zsh, add to the ~/.zshrc file. 
```bash
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```
do `source ~/.zshrc` or `source ~/.bashrc` after this then try `nvcc -V` again

## Alternatively - Containerized

Ubuntu system having a GPU, and use [Docker](https://www.docker.com/) as a container ecosystem to run images and access GPU on it.

- Create container using either [L4T(-CUDA)](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=scoreDESC&query=cuda&page=&pageSize=), [CUDA](https://hub.docker.com/r/nvidia/cuda/tags), or custom build (e.g., for ROS or specialized sub-versions of L4T not building all ML libraries, only required ones)
    - useful hack is to add ```tail -f /dev/null``` as an entrypoint command to hold container alive in case of no auto-exec at the end of the Dockerfile -- easy access & debugging approach
- (login+) Pull Image 
- Run Container with relevant flags, e.g.
```bash
sudo docker run --name cuda-research -it --rm --net=host --runtime nvidia -p 8051:8051 -v /storage/ssd/:/home/cuda-research/storage IMAGE_NAME
```
- Enter & use container
```bash
sudo docker exec -it cuda-research bash
```
```bash
nvcc --version
```