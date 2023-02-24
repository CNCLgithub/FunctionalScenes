#!/usr/bin/env sh

# install pytorch3d. current requires special treatment
#
# OPTION 1: compile from  src
# curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz && \
# tar xzf 1.10.0.tar.gz -C "${SENV[pyenv]}" && \
python -m pip install -v git+https://github.com/facebookresearch/pytorch3d.git@main
#
# OPTION 2: use wheel
# pip install --no-index --no-cache-dir pytorch3d \
#     -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html
