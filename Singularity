bootstrap: docker
from: nvidia/cuda:11.0.3-devel-ubuntu20.04


%environment
    # setup necessary bash variables
    export POETRY_VIRTUALENVS_IN_PROJECT=1
    # setup PATH to point to julia, poetry, and blender
    export PATH=$PATH:/julia/bin
    export PATH=$PATH:/poetry/bin
    export PATH=$PATH:/blender

%runscript
    exec bash "$@"

%post

    # install deps available in os repos
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    export LC_ALL=en_US.utf8
    apt-get install -y  build-essential \
                        curl \
                        graphviz \
                        git \
                        wget \
                        ffmpeg \
                        cmake \
                        pipenv
    apt-get clean

    /usr/bin/python3 -m pip install --upgrade pip
    /usr/bin/python3 -m pip install virtualenv

    # build context
    mkdir /build-ctx && cd /build-ctx

    # set up poetry (package manager for python)
    export POETRY_HOME=/poetry
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
    chmod +x /poetry/bin/*


    # Setup blender
    wget "https://yale.box.com/shared/static/nn6n5iyo5m4tzl5u9yoy2dvv1ohk22xj.xz" \
        -O blender.tar.gz
    tar -xf blender.tar.gz
    mv blender-2.* /blender
    chmod +x /blender/blender

    # Set up Julia
    JULIA_VER="1.5.2"
    wget "https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-${JULIA_VER}-linux-x86_64.tar.gz"
    tar -xzf "julia-${JULIA_VER}-linux-x86_64.tar.gz"
    mv "julia-${JULIA_VER}" "/julia"
    chmod +x /julia/bin/*

    # clean up
    rm -rf /build-ctx

    # Add an sbatch workaround
    echo '#!/bin/bash\nssh -y "$HOSTNAME"  sbatch "$@"'  > /usr/bin/sbatch
    chmod +x /usr/bin/sbatch

    # Add an scancel workaround
    echo '#!/bin/bash\nssh -y "$HOSTNAME"  scancel "$@"'  > /usr/bin/scancel
    chmod +x /usr/bin/scancel

    # Add an srun workaround
    echo '#!/bin/bash\nssh -y "$HOSTNAME"  srun "$@"'  > /usr/bin/srun
    chmod +x /usr/bin/srun
