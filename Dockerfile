FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update --fix-missing
RUN apt install -y \
    build-essential  \
    cmake \
    pkg-config  \
    htop  \
    gedit  \
    wget \
    git \
    unzip  \
    curl \
    vim \
    software-properties-common \
    net-tools \
    iputils-ping \
    && apt clean

RUN mkdir -p /root/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh
RUN /bin/bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3
RUN rm /root/miniconda3/miniconda.sh

RUN /root/miniconda3/bin/conda init bash
ENV PATH=/root/miniconda3/bin:${PATH}

ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8:${LD_LIBRARY_PATH}

RUN echo "PS1='\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]# '\n\
if [ -x /usr/bin/dircolors ]; then\n\
    test -r ~/.dircolors && eval \"\$(dircolors -b ~/.dircolors)\" || eval \"\$(dircolors -b)\"\n\
    alias ls='ls --color=auto'\n\
    alias grep='grep --color=auto'\n\
    alias fgrep='fgrep --color=auto'\n\
    alias egrep='egrep --color=auto'\n\
fi" >> /root/.bashrc

RUN conda create -n conceptgraph python=3.10 -y
RUN conda run -n conceptgraph conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl -y
RUN conda run -n conceptgraph pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy
RUN conda run -n conceptgraph pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN conda run -n conceptgraph conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2 -y

ARG USE_CUDA=1
ARG TORCH_ARCH=3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.8/

RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
libsm6=2:* libxext6=2:* git=1:* \
vim=2:* -y \
&& apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git Grounded-Segment-Anything
WORKDIR /opt/Grounded-Segment-Anything
RUN conda run -n conceptgraph pip install --no-cache-dir -e segment_anything

# When using build isolation, PyTorch with newer CUDA is installed and can't compile GroundingDINO
RUN conda run -n conceptgraph pip install --no-cache-dir wheel
RUN conda run -n conceptgraph pip install --no-cache-dir --no-build-isolation -e GroundingDINO

WORKDIR /opt
RUN git clone https://github.com/xinyu1205/recognize-anything.git
RUN conda run -n conceptgraph pip install --no-cache-dir --no-build-isolation -r ./recognize-anything/requirements.txt
RUN conda run -n conceptgraph pip install --no-cache-dir --no-build-isolation -e ./recognize-anything/

RUN conda run -n conceptgraph \
    pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai

WORKDIR /opt
RUN git clone https://github.com/krrish94/chamferdist.git chamferdist && \
    cd chamferdist && \
    conda run -n conceptgraph pip install .
WORKDIR /opt
RUN git clone https://github.com/Dhruv2012/gradslam.git gradslam && \
    cd gradslam && \
    conda run -n conceptgraph pip install .

RUN conda run -n conceptgraph conda install -c conda-forge libstdcxx-ng -y

WORKDIR /root
RUN git clone https://github.com/concept-graphs/concept-graphs.git concept-graphs
WORKDIR /root/concept-graphs
RUN conda run -n conceptgraph pip install -e .
CMD ["/bin/bash"]
