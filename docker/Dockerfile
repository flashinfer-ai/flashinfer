ARG CUDA_BASE_IMAGE
FROM ${CUDA_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget

# ---- test target ----
FROM base AS test

# Install python
COPY docker/install/install_python.sh /install/install_python.sh
RUN bash /install/install_python.sh /opt/conda py312

WORKDIR /workspace

RUN echo "source activate py312" >> ~/.bashrc
ENV PATH="/opt/conda/bin:$PATH"
ENV PATH="/opt/conda/envs/py312/bin:$PATH"

# Set LD_LIBRARY_PATH to ensure pip-installed nvidia libs take precedence over system libraries
ARG NVIDIA_LIB_PATH
ENV LD_LIBRARY_PATH="/opt/conda/envs/py312/lib/python3.12/site-packages/${NVIDIA_LIB_PATH}/:$LD_LIBRARY_PATH"

# Triton
ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# Install torch and other python packages
COPY requirements.txt /install/requirements.txt
COPY docker/install/install_python_packages.sh /install/install_python_packages.sh
ARG CUDA_VERSION
RUN bash /install/install_python_packages.sh ${CUDA_VERSION}

ARG INSTALL_TILELANG=false
RUN if [ "$INSTALL_TILELANG" = "true" ]; then pip install tilelang cuda-tile; fi

# Install mpi4py in the conda environment
RUN conda install -n py312 -y mpi4py mpich

# Configure pip for user-site installations (allows arbitrary users to install packages)
# This enables 'pip install --user' and 'pip install -e .' to work for any user
RUN mkdir -p /opt/pip-user && chmod 1777 /opt/pip-user
ENV PYTHONUSERBASE=/opt/pip-user
ENV PATH="/opt/pip-user/bin:$PATH"

# ---- dev target ----
FROM base AS dev

RUN apt-get update && apt-get install -y \
    clang-format \
    clangd-19 \
    vim \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=devuser
ARG USER_UID=1003
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Remove default 'ubuntu' user (UID 1000) to prevent devcontainer permission conflicts
# Ref: https://github.com/rapidsai/devcontainers/pull/373
RUN if grep ubuntu:x:1000:1000 /etc/passwd >/dev/null; then userdel -f -r ubuntu; fi

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install python
COPY docker/install/install_python.sh /install/install_python.sh
RUN bash /install/install_python.sh /home/$USERNAME/conda py312

RUN echo "source activate py312" >> ~/.bashrc
ENV PATH="/home/$USERNAME/conda/bin:$PATH"
ENV PATH="/home/$USERNAME/conda/envs/py312/bin:$PATH"

# Install torch and other python packages
COPY requirements.txt /install/requirements.txt
COPY docker/install/install_python_packages.sh /install/install_python_packages.sh
ARG CUDA_VERSION
ARG INSTALL_TILELANG=false
RUN bash /install/install_python_packages.sh ${CUDA_VERSION} && pip3 install pre-commit && \
    if [ "$INSTALL_TILELANG" = "true" ]; then pip install tilelang cuda-tile; fi

# Install mpi4py in the conda environment
RUN conda install -n py312 -y mpi4py mpich

# Install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Configure zsh
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="fino-time"/' ~/.zshrc && \
    sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions)/' ~/.zshrc

# clangd
ENV PATH="/usr/lib/llvm-19/bin:$PATH"

# Triton
ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# Set zsh as default shell
ENV SHELL=/bin/zsh
CMD [ "zsh" ]
