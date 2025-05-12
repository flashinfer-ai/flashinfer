ARG ROCM_VERSION=6.4

FROM mambaorg/micromamba:2.1.1 as micromamba

FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete

ARG ROCM_VERSION
ARG PY_VERSION=3.12
ARG TORCH_VERSION=2.6

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    clang-format \
    clangd-19 \
    vim \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=devuser
ARG USER_UID=1003
ARG USER_GID=$USER_UID
ARG MAMBA_USER_ID=${USER_UID}
ARG MAMBA_USER_GID=${USER_GID}
ENV MAMBA_USER=$USERNAME
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

# To silence warning where the UID and GID are out of the usual range
RUN sed -i 's/^\(UID_MAX\s*\).*$/\11000000000/' /etc/login.defs && \
    sed -i 's/^\(GID_MAX\s*\).*$/\11000000000/' /etc/login.defs

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

RUN usermod -a -G render $USERNAME

# Copy over needed files from the micromamba image
# Refer: https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html
COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Setup useful git aliases
RUN git config --global alias.st status
RUN git config --global alias.co checkout
RUN git config --global alias.br branch
RUN git config --global alias.ci commit

# Create the new micromamba environment
ARG MAMBA_ENV_NAME=flashinfer-py${PY_VERSION}-torch${TORCH_VERSION}-rocm${ROCM_VERSION}

RUN /bin/micromamba create -n ${MAMBA_ENV_NAME} python=${PY_VERSION}

# Install needed packages for flashinfer development within the environment
RUN /bin/micromamba run -n ${MAMBA_ENV_NAME} pip install torch==${TORCH_VERSION} -f https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/
RUN /bin/micromamba run -n ${MAMBA_ENV_NAME} pip install cmake ninja scikit-build-core setuptools-scm
RUN /bin/micromamba run -n ${MAMBA_ENV_NAME} pip install pre-commit

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

CMD ["/bin/bash"]
