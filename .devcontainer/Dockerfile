FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    clang-format \
    vim \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Configure zsh
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="fino-time"/' ~/.zshrc && \
    sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions)/' ~/.zshrc

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

# Copy zsh configuration to the new user's home
RUN cp -r /root/.oh-my-zsh /home/$USERNAME/.oh-my-zsh && \
    cp /root/.zshrc /home/$USERNAME/.zshrc && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.oh-my-zsh && \
    chown $USERNAME:$USERNAME /home/$USERNAME/.zshrc

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install python
COPY install/install_python.sh /install/install_python.sh
RUN bash /install/install_python.sh py312

RUN echo "source activate py312" >> ~/.bashrc
ENV PATH="/home/devuser/conda/bin:$PATH"
ENV PATH="/home/devuser/envs/py312/bin:$PATH"

# Install python packages
COPY install/install_python_packages.sh /install/install_python_packages.sh
RUN bash /install/install_python_packages.sh

# Set zsh as default shell
ENV SHELL=/bin/zsh
CMD [ "zsh" ]
