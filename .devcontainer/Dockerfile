FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Install basic utilities
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install NVIDIA Container Toolkit
RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install zsh and required packages
RUN apt-get update && apt-get install -y \
    zsh \
    git-core \
    curl \
    fonts-powerline \
    sudo \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install useful zsh plugins
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Configure zsh
RUN sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# Add source command to zshrc
RUN echo "source \$HOME/.local/bin/env" >> ~/.zshrc

# Set zsh as default shell
RUN chsh -s $(which zsh)