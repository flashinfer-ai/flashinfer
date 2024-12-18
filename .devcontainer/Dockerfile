FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    clang-format \
    libibverbs-dev \
    librdmacm-dev \
    rdma-core \
    libnuma-dev \
    vim \
    openmpi-bin \
    libopenmpi-dev \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install powerlevel10k theme
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

# Install zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Configure zsh
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="powerlevel10k\/powerlevel10k"/' ~/.zshrc && \
    sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions)/' ~/.zshrc

# Set zsh as default shell
ENV SHELL=/bin/zsh
CMD [ "zsh" ]
