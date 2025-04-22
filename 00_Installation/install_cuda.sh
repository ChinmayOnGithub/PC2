#!/usr/bin/env bash
# install_cuda.sh: Network-based CUDA Toolkit installer for Ubuntu with Docker support
# Checks if CUDA is already installed; if so, skips installation.
# Use --skip-nvidia to avoid NVIDIA driver installation (for containers)

set -e

# Function to print messages
log() { echo -e "[\e[1;32mINFO\e[0m] $*"; }
err() { echo -e "[\e[1;31mERROR\e[0m] $*" >&2; exit 1; }

# Check for root/sudo
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# Parse arguments
SKIP_NVIDIA=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-nvidia)
            SKIP_NVIDIA=true
            shift
            ;;
        *)
            err "Unknown argument: $1"
            ;;
    esac
done

# Check if nvcc (CUDA compiler) is already available
if command -v nvcc &>/dev/null; then
    log "CUDA (nvcc) is already installed. Skipping installation."
    nvcc --version
    exit 0
fi

log "CUDA not found. Proceeding with installation..."

# 1. Update system and install prerequisites
log "Updating apt repositories and installing prerequisites..."
$SUDO apt-get update
$SUDO apt-get install -y wget build-essential lsb-release software-properties-common

# 2. Skip NVIDIA driver installation if requested
if ! $SKIP_NVIDIA; then
    if ! command -v nvidia-smi &>/dev/null; then
        log "Installing NVIDIA driver..."
        $SUDO apt-get install -y nvidia-open
        log "Reboot may be required for the new driver to take effect."
    else
        log "NVIDIA driver already installed."
    fi
else
    log "Skipping NVIDIA driver installation as requested."
fi

# 3. Install CUDA keyring package (new version 1.1-1)
log "Installing CUDA keyring package..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
$SUDO dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# 4. Update package lists
log "Updating package lists..."
$SUDO apt-get update

# 5. Install CUDA Toolkit (specific version 12.8)
log "Installing CUDA Toolkit 12.8..."
$SUDO apt-get -y install cuda-toolkit-12-8

# 6. Setup environment variables
BASHRC="$HOME/.bashrc"
CUDA_PATH="/usr/local/cuda"
if ! grep -q "export PATH=${CUDA_PATH}/bin" "${BASHRC}"; then
    log "Adding CUDA to PATH in ${BASHRC}..."
    echo "export PATH=${CUDA_PATH}/bin:\$PATH" >> "${BASHRC}"
fi
if ! grep -q "export LD_LIBRARY_PATH=${CUDA_PATH}/lib64" "${BASHRC}"; then
    log "Adding CUDA libraries to LD_LIBRARY_PATH in ${BASHRC}..."
    echo "export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH" >> "${BASHRC}"
fi

log "Reloading shell configuration..."
# shellcheck disable=SC1090
source "${BASHRC}" || true

# 7. Verification
log "Verifying CUDA installation..."
nvcc --version || err "nvcc not found after installation!"
if ! $SKIP_NVIDIA; then
    nvidia-smi || err "nvidia-smi failed!"
fi

log "CUDA installation completed successfully!"