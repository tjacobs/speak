#!/usr/bin/env bash
# Install faster-whisper with a CUDA build of ctranslate2 so listen.py runs on GPU.
# Usage: ./install_listen.sh

# Exit on error, undefined variables, and pipe failure
set -euo pipefail

# Config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
FASTER_WHISPER_DIR="${HOME}/faster-whisper"
CTRANSLATE2_DIR="${HOME}/CTranslate2"
CTRANSLATE2_LIBRARY="/usr/local/lib/libctranslate2.so"
CUDA_BIN="/usr/local/cuda/bin"
CUDA_ARCHITECTURE=87
BUILD_JOBS=3

# Main
main() {
    check_prerequisites
    clone_faster_whisper
    clone_ctranslate2
    build_ctranslate2
    install_ctranslate2
    install_python_packages
    verify_cuda
    echo "Done. Run ./listen.py to transcribe from the microphone."
}

# Quit early when build tools or CUDA are missing
check_prerequisites() {
    # Check CUDA toolkit
    if [[ ! -x "${CUDA_BIN}/nvcc" ]]; then
        echo "CUDA toolkit not found at ${CUDA_BIN}. Install it first."
        exit 1
    fi

    # Check cmake
    if ! command -v cmake >/dev/null 2>&1; then
        echo "cmake not found. Install with: sudo apt install cmake"
        exit 1
    fi

    # Check venv
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        echo "Python venv not found at ${VENV_DIR}. Create it first."
        exit 1
    fi
}

# Clone faster-whisper when missing
clone_faster_whisper() {
    if [[ -d "${FASTER_WHISPER_DIR}" ]]; then
        echo "faster-whisper already cloned."
        return 0
    fi
    git clone https://github.com/SYSTRAN/faster-whisper.git "${FASTER_WHISPER_DIR}"
}

# Clone ctranslate2 with submodules when missing
clone_ctranslate2() {
    if [[ -d "${CTRANSLATE2_DIR}" ]]; then
        echo "CTranslate2 already cloned."
        return 0
    fi
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git "${CTRANSLATE2_DIR}"
}

# Build ctranslate2 with CUDA for the jetson orin gpu
build_ctranslate2() {
    # Skip when already built
    if [[ -f "${CTRANSLATE2_DIR}/build/libctranslate2.so.4.8.1" || -n "$(ls "${CTRANSLATE2_DIR}/build/libctranslate2.so"* 2>/dev/null)" ]]; then
        echo "CTranslate2 already built."
        return 0
    fi

    # Configure and build, takes around 30 minutes
    cd "${CTRANSLATE2_DIR}"
    PATH="${CUDA_BIN}:${PATH}" cmake -B build -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURE}"
    PATH="${CUDA_BIN}:${PATH}" cmake --build build "-j${BUILD_JOBS}"
}

# Install the ctranslate2 library system wide
install_ctranslate2() {
    # Skip when already installed
    if [[ -f "${CTRANSLATE2_LIBRARY}" ]]; then
        echo "CTranslate2 library already installed."
        return 0
    fi
    sudo cmake --install "${CTRANSLATE2_DIR}/build"
    sudo ldconfig
}

# Install python packages into the speak venv
install_python_packages() {
    # Install the ctranslate2 python wrapper built against the local library
    "${VENV_DIR}/bin/pip" install pybind11 wheel
    "${VENV_DIR}/bin/pip" install "${CTRANSLATE2_DIR}/python"

    # Install faster-whisper from the local clone
    "${VENV_DIR}/bin/pip" install "${FASTER_WHISPER_DIR}"
}

# Quit when the installed ctranslate2 cannot see the gpu
verify_cuda() {
    local device_count
    device_count="$(cd "${SCRIPT_DIR}" && "${VENV_DIR}/bin/python" -c 'import ctranslate2; print(ctranslate2.get_cuda_device_count())')"
    if [[ "${device_count}" -lt 1 ]]; then
        echo "CUDA verification failed, ctranslate2 sees no GPU."
        exit 1
    fi
    echo "CUDA OK, ${device_count} GPU found."
}

main "$@"
