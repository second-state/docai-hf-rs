#!/bin/bash

# Init the WasmEdge environment  (with wasi-nn plugin and pytorch backend)

set -ex

source "$(dirname -- "$0")/env.sh"

# Must use the version after 0.13.1 or build from master branch.
WASMEDGE_VERSION=0.13.1
PYTORCH_VERSION="1.8.2"

download_wasmedge_pytorch_deps() {
  #install pytorch
  #support for linux version Ubuntu 20.04 onwards

  curl -s -L -O --remote-name-all https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip
  unzip -q "libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
  rm -f "libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"

  cp -r "$(pwd)"/libtorch "${WASMEDGE_PATH}"
  export LD_LIBRARY_PATH=${WASMEDGE_PATH}/libtorch/lib:${LD_LIBRARY_PATH}
  export Torch_DIR=${WASMEDGE_PATH}/libtorch
  rm -rf "$(pwd)"/libtorch
}

build_wasmedge_from_source_with_wasi_nn_pytorch() {
  # install requirements
  apt update && apt install git software-properties-common libboost-all-dev llvm-14-dev liblld-14-dev cmake ninja-build gcc g++ -y

  REPO_CURL="https://github.com/WasmEdge/WasmEdge.git"
  REPO_BRANCH="master"

  git clone "${REPO_CURL}"
  pushd WasmEdge
  git checkout "${REPO_BRANCH}"
  mkdir build && pushd build
  cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release \
    -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="PyTorch" -DCMAKE_INSTALL_PREFIX="${WASMEDGE_PATH}"
  ninja && ninja install

  popd
  popd
  rm -rf "$(pwd)"/WasmEdge
}

download_wasmedge_pytorch_deps
build_wasmedge_from_source_with_wasi_nn_pytorch

# ToDo : Add support to directly download WasmEdge with Wasi-NN Pytorch
