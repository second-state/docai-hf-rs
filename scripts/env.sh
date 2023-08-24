#!/bin/bash

WASMEDGE_DEFAULT_PATH=${HOME}/.wasmedge

# if install wasmedge in custom path, please set the variable `WASMEDGE_PATH` before use the env.sh

if [[ -z "${WASMEDGE_PATH}" ]]; then
  export WASMEDGE_PATH=${WASMEDGE_DEFAULT_PATH}
fi

export WASMEDGE_BIN_PATH=${WASMEDGE_PATH}/bin
export WASMEDGE_LIB_PATH=${WASMEDGE_PATH}/lib

#ToDo : need to get pytorch deps wihin this folder / wasmedge 
export LD_LIBRARY_PATH=${WASMEDGE_PATH}/libtorch/lib:${LD_LIBRARY_PATH}
export Torch_DIR=${WASMEDGE_PATH}/libtorch

# need these environment variables to run
export WASMEDGE_PLUGIN_PATH=${WASMEDGE_LIB_PATH}/wasmedge
export PATH=${WASMEDGE_BIN_PATH}:${PATH}
export LD_LIBRARY_PATH=${WASMEDGE_LIB_PATH}:${LD_LIBRARY_PATH}
