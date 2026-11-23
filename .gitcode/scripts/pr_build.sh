#! /bin/bash
set -e

export CCACHE_DIR="$(pwd)/.ccache"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_NOHASHDIR=true
export CCACHE_MAXSIZE=2G
mkdir -p "${CCACHE_DIR}"
##编译过程中需要下载的依赖包获取解压
wget -q https://boostkit-cicd-view.obs.cn-north-4.myhuaweicloud.com/PR/vsag/v1.0.0/vsag.tar.gz
tar -xf vsag.tar.gz
VSAG_ENABLE_TOOLS=ON make release
ccache -s || true
