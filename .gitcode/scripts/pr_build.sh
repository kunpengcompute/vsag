#! /bin/bash
set -e

export CCACHE_DIR="$(pwd)/.ccache"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_NOHASHDIR=true
export CCACHE_MAXSIZE=2G
mkdir -p "${CCACHE_DIR}"

git clone -b v1.0.0 https://gh-proxy.test.osinfra.cn/https://github.com/antgroup/vsag.git vsag_sourcecode
cd vsag_sourcecode
patch -p1 < ../0001-vsag_1.0.0-optimize-rabitq_sq8.patch
wget -q https://boostkit-cicd-view.obs.cn-north-4.myhuaweicloud.com/PR/vsag/v1.0.0/vsag.tar.gz
tar -xf vsag.tar.gz
VSAG_ENABLE_TOOLS=ON make release
ccache -s || true
