set -ex
export CCACHE_DIR="$(pwd)/.ccache"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_NOHASHDIR=true
export CCACHE_MAXSIZE=2G
mkdir -p "${CCACHE_DIR}"
git clone -b v1.0.0 https://gh-proxy.test.osinfra.cn/https://github.com/antgroup/vsag.git vsag_sourcecode
cd vsag_sourcecode
patch -p1 < ../0001-vsag_1.0.0-optimize-rabitq_sq8.patch
wget -q https://boostkit-cicd-view.obs.cn-north-4.myhuaweicloud.com/PR/vsag/v1.0.0/vsag_ut.tar.gz
tar -xf vsag_ut.tar.gz
bash scripts/coverage/run_ut.sh --no-coverage \
    --filter '[ut][RaBitQuantizer],[ut][RaBitQuantizerParameter],[ut][RaBitQSplitDataCell],[ut][query_computer_pool],[ut
  ][SQ8Quantizer],[ut][SQ8QuantizerParameter],[ut][SQ8UniformQuantizer],[ut][SQ8UniformQuantizerParameter],[ut][simd]~[!
  benchmark],[ft][hgraph][rabitq]'
ccache -s || true
