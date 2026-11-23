set -ex
export CCACHE_DIR="$(pwd)/.ccache"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_NOHASHDIR=true
export CCACHE_MAXSIZE=2G
mkdir -p "${CCACHE_DIR}"
##编译过程中需要下载的依赖包获取解压
wget -q https://boostkit-cicd-view.obs.cn-north-4.myhuaweicloud.com/PR/vsag/v1.0.0/vsag_ut.tar.gz
tar -xf vsag_ut.tar.gz
bash scripts/coverage/run_ut.sh --no-coverage \
    --filter '[ut][RaBitQuantizer],[ut][RaBitQuantizerParameter],[ut][RaBitQSplitDataCell],[ut][query_computer_pool],[ut
  ][SQ8Quantizer],[ut][SQ8QuantizerParameter],[ut][SQ8UniformQuantizer],[ut][SQ8UniformQuantizerParameter],[ut][simd]~[!
  benchmark],[ft][hgraph][rabitq]'
ccache -s || true
