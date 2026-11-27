#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export MPLBACKEND="${MPLBACKEND:-Agg}"

# 定义工作路径
JOB_DIR="${JOB_DIR:-work_dir/example}"
TIF_FILE="${TIF_FILE:-image.tif}"
TIF_DIR="${TIF_DIR:-}"

# 定义时间范围
START_DATE="${START_DATE:-2019-01-01}"
END_DATE="${END_DATE:-2025-12-31}"

# 定义 ATL03 / CMR 参数
SHORT_NAME="${SHORT_NAME:-ATL03}"
VERSION="${VERSION:-007}"
PAGE_SIZE="${PAGE_SIZE:-2000}"

# 定义下载行为
DOWNLOAD="${DOWNLOAD:-true}"               # false: 只查询；true: 下载
HARMONY_SUBSET="${HARMONY_SUBSET:-true}"         # true: 下载空间裁剪后的 *_subsetted.h5
DOWNLOAD_FULL_GRANULE="${DOWNLOAD_FULL_GRANULE:-false}" # true: 下载完整 ATL03 granule
OVERWRITE="${OVERWRITE:-false}"             # false: 跳过已存在文件；true: 覆盖
PROMPT_EARTHDATA="${PROMPT_EARTHDATA:-false}"       # true: 下载前在终端提示输入 Earthdata 帐号密码；false: 使用 ~/.netrc 或已有环境变量
EARTHDATA_USERNAME_PRESET="${EARTHDATA_USERNAME:-}"
EARTHDATA_PASSWORD_PRESET="${EARTHDATA_PASSWORD:-}"

# 定义运行稳定性参数
TIMEOUT="${TIMEOUT:-60}"
MAX_RETRIES="${MAX_RETRIES:-3}"
HARMONY_TIMEOUT="${HARMONY_TIMEOUT:-7200}"
HARMONY_POLL_INTERVAL="${HARMONY_POLL_INTERVAL:-20}"
HARMONY_JOB_ID="${HARMONY_JOB_ID:-}"
HARMONY_IGNORE_ERRORS="${HARMONY_IGNORE_ERRORS:-false}"

# 定义 Conda 环境
CONDA_ENV="${CONDA_ENV:-${GEO_ENV:-hab-core}}"
CONDA_BIN="${CONDA_BIN:-conda}"

TIF_PATH="${TIF_PATH:-${JOB_DIR}/${TIF_FILE}}"
OUTPUT_DIR="${ATL03_OUTPUT_DIR:-${JOB_DIR}/ICESat-2}"
CODE_PATH="tools/download_atl03_by_tif.py"

if [ "$HARMONY_SUBSET" = "true" ] && [ "$DOWNLOAD_FULL_GRANULE" = "true" ]; then
    echo "Error: HARMONY_SUBSET and DOWNLOAD_FULL_GRANULE cannot both be true."
    exit 1
fi

echo "正在环境 [${CONDA_ENV}] 中查询 ICESat-2 [${SHORT_NAME} V${VERSION}]..."
if [ -n "${TIF_DIR}" ]; then
    echo "GeoTIFF directory (combined extent): ${TIF_DIR}"
    TIF_ARGS=(--tif-dir "${TIF_DIR}")
else
    echo "GeoTIFF: ${TIF_PATH}"
    TIF_ARGS=(--tif-path "${TIF_PATH}")
fi
echo "Output : ${OUTPUT_DIR}"
echo "Time   : ${START_DATE} 到 ${END_DATE}"

if [ "$DOWNLOAD" = "true" ]; then
    if [ -n "$EARTHDATA_USERNAME_PRESET" ]; then
        EARTHDATA_USERNAME="$EARTHDATA_USERNAME_PRESET"
    else
        EARTHDATA_USERNAME="${EARTHDATA_USERNAME:-}"
    fi

    if [ -n "$EARTHDATA_PASSWORD_PRESET" ]; then
        EARTHDATA_PASSWORD="$EARTHDATA_PASSWORD_PRESET"
    else
        EARTHDATA_PASSWORD="${EARTHDATA_PASSWORD:-}"
    fi

    if [ "$PROMPT_EARTHDATA" = "true" ] && { [ -z "$EARTHDATA_USERNAME" ] || [ -z "$EARTHDATA_PASSWORD" ]; }; then
        if [ ! -t 0 ]; then
            echo "Error: PROMPT_EARTHDATA=true requires an interactive terminal."
            echo "Please export EARTHDATA_USERNAME/EARTHDATA_PASSWORD, set PROMPT_EARTHDATA=false, or configure ~/.netrc."
            exit 1
        fi

        if [ -z "$EARTHDATA_USERNAME" ]; then
            read -r -p "Earthdata username: " EARTHDATA_USERNAME
        fi

        if [ -z "$EARTHDATA_PASSWORD" ]; then
            read -r -s -p "Earthdata password: " EARTHDATA_PASSWORD
            echo
        fi
    fi

    if { [ -n "$EARTHDATA_USERNAME" ] && [ -z "$EARTHDATA_PASSWORD" ]; } || { [ -z "$EARTHDATA_USERNAME" ] && [ -n "$EARTHDATA_PASSWORD" ]; }; then
        echo "Error: Earthdata username and password must be set together."
        exit 1
    fi

    if [ -n "$EARTHDATA_USERNAME" ] && [ -n "$EARTHDATA_PASSWORD" ]; then
        export EARTHDATA_USERNAME
        export EARTHDATA_PASSWORD
    fi
fi

DOWNLOAD_FLAG=""
HARMONY_FLAG=""
FULL_GRANULE_FLAG=""
OVERWRITE_FLAG=""
HARMONY_JOB_ARGS=()
HARMONY_ERROR_ARGS=()

if [ "${HARMONY_IGNORE_ERRORS}" = "true" ]; then
    HARMONY_ERROR_ARGS=(--harmony-ignore-errors)
elif [ "${HARMONY_IGNORE_ERRORS}" != "false" ]; then
    echo "Error: HARMONY_IGNORE_ERRORS must be true or false."
    exit 2
fi

if [ -n "${HARMONY_JOB_ID}" ]; then
    HARMONY_JOB_ARGS=(--harmony-job-id "${HARMONY_JOB_ID}")
fi

if [ "$DOWNLOAD" = "true" ]; then
    DOWNLOAD_FLAG="--download"
fi
if [ "$HARMONY_SUBSET" = "true" ]; then
    HARMONY_FLAG="--harmony-subset"
fi
if [ "$DOWNLOAD_FULL_GRANULE" = "true" ]; then
    FULL_GRANULE_FLAG="--download-full-granule"
fi
if [ "$OVERWRITE" = "true" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

# 使用 conda run -n 指定环境运行，--no-capture-output 确保 print 日志能实时打印出来
"${CONDA_BIN}" run -n "$CONDA_ENV" --no-capture-output \
python "$CODE_PATH" \
    "${TIF_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --short-name "$SHORT_NAME" \
    --version "$VERSION" \
    --page-size "$PAGE_SIZE" \
    --timeout "$TIMEOUT" \
    --max-retries "$MAX_RETRIES" \
    --harmony-timeout "$HARMONY_TIMEOUT" \
    --harmony-poll-interval "$HARMONY_POLL_INTERVAL" \
    "${HARMONY_JOB_ARGS[@]}" \
    "${HARMONY_ERROR_ARGS[@]}" \
    $DOWNLOAD_FLAG \
    $HARMONY_FLAG \
    $FULL_GRANULE_FLAG \
    $OVERWRITE_FLAG
