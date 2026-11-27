#!/bin/bash
set -euo pipefail
export MPLBACKEND="${MPLBACKEND:-Agg}"


# Work paths
JOB_DIR="${JOB_DIR:-work_dir/example}"
TIF_IMG="${TIF_IMG:-image.tif}"
ICESAT_DIR="${ICESAT_DIR:-ICESat-2}"

# Patch/depth parameters
WS="${WS:-1024}"
OL="${OL:-0.5}"
DR="${DR:-1}"
PR="${PR:-700}"
BS="${BS:--1,-1}"
HEIGHT_SOURCE="${HEIGHT_SOURCE:-absolute}" # [absolute, coarse_ah]
CONTROL_SOURCE="${CONTROL_SOURCE:-photon}" # [photon, model]
COMPUTE_UNCERTAINTY="${COMPUTE_UNCERTAINTY:-1}" # [0, 1]
FUSION_MODE="${FUSION_MODE:-mean}" # [mean, uncertainty]
GROUND_FILTER="${GROUND_FILTER:-auto}" # [auto, 0, 1]
EXPORT_CALIBRATED_WINDOWS="${EXPORT_CALIBRATED_WINDOWS:-0}"
GRAPH_CONTROL_LAMBDA="${GRAPH_CONTROL_LAMBDA:-1.0}"
GRAPH_OVERLAP_LAMBDA="${GRAPH_OVERLAP_LAMBDA:-1.0}"
OVERLAP_MASK="${OVERLAP_MASK:-all}" # [all, nonground]
PHOTON_TABLE="${PHOTON_TABLE:-}"
PHOTON_TABLE_SHARED="${PHOTON_TABLE_SHARED:-0}"
ALLOW_MODEL_FALLBACK="${ALLOW_MODEL_FALLBACK:-0}"
MIN_PHOTON_CONTROLS="${MIN_PHOTON_CONTROLS:-4}"
ATL03_WORKERS="${ATL03_WORKERS:-1}"
ATL03_WORKER_THREADS="${ATL03_WORKER_THREADS:-1}"

# Model/env parameters
MODEL_NAME="${MODEL_NAME:-Distill-Any-Depth}" # [Distill-Any-Depth, DEPTH-ANYTHING-V1, DEPTH-ANYTHING-V2, SynRS3D]
MLSBRN_ENV="${MLSBRN_ENV:-mlsbrn}"
MLS_FAST_MASK_ENCODING="${MLS_FAST_MASK_ENCODING:-0}" # [0, 1]; exact CPU RLE optimization
GEO_ENV="${GEO_ENV:-hab-core}"
CONDA_BIN="${CONDA_BIN:-conda}"
GPU_ID="${GPU_ID:-0}"

# MLS-BRN parameters
GPUS="${GPUS:-1}"
CITY="test"
NMS_IOU_THRESHOLD=0.2

# PLiDAR parameters
SEG_LEN=1000

# Distributed settings for MLS-BRN
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29705}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

RESUME=${RESUME:-1}
FROM_STEP=1
THROUGH_STEP=10
FROM_STEP_SET=0
STEP_RANGE_SET=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_full_pipeline.sh [OPTIONS]

Options:
  --force         Recompute the selected stages.
  --from-step N   Force a restart from stage N through stage 10.
  --steps N[-M]   Run only stage N or stages N through M, with normal cache reuse.
  --help          Show this help message.

Stages:
  1  Split GeoTIFF into patches
  2  Convert GeoTIFF patches to PNG/COCO JSON
  3  Run MLS-BRN footprint inference
  4  Convert MLS-BRN result.pkl to CSV
  5  Convert footprint_direct.csv to GeoJSON
  6  Extract ICESat-2 ground/building photons
  7  Build sparse control-height GeoTIFF
  8  Run depth estimation for MODEL_NAME
  9  Convert relative heights to absolute height
  10 Attach absolute heights to building polygons

Environment:
  RESUME=1 is the default and skips completed stages.
  RESUME=0 is equivalent to --force.

Configuration:
  HEIGHT_SOURCE=absolute|coarse_ah
  CONTROL_SOURCE=photon|model
    photon: use ICESat-2 sparse controls (the original workflow).
    model: preserve a metric model's scale and optimize patch consistency without photon data.
  MLS_FAST_MASK_ENCODING=1 enables exact cropped CPU mask RLE encoding (default 0).
  ATL03_WORKERS=1 controls independent photon beam processes; set 4 for optional parallel extraction.
  ATL03_WORKER_THREADS=1 controls numerical threads per parallel worker.
  COMPUTE_UNCERTAINTY=0|1 controls test-time augmentation and uncertainty output.
  FUSION_MODE=mean|uncertainty controls overlap fusion.
  GROUND_FILTER=auto|0|1 controls RH2AH ground filtering. auto disables it for SynRS3D.
  GRAPH_CONTROL_LAMBDA=1.0 and GRAPH_OVERLAP_LAMBDA=1.0 weight squared graph residuals.
  OVERLAP_MASK=all|nonground selects pixels for both overlap-statistic stages.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --force)
            RESUME=0
            shift
            ;;
        --from-step)
            if [ "${STEP_RANGE_SET}" -eq 1 ]; then
                echo "ERROR: --steps and --from-step cannot be combined." >&2
                exit 2
            fi
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --from-step requires a stage number from 1 to 10." >&2
                exit 2
            fi
            FROM_STEP="$2"
            FROM_STEP_SET=1
            shift 2
            ;;
        --steps)
            if [ "${FROM_STEP_SET}" -eq 1 ] || [ "${STEP_RANGE_SET}" -eq 1 ]; then
                echo "ERROR: select a stage range only once." >&2
                exit 2
            fi
            if [ "$#" -lt 2 ] || ! [[ "$2" =~ ^([1-9]|10)(-([1-9]|10))?$ ]]; then
                echo "ERROR: --steps needs a stage or ascending range within 1-10." >&2
                exit 2
            fi
            FROM_STEP="${BASH_REMATCH[1]}"
            THROUGH_STEP="${BASH_REMATCH[3]:-${FROM_STEP}}"
            STEP_RANGE_SET=1
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if ! [[ "${FROM_STEP}" =~ ^[0-9]+$ ]] || [ "${FROM_STEP}" -lt 1 ] || [ "${FROM_STEP}" -gt 10 ]; then
    echo "ERROR: --from-step must be a stage number from 1 to 10." >&2
    exit 2
fi

if [ "${FROM_STEP}" -gt "${THROUGH_STEP}" ]; then
    echo "ERROR: --steps needs an ascending range." >&2
    exit 2
fi

stage_selected() {
    [ "$1" -ge "${FROM_STEP}" ] && [ "$1" -le "${THROUGH_STEP}" ]
}

if [ "${RESUME}" != "0" ] && [ "${RESUME}" != "1" ]; then
    echo "ERROR: RESUME must be 0 or 1." >&2
    exit 2
fi

graph_lambda_values="$(python - "${GRAPH_CONTROL_LAMBDA}" "${GRAPH_OVERLAP_LAMBDA}" <<'PY'
import math
import sys
try:
    values = [float(value) for value in sys.argv[1:]]
except ValueError:
    raise SystemExit("ERROR: graph coefficients must be numbers.")
if not all(math.isfinite(value) and value >= 0 for value in values) or not any(values):
    raise SystemExit("ERROR: graph coefficients must be finite, nonnegative, and not both zero.")
print(*values)
PY
)"
read -r GRAPH_CONTROL_LAMBDA GRAPH_OVERLAP_LAMBDA <<< "${graph_lambda_values}"
case "${OVERLAP_MASK}" in
    all|nonground) ;;
    *) echo "ERROR: OVERLAP_MASK must be all or nonground." >&2; exit 2 ;;
esac

if [ "${MLS_FAST_MASK_ENCODING}" != "0" ] && [ "${MLS_FAST_MASK_ENCODING}" != "1" ]; then
    echo "ERROR: MLS_FAST_MASK_ENCODING must be 0 or 1." >&2
    exit 2
fi

if [ "${COMPUTE_UNCERTAINTY}" != "0" ] && [ "${COMPUTE_UNCERTAINTY}" != "1" ]; then
    echo "ERROR: COMPUTE_UNCERTAINTY must be 0 or 1." >&2
    exit 2
fi
if [ "${PHOTON_TABLE_SHARED}" != "0" ] && [ "${PHOTON_TABLE_SHARED}" != "1" ]; then
    echo "ERROR: PHOTON_TABLE_SHARED must be 0 or 1." >&2
    exit 2
fi
if [ "${ALLOW_MODEL_FALLBACK}" != "0" ] && [ "${ALLOW_MODEL_FALLBACK}" != "1" ]; then
    echo "ERROR: ALLOW_MODEL_FALLBACK must be 0 or 1." >&2
    exit 2
fi
if [ "${GROUND_FILTER}" != "auto" ] && [ "${GROUND_FILTER}" != "0" ] && [ "${GROUND_FILTER}" != "1" ]; then
    echo "ERROR: GROUND_FILTER must be auto, 0, or 1." >&2
    exit 2
fi
if ! [[ "${MIN_PHOTON_CONTROLS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MIN_PHOTON_CONTROLS must be a positive integer." >&2
    exit 2
fi

if ! [[ "${ATL03_WORKERS}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${ATL03_WORKER_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ATL03_WORKERS and ATL03_WORKER_THREADS must be positive integers." >&2
    exit 2
fi

case "${CONTROL_SOURCE}" in
    photon|model) ;;
    *) echo "ERROR: CONTROL_SOURCE must be photon or model." >&2; exit 2 ;;
esac

case "${FUSION_MODE}" in
    mean|uncertainty) ;;
    *) echo "ERROR: FUSION_MODE must be mean or uncertainty." >&2; exit 2 ;;
esac

if stage_selected 8 && stage_selected 9 \
    && [ "${FUSION_MODE}" = "uncertainty" ] && [ "${COMPUTE_UNCERTAINTY}" != "1" ]; then
    echo "ERROR: FUSION_MODE=uncertainty requires COMPUTE_UNCERTAINTY=1." >&2
    exit 2
fi

# Resolve project root so the script can be launched from any directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_DIR}"

# Resolve the same backend and checkpoints used by the batch entry point.
model_config="$(python scripts/runtime/model_config.py "${MODEL_NAME}")" || exit 2
mapfile -t model_values <<< "${model_config}"
DEPTH_CODE_PATH="${model_values[0]}"
DEPTH_ENV="${model_values[1]}"
DEPTH_CHECKPOINT="${model_values[2]}"
CONFIG_FILE="${model_values[3]}"
CHECKPOINT="${model_values[4]}"

if [ "${GROUND_FILTER}" = "auto" ]; then
    if [ "${MODEL_NAME}" = "SynRS3D" ]; then
        GROUND_FILTER=0
    else
        GROUND_FILTER=1
    fi
fi

if stage_selected 9 && [ "${CONTROL_SOURCE}" = "model" ] && [ "${MODEL_NAME}" != "SynRS3D" ]; then
    echo "ERROR: CONTROL_SOURCE=model currently requires MODEL_NAME=SynRS3D because its output has metric scale." >&2
    exit 2
fi
if stage_selected 9 && [ "${ALLOW_MODEL_FALLBACK}" = "1" ] && [ "${MODEL_NAME}" != "SynRS3D" ]; then
    echo "ERROR: photon fallback requires metric-scale MODEL_NAME=SynRS3D." >&2
    exit 2
fi

if stage_selected 8 && [ "${MODEL_NAME}" = "DEPTH-ANYTHING-V1" ] && [ "${COMPUTE_UNCERTAINTY}" = "1" ]; then
    echo "ERROR: V1 requires COMPUTE_UNCERTAINTY=0; uncertainty is unavailable in its adapter." >&2
    exit 2
fi

# Derived paths
BOUND_TIF="${INPUT_TIF:-${JOB_DIR}/${TIF_IMG}}"
ICESAT_PATH="${ICESAT_PATH:-}"
PATCH_DIR="${JOB_DIR}/windowsize-${WS}-overlap-${OL}-downratio-${DR}-patches"
IMAGE_DIR="${JOB_DIR}/windowsize-${WS}-overlap-${OL}-downratio-${DR}-images"
ANNO_FILE="${JOB_DIR}/$(basename "${JOB_DIR}")_roof.json"
DEPTH_WORK_DIR="${JOB_DIR}/${MODEL_NAME}-windowsize-${WS}-overlap-${OL}-downratio-${DR}-depth"

PHOTON_XLSX="${PHOTON_TABLE:-${JOB_DIR}/extracted_height.xlsx}"
FOOTPRINT_GEOJSON="${JOB_DIR}/footprint_direct_nms.geojson"
CONTROL_HEIGHT_TIF="${JOB_DIR}/sparse_control_height.tif"
ABSOLUTE_HEIGHT_TIF="${DEPTH_WORK_DIR}/AbsoluteHeight/output.tif"
COARSE_AH_TIF="${DEPTH_WORK_DIR}/coarse_AH/CH_final_ransac.tif"

case "${HEIGHT_SOURCE}" in
    "absolute")
        HEIGHT_TIF="${ABSOLUTE_HEIGHT_TIF}"
        FINAL_GEOJSON="${JOB_DIR}/footprint_direct_nms_absolute_height.geojson"
        ;;
    "coarse_ah")
        HEIGHT_TIF="${COARSE_AH_TIF}"
        FINAL_GEOJSON="${JOB_DIR}/footprint_direct_nms_coarse_ah_height.geojson"
        ;;
    *)
        echo "ERROR: unknown HEIGHT_SOURCE: ${HEIGHT_SOURCE}" >&2
        echo "Supported height sources: absolute, coarse_ah" >&2
        exit 2
        ;;
esac

MLS_BRN_DIR="${PROJECT_DIR}/MLS-BRN"
MLS_OUTPUT_DIR="${MLS_OUTPUT_DIR:-${JOB_DIR}/mls_results}"
PKL_FILE="${MLS_OUTPUT_DIR}/result.pkl"
FOOTPRINT_CSV="${MLS_OUTPUT_DIR}/footprint_direct.csv"

mkdir -p "${JOB_DIR}"
CACHE_DIR="${JOB_DIR}/.pipeline-cache"
exec 8>"${JOB_DIR}/.pipeline.lock"
flock -n 8 || { echo "ERROR: another pipeline is using ${JOB_DIR}" >&2; exit 1; }

stage_status_file="${JOB_DIR}/stage_status.json"
active_step=""
active_title=""
stage_started=0

finish_run() {
    local exit_code=$?
    local stage_args=()
    if [ -n "${active_step}" ]; then
        stage_args=(--step "${active_step}" --title "${active_title}" --elapsed "$((SECONDS - stage_started))")
    fi
    python scripts/runtime/run_status.py finish "${stage_status_file}" "${exit_code}" "${stage_args[@]}" \
        || echo "ERROR: Could not write final stage status to ${stage_status_file}" >&2
    return "${exit_code}"
}

mark_active_stage() {
    active_step="$1"
    active_title="$2"
    stage_started=${SECONDS}
}

record_stage() {
    python scripts/runtime/run_status.py stage "${stage_status_file}" "${active_step}" "${active_title}" \
        "$1" "$((SECONDS - stage_started))"
}

skip_stage() {
    mark_active_stage "$1" "$2"
    echo "[SKIP] [$1/10] $2"
    record_stage skipped
    active_step=""
}

pipeline_run_id="${HAB_PIPELINE_RUN_ID:-${BASHPID}-$(date -u +%s)}"
python scripts/runtime/run_status.py start "${stage_status_file}" "${pipeline_run_id}"
trap finish_run EXIT

pipeline_cache() {
    python scripts/runtime/pipeline_cache.py "$@"
}

if stage_selected 6 && [ "${CONTROL_SOURCE}" = "photon" ] \
    && { [ "${PHOTON_TABLE_SHARED}" != "1" ] || [ -n "${ICESAT_PATH}" ]; }; then
    atl03_args=()
    if [ -n "${ICESAT_PATH}" ]; then
        atl03_args+=(--directory "${ICESAT_PATH}")
    fi
    mark_active_stage 6 "Select ATL03 files"
    ICESAT_PATH="$(pipeline_cache select-atl03 "${JOB_DIR}" "${JOB_DIR}/${ICESAT_DIR}" "${atl03_args[@]}")"
    active_step=""
fi

count_files() {
    local dir="$1"
    local pattern="$2"
    if [ -d "${dir}" ]; then
        find "${dir}" -maxdepth 1 -type f -name "${pattern}" | wc -l | tr -d ' '
    else
        echo 0
    fi
}

nonempty_file() {
    [ -s "$1" ]
}

patch_count() {
    count_files "${PATCH_DIR}" "*.tif"
}

check_split() {
    "${CONDA_BIN}" run -n "${GEO_ENV}" python tools/validate_patch_grid.py \
        --input-tif "${BOUND_TIF}" \
        --patch-dir "${PATCH_DIR}" \
        --window-size "${WS}" \
        --overlap "${OL}" \
        --down-ratio "${DR}" >/dev/null 2>&1
}

check_depth() {
    local patches
    local depths
    local uncertainties
    patches="$(patch_count)"
    depths="$(count_files "${DEPTH_WORK_DIR}/depth" "*_depth.tif")"
    uncertainties="$(count_files "${DEPTH_WORK_DIR}/uncertainty" "*_uncertainty.tif")"
    [ "${patches}" -gt 0 ] && [ "${depths}" -eq "${patches}" ] \
        && { [ "${COMPUTE_UNCERTAINTY}" = "0" ] || [ "${uncertainties}" -eq "${patches}" ]; }
}

check_png_json() {
    local patches
    local pngs
    patches="$(patch_count)"
    pngs="$(count_files "${IMAGE_DIR}" "*.png")"
    [ "${patches}" -gt 0 ] && [ "${pngs}" -eq "${patches}" ] && nonempty_file "${ANNO_FILE}"
}

check_control_height_artifact() {
    "${CONDA_BIN}" run -n "${GEO_ENV}" python -c '
import sys
import rasterio
with rasterio.open(sys.argv[1]) as src:
    assert src.count == 1 and src.crs is not None
' "${CONTROL_HEIGHT_TIF}" >/dev/null 2>&1
}

count_control_heights() {
    "${CONDA_BIN}" run -n "${GEO_ENV}" python -c '
import sys
import numpy as np
import rasterio
with rasterio.open(sys.argv[1]) as src:
    values = src.read(1, masked=True)
print(int(np.count_nonzero(np.ma.filled(values, np.nan) >= 5)))
' "${CONTROL_HEIGHT_TIF}"
}

run_stage() {
    local step="$1" title="$2" check_func="$3" run_func="$4"
    shift 4
    if ! stage_selected "${step}"; then
        if [ "${FROM_STEP_SET}" -eq 1 ]; then
            echo "[SKIP] [${step}/10] ${title} (before --from-step ${FROM_STEP})"
        fi
        return 0
    fi

    local key record="${CACHE_DIR}/stage-${step}.json"
    local outputs=()
    while [ "$1" != "--" ]; do
        outputs+=("$1")
        shift
    done
    shift
    mark_active_stage "${step}" "${title}"
    key="$(pipeline_cache key --value "stage=${step}" "$@")"
    if [ "${FROM_STEP_SET}" -eq 0 ] && [ "${RESUME}" = "1" ] \
        && pipeline_cache matches "${record}" "${key}" "${outputs[@]}"; then
        echo "[SKIP] [${step}/10] ${title}"
        record_stage reused
        active_step=""
        return
    fi

    rm -f -- "${record}"
    echo "[RUN] [${step}/10] ${title}"
    record_stage running
    "${run_func}"
    if ! "${check_func}"; then
        echo "ERROR: stage ${step} output failed validation: ${title}" >&2
        return 1
    fi
    pipeline_cache commit "${record}" "${key}" "${outputs[@]}"
    record_stage success
    active_step=""
}

stage_split_image() {
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
    python utils/split_image.py \
        --input_path "${BOUND_TIF}" \
        --window_size="${WS}" \
        --overlap="${OL}" \
        --down_ratio="${DR}" \
        --output_dir "${PATCH_DIR}"
}

stage_estimate_depth() {
    if [ -d "${DEPTH_WORK_DIR}/depth" ]; then
        find "${DEPTH_WORK_DIR}/depth" -maxdepth 1 -type f -name 'window_*_depth.tif' -delete
    fi
    if [ -d "${DEPTH_WORK_DIR}/uncertainty" ]; then
        find "${DEPTH_WORK_DIR}/uncertainty" -maxdepth 1 -type f -name 'window_*_uncertainty.tif' -delete
    fi
    local uncertainty_args=()
    if [ "${COMPUTE_UNCERTAINTY}" = "1" ]; then
        uncertainty_args+=(--get_conf)
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${CONDA_BIN}" run -n "${DEPTH_ENV}" --no-capture-output \
    python scripts/runtime/run_depth_backend.py \
        --model "${MODEL_NAME}" \
        --img_path "${PATCH_DIR}" \
        --checkpoint "${DEPTH_CHECKPOINT}" \
        --processing_res "${PR}" \
        --output_dir "${DEPTH_WORK_DIR}" \
        --seed 1234 "${uncertainty_args[@]}"
}

stage_prepare_footprint_inputs() {
    if [ -d "${IMAGE_DIR}" ]; then
        find "${IMAGE_DIR}" -maxdepth 1 -type f -name 'window_*.png' -delete
    fi
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
    python tools/prepare_footprint_inputs.py \
        --input-folder "${PATCH_DIR}" \
        --output-folder "${IMAGE_DIR}" \
        --json-output "${ANNO_FILE}"
}

stage_run_mlsbrn() {
    mkdir -p "${MLS_OUTPUT_DIR}"
    local encoding_args=()
    if [ "${MLS_FAST_MASK_ENCODING}" = "1" ]; then
        encoding_args+=(--fast-mask-encoding)
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${MLS_BRN_DIR}:${MLS_BRN_DIR}/bstool":${PYTHONPATH:-} \
    "${CONDA_BIN}" run --no-capture-output -n "${MLSBRN_ENV}" \
    python -m torch.distributed.launch \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --nproc_per_node="${GPUS}" \
        --master_port="${PORT}" \
        "${MLS_BRN_DIR}/tools/bonai/bonai_test.py" \
        "${CONFIG_FILE}" \
        "${CHECKPOINT}" \
        --out "${PKL_FILE}" \
        --city "${CITY}" \
        --test_image_dir "${IMAGE_DIR}" \
        --anno_file "${ANNO_FILE}" \
        "${encoding_args[@]}" \
        --launcher pytorch
}

stage_export_footprint_csv() {
    PYTHONPATH="${MLS_BRN_DIR}:${MLS_BRN_DIR}/bstool":${PYTHONPATH:-} \
    "${CONDA_BIN}" run --no-capture-output -n "${MLSBRN_ENV}" \
    python "${MLS_BRN_DIR}/tools/bonai/bonai_evaluation.py" \
        "${PKL_FILE}" \
        "${MLS_OUTPUT_DIR}" \
        --city "${CITY}" \
        --test_image_dir "${IMAGE_DIR}" \
        --anno_file "${ANNO_FILE}" \
        --skip-metrics
}

stage_convert_footprints() {
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
    python tools/convert_footprints.py \
        --csv-path "${FOOTPRINT_CSV}" \
        --geotiff-folder "${PATCH_DIR}" \
        --output-path "${FOOTPRINT_GEOJSON}" \
        --iou-threshold "${NMS_IOU_THRESHOLD}"
}

stage_extract_photons() {
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
    python PLiDAR_processing/PLiDAR_HeightExtract.py \
        --bound-file "${BOUND_TIF}" \
        --plidar-folder "${ICESAT_PATH}" \
        --save-path "${PHOTON_XLSX}" \
        --seg-len "${SEG_LEN}" \
        --workers "${ATL03_WORKERS}" \
        --worker-threads "${ATL03_WORKER_THREADS}"
}

stage_build_control_height() {
    local empty_args=()
    if [ "${ALLOW_MODEL_FALLBACK}" = "1" ]; then
        empty_args+=(--allow-empty)
    fi
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
    python tools/build_control_height.py \
        --geojson-path "${FOOTPRINT_GEOJSON}" \
        --photon-path "${PHOTON_XLSX}" \
        --reference-tif-path "${BOUND_TIF}" \
        --output-tif-path "${CONTROL_HEIGHT_TIF}" \
        "${empty_args[@]}"
}

stage_optimize_height() {
    # MPLBACKEND=Agg \
    # MPLCONFIGDIR=/tmp/matplotlib-heightanybuilding \
    local rh2ah_args=(
        --work_dir "${DEPTH_WORK_DIR}"
        --window_size="${WS}"
        --overlap="${OL}"
        --down_ratio="${DR}"
        --block_size="${BS}"
        --fusion_mode="${FUSION_MODE}"
        --graph-control-lambda="${GRAPH_CONTROL_LAMBDA}"
        --graph-overlap-lambda="${GRAPH_OVERLAP_LAMBDA}"
        --overlap-mask="${OVERLAP_MASK}"
    )
    if [ "${EFFECTIVE_CONTROL_SOURCE}" = "model" ]; then
        rh2ah_args+=(--metric_input)
    else
        rh2ah_args+=(--ch_img_path "${CONTROL_HEIGHT_TIF}")
    fi
    if [ "${GROUND_FILTER}" = "0" ]; then
        rh2ah_args+=(--skip-ground-filter)
    fi
    if [ "${EXPORT_CALIBRATED_WINDOWS}" = "1" ]; then
        rh2ah_args+=(--export-calibrated)
    fi
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
    python tools/calibrate_heights.py "${rh2ah_args[@]}"
}

stage_assign_building_heights() {
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
    python tools/assign_building_heights.py \
        --geojson-path "${FOOTPRINT_GEOJSON}" \
        --geotiff-path "${HEIGHT_TIF}" \
        --output-path "${FINAL_GEOJSON}" \
        --apply-offset
}

echo "========== HeightAnyBuilding full pipeline =========="
echo "JOB_DIR: ${JOB_DIR}"
echo "Input GeoTIFF: ${BOUND_TIF}"
echo "ICESat-2 dir: ${ICESAT_PATH}"
echo "Depth model: ${MODEL_NAME}"
echo "Depth env: ${DEPTH_ENV}"
echo "Geospatial env: ${GEO_ENV}"
echo "Footprint env: ${MLSBRN_ENV}"
echo "Depth code: ${DEPTH_CODE_PATH}"
echo "Depth checkpoint: ${DEPTH_CHECKPOINT}"
echo "Scale/control source: ${CONTROL_SOURCE}"
if [ "${CONTROL_SOURCE}" = "photon" ]; then
    echo "Photon table: ${PHOTON_XLSX}"
    echo "Minimum controls: ${MIN_PHOTON_CONTROLS}; metric fallback: ${ALLOW_MODEL_FALLBACK}"
fi
echo "ATL03 workers: ${ATL03_WORKERS}; numerical threads per parallel worker: ${ATL03_WORKER_THREADS}"
echo "MLS exact fast mask encoding: ${MLS_FAST_MASK_ENCODING}"
echo "Compute uncertainty: ${COMPUTE_UNCERTAINTY}"
echo "Overlap fusion: ${FUSION_MODE}"
echo "Ground filter: ${GROUND_FILTER}"
echo "Height source: ${HEIGHT_SOURCE}"
echo "Height GeoTIFF: ${HEIGHT_TIF}"
echo "Final GeoJSON: ${FINAL_GEOJSON}"
echo "Selected stages: ${FROM_STEP}-${THROUGH_STEP}"
echo "Resume mode: ${RESUME}"
if [ "${FROM_STEP_SET}" -eq 1 ]; then
    echo "Forced start step: ${FROM_STEP}"
fi

patch_files=(--files "${PATCH_DIR}" 'window_*.tif')
png_files=(--files "${IMAGE_DIR}" 'window_*.png' --file "${ANNO_FILE}")
depth_files=(--files "${DEPTH_WORK_DIR}/depth" 'window_*_depth.tif')
uncertainty_files=(--files "${DEPTH_WORK_DIR}/uncertainty" 'window_*_uncertainty.tif')
grid_values=(--value "window=${WS}" --value "overlap=${OL}" --value "downsample=${DR}")
thread_values=(--value "OMP=${OMP_NUM_THREADS:-}" --value "BLAS=${OPENBLAS_NUM_THREADS:-}"
               --value "MKL=${MKL_NUM_THREADS:-}" --value "NUMEXPR=${NUMEXPR_NUM_THREADS:-}")

run_stage 1 "Split GeoTIFF into patches" check_split stage_split_image "${patch_files[@]}" -- \
    --file "${BOUND_TIF}" "${grid_values[@]}" --value "${GEO_ENV}" \
    --file utils/split_image.py --file utils/window_grid.py --file utils/raster_io.py
run_stage 2 "Convert GeoTIFF patches to PNG/COCO JSON" check_png_json stage_prepare_footprint_inputs "${png_files[@]}" -- \
    "${patch_files[@]}" --value "${GEO_ENV}" --file tools/prepare_footprint_inputs.py
run_stage 3 "Run MLS-BRN footprint inference" true stage_run_mlsbrn --file "${PKL_FILE}" -- \
    "${png_files[@]}" --file "${CONFIG_FILE}" --file "${CHECKPOINT}" --value "${MLSBRN_ENV}" \
    --value "fast-mask=${MLS_FAST_MASK_ENCODING}" --value "workers=${GPUS}:${NNODES}" \
    --files "${MLS_BRN_DIR}/mmdet" '**/*.py' --file tools/mls_mask_encoding.py \
    --file "${MLS_BRN_DIR}/backend_sources.local.json" --file "${MLS_BRN_DIR}/tools/bonai/bonai_test.py"
run_stage 4 "Convert MLS-BRN result.pkl to CSV" true stage_export_footprint_csv --file "${FOOTPRINT_CSV}" -- \
    --file "${PKL_FILE}" "${png_files[@]}" --value "${MLSBRN_ENV}" \
    --files "${MLS_BRN_DIR}/bstool/bstool" '**/*.py' \
    --file "${MLS_BRN_DIR}/tools/bonai/bonai_evaluation.py"
run_stage 5 "Convert footprint_direct.csv to GeoJSON" true stage_convert_footprints --file "${FOOTPRINT_GEOJSON}" -- \
    --file "${FOOTPRINT_CSV}" "${patch_files[@]}" --value "${GEO_ENV}" \
    --value "iou=${NMS_IOU_THRESHOLD}" --file tools/convert_footprints.py
if stage_selected 6; then
    if [ "${CONTROL_SOURCE}" = "model" ]; then
        skip_stage 6 "Extract ICESat-2 ground/building photons (metric model)"
    elif [ "${PHOTON_TABLE_SHARED}" = "1" ]; then
        mark_active_stage 6 "Validate shared photon table"
        if [ ! -s "${PHOTON_XLSX}" ]; then
            echo "ERROR: missing shared photon table: ${PHOTON_XLSX}" >&2
            exit 1
        fi
        skip_stage 6 "Extract ICESat-2 ground/building photons (shared table)"
    else
        run_stage 6 "Extract ICESat-2 ground/building photons" true stage_extract_photons --file "${PHOTON_XLSX}" -- \
            --file "${BOUND_TIF}" --files "${ICESAT_PATH}" '**/*.h5' --files PLiDAR_processing '**/*.py' \
            --value "segment=${SEG_LEN}" --value "threads=${ATL03_WORKER_THREADS}" --value "${GEO_ENV}"
    fi
fi
if stage_selected 7; then
    if [ "${CONTROL_SOURCE}" = "model" ]; then
        skip_stage 7 "Build sparse control-height GeoTIFF (metric model)"
    else
        run_stage 7 "Build sparse control-height GeoTIFF" check_control_height_artifact stage_build_control_height \
            --file "${CONTROL_HEIGHT_TIF}" -- --file "${PHOTON_XLSX}" --file "${FOOTPRINT_GEOJSON}" \
            --file "${BOUND_TIF}" --value "fallback=${ALLOW_MODEL_FALLBACK}" --value "${GEO_ENV}" \
            --file tools/build_control_height.py
    fi
fi
EFFECTIVE_CONTROL_SOURCE="${CONTROL_SOURCE}"
if stage_selected 9 && [ "${CONTROL_SOURCE}" = "photon" ]; then
    mark_active_stage 9 "Check usable photon controls"
    VALID_CONTROL_COUNT="$(count_control_heights)"
    if [ "${VALID_CONTROL_COUNT}" -lt "${MIN_PHOTON_CONTROLS}" ]; then
        if [ "${ALLOW_MODEL_FALLBACK}" = "1" ]; then
            EFFECTIVE_CONTROL_SOURCE="model"
            echo "[FALLBACK] ${VALID_CONTROL_COUNT} usable photon controls (<${MIN_PHOTON_CONTROLS}); preserving SynRS3D metric scale"
        else
            echo "ERROR: only ${VALID_CONTROL_COUNT} usable photon controls; at least ${MIN_PHOTON_CONTROLS} are required." >&2
            exit 1
        fi
    else
        echo "[CONTROL] ${VALID_CONTROL_COUNT} usable photon controls; applying HeightAnyBuilding refinement"
    fi
    active_step=""
fi
depth_outputs=("${depth_files[@]}")
if [ "${COMPUTE_UNCERTAINTY}" = "1" ]; then
    depth_outputs+=("${uncertainty_files[@]}")
fi
run_stage 8 "Run ${MODEL_NAME} depth estimation" check_depth stage_estimate_depth "${depth_outputs[@]}" -- \
    "${patch_files[@]}" --file "${DEPTH_CHECKPOINT}" --value "${MODEL_NAME}:${DEPTH_ENV}" \
    --value "processing-res=${PR}" --value "uncertainty=${COMPUTE_UNCERTAINTY}" \
    --file scripts/runtime/run_depth_backend.py --files "MDE/${MODEL_NAME}" '**/*.py'

height_inputs=("${depth_files[@]}")
height_outputs=(--file "${ABSOLUTE_HEIGHT_TIF}")
if [ "${EXPORT_CALIBRATED_WINDOWS}" = "1" ]; then
    height_outputs+=(--file "${DEPTH_WORK_DIR}/CalibratedWindows/manifest.json"
        --files "${DEPTH_WORK_DIR}/CalibratedWindows" 'x*_y*.npz')
fi
if [ "${FUSION_MODE}" = "uncertainty" ]; then
    height_inputs+=("${uncertainty_files[@]}")
fi
if [ "${EFFECTIVE_CONTROL_SOURCE}" = "photon" ]; then
    height_inputs+=(--file "${CONTROL_HEIGHT_TIF}")
    if [[ "${BS}" =~ ^-1(,-1)*$ ]] || [ "${HEIGHT_SOURCE}" = "coarse_ah" ]; then
        height_outputs+=(--file "${COARSE_AH_TIF}")
    fi
fi
run_stage 9 "Convert relative heights to absolute height" true stage_optimize_height "${height_outputs[@]}" -- \
    "${height_inputs[@]}" "${grid_values[@]}" "${thread_values[@]}" --value "${GEO_ENV}" \
    --value "scale=${EFFECTIVE_CONTROL_SOURCE}" --value "fusion=${FUSION_MODE}" \
    --value "ground-filter=${GROUND_FILTER}" --value "block=${BS}" \
    --value "graph=${GRAPH_CONTROL_LAMBDA}:${GRAPH_OVERLAP_LAMBDA}:${OVERLAP_MASK}" \
    --file tools/calibrate_heights.py --files utils '*.py'
run_stage 10 "Attach absolute heights to building polygons" true stage_assign_building_heights \
    --file "${FINAL_GEOJSON}" -- --file "${FOOTPRINT_GEOJSON}" --file "${HEIGHT_TIF}" \
    --value "${GEO_ENV}" --file tools/assign_building_heights.py

echo "========== Done =========="
if stage_selected 1; then echo "Patches: ${PATCH_DIR}"; fi
if stage_selected 5; then echo "Footprint GeoJSON: ${FOOTPRINT_GEOJSON}"; fi
if stage_selected 7 && [ "${CONTROL_SOURCE}" = "photon" ]; then
    echo "Sparse control height: ${CONTROL_HEIGHT_TIF}"
fi
if stage_selected 8; then echo "Depth work dir: ${DEPTH_WORK_DIR}"; fi
if stage_selected 9; then
    echo "Absolute height: ${ABSOLUTE_HEIGHT_TIF}"
    if [ "${EFFECTIVE_CONTROL_SOURCE}" = "photon" ] && [ -f "${COARSE_AH_TIF}" ]; then
        echo "Coarse AH height: ${COARSE_AH_TIF}"
    fi
fi
if stage_selected 10; then echo "Final building-height GeoJSON: ${FINAL_GEOJSON}"; fi
