#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

INPUT_DIR="${INPUT_DIR:-work_dir/example/tiles_25km2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-work_dir/example/building_height_results}"
ONLY_TILE=""
DRY_RUN=0
CHECK_ONLY=0
FORCE=0
SHOW_PROGRESS="${SHOW_PROGRESS:-0}"
MIN_FREE_GIB="${MIN_FREE_GIB:-10}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
GEO_ENV="${GEO_ENV:-hab-core}"

WS="${WS:-1024}"
OL="${OL:-0.5}"
WINDOW_OPTIONS_SET=0
DR="${DR:-1}"
PR="${PR:-700}"
BS="${BS:--1,-1}"
MODEL_NAME="${MODEL_NAME:-Distill-Any-Depth}"
HEIGHT_SOURCE="${HEIGHT_SOURCE:-absolute}"
CONTROL_SOURCE="${CONTROL_SOURCE:-photon}"
COMPUTE_UNCERTAINTY="${COMPUTE_UNCERTAINTY:-1}"
FUSION_MODE="${FUSION_MODE:-mean}"
GROUND_FILTER="${GROUND_FILTER:-auto}"
GPU_IDS_CSV="${GPU_IDS:-0}"
ATL03_SCOPE="${ATL03_SCOPE:-tile}"
ATL03_START_DATE="${ATL03_START_DATE:-2019-01-01}"
ATL03_END_DATE="${ATL03_END_DATE:-2025-12-31}"
SHARED_ATL03_DIR="${SHARED_ATL03_DIR:-}"
LOCAL_ATL03_DIR="${LOCAL_ATL03_DIR:-}"
SHARED_PHOTON_TABLE="${SHARED_PHOTON_TABLE:-}"
PHOTON_WORKERS="${PHOTON_WORKERS:-16}"
MIN_PHOTON_CONTROLS="${MIN_PHOTON_CONTROLS:-4}"
ALLOW_MODEL_FALLBACK="${ALLOW_MODEL_FALLBACK:-auto}"
MERGE_CITY="${MERGE_CITY:-0}"
CITY_OUTPUT_TIF="${CITY_OUTPUT_TIF:-}"
CITY_OUTPUT_GEOJSON="${CITY_OUTPUT_GEOJSON:-}"
GLOBAL_OPTIMIZE=0
GLOBAL_ONLY=0
GLOBAL_CONTROL_ROOT=""
EXPORT_SYNRS3D_RAW_CITY=0
GLOBAL_PATCH_OPTIMIZE=0
GLOBAL_PATCH_HAB_PRIOR=0
GLOBAL_PATCH_COMPONENTS=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_batch_work_dir.sh [OPTIONS]

Options:
  --input-dir DIR     Input directory (default: work_dir/example/tiles_25km2)
  --output-root DIR   Output root (default: work_dir/example/building_height_results)
  --tile STEM         Run only one tile, e.g. Wuhan_tile_1
  --model NAME        Depth/height model (default: Distill-Any-Depth)
  --window-size N     Inference window size in pixels (default: 1024)
  --overlap RATE      Inference window overlap fraction (default: 0.5)
  --control-source S  photon or model (default: photon)
  --uncertainty       Enable augmented inference and uncertainty output
  --no-uncertainty    Run exactly one model forward pass per patch
  --fusion MODE       mean or uncertainty (default: mean)
  --ground-filter     Enable RH2AH local-minimum ground filtering
  --no-ground-filter  Skip RH2AH ground filtering (default for SynRS3D)
  --gpus IDS          Comma-separated physical GPU IDs (default: 0)
  --atl03-scope MODE  tile or shared (default: tile)
  --atl03-start-date DATE  ATL03 start date (default: 2019-01-01)
  --atl03-end-date DATE    ATL03 end date (default: 2025-12-31)
  --shared-atl03-dir DIR   Root for request-specific downloads (default: OUTPUT_ROOT/_shared_atl03)
  --local-atl03-dir DIR    Use caller-selected local HDF5 granules without downloading
  --shared-photon-table FILE  Name/location template for request-specific city photon tables
  --photon-workers N   Parallel ATL03 extraction workers (default: 16)
  --min-photon-controls N  Minimum controls required for refinement (default: 4)
  --photon-fallback    Preserve SynRS3D metric scale when controls are insufficient (default for SynRS3D)
  --strict-photon-controls  Fail instead of falling back when controls are insufficient
  --merge-city        Merge all successful tile outputs into one city raster and GeoJSON
  --city-output-tif FILE  City UInt16 1 m GeoTIFF (implies --merge-city)
  --city-output-geojson FILE  City building GeoJSON (implies --merge-city)
  --global-optimize   Infer seam windows and jointly refine retained tile heights
  --export-synrs3d-raw-city  Mosaic raw SynRS3D patches into city UInt16 TIFF and building GeoJSON
  --global-patch-optimize   Refine calibrated height windows with independent SynRS3D seam predictions
  --global-patch-hab-prior  Use the legacy raw SynRS3D optimizer with priors from per-tile HAB
  --global-patch-components  Optimize disconnected regions separately (implies patch optimization)
  --global-control-root DIR  Sparse controls as DIR/TILE/sparse_control_height.tif
  --global-only       Use completed HAB tiles; never rerun tiles or ATL03 extraction
  --show-progress     Stream both workers' detailed output while retaining per-tile logs
  --force             Re-run completed pipeline stages and overwrite stable result copies
  --dry-run           Print the tile and output plan without loading models or writing outputs
  --check             Check the required runtime and inputs without starting the job
  --help              Show this help
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input-dir)
            [ "$#" -ge 2 ] || { echo "ERROR: --input-dir requires a value" >&2; exit 2; }
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-root)
            [ "$#" -ge 2 ] || { echo "ERROR: --output-root requires a value" >&2; exit 2; }
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --tile)
            [ "$#" -ge 2 ] || { echo "ERROR: --tile requires a value" >&2; exit 2; }
            ONLY_TILE="${2%.tif}"
            shift 2
            ;;
        --model)
            [ "$#" -ge 2 ] || { echo "ERROR: --model requires a value" >&2; exit 2; }
            MODEL_NAME="$2"
            shift 2
            ;;
        --window-size|--overlap)
            [ "$#" -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            if [ "$1" = "--window-size" ]; then WS="$2"; else OL="$2"; fi
            WINDOW_OPTIONS_SET=1
            shift 2
            ;;
        --control-source)
            [ "$#" -ge 2 ] || { echo "ERROR: --control-source requires a value" >&2; exit 2; }
            CONTROL_SOURCE="$2"
            shift 2
            ;;
        --uncertainty)
            COMPUTE_UNCERTAINTY=1
            shift
            ;;
        --no-uncertainty)
            COMPUTE_UNCERTAINTY=0
            shift
            ;;
        --fusion)
            [ "$#" -ge 2 ] || { echo "ERROR: --fusion requires a value" >&2; exit 2; }
            FUSION_MODE="$2"
            shift 2
            ;;
        --ground-filter)
            GROUND_FILTER=1
            shift
            ;;
        --no-ground-filter)
            GROUND_FILTER=0
            shift
            ;;
        --gpus)
            [ "$#" -ge 2 ] || { echo "ERROR: --gpus requires a value" >&2; exit 2; }
            GPU_IDS_CSV="$2"
            shift 2
            ;;
        --atl03-scope)
            [ "$#" -ge 2 ] || { echo "ERROR: --atl03-scope requires a value" >&2; exit 2; }
            ATL03_SCOPE="$2"
            shift 2
            ;;
        --atl03-start-date)
            [ "$#" -ge 2 ] || { echo "ERROR: --atl03-start-date requires a value" >&2; exit 2; }
            ATL03_START_DATE="$2"
            shift 2
            ;;
        --atl03-end-date)
            [ "$#" -ge 2 ] || { echo "ERROR: --atl03-end-date requires a value" >&2; exit 2; }
            ATL03_END_DATE="$2"
            shift 2
            ;;
        --shared-atl03-dir)
            [ "$#" -ge 2 ] || { echo "ERROR: --shared-atl03-dir requires a value" >&2; exit 2; }
            SHARED_ATL03_DIR="$2"
            shift 2
            ;;
        --local-atl03-dir)
            [ "$#" -ge 2 ] || { echo "ERROR: --local-atl03-dir requires a value" >&2; exit 2; }
            LOCAL_ATL03_DIR="$2"
            shift 2
            ;;
        --shared-photon-table)
            [ "$#" -ge 2 ] || { echo "ERROR: --shared-photon-table requires a value" >&2; exit 2; }
            SHARED_PHOTON_TABLE="$2"
            shift 2
            ;;
        --photon-workers)
            [ "$#" -ge 2 ] || { echo "ERROR: --photon-workers requires a value" >&2; exit 2; }
            PHOTON_WORKERS="$2"
            shift 2
            ;;
        --min-photon-controls)
            [ "$#" -ge 2 ] || { echo "ERROR: --min-photon-controls requires a value" >&2; exit 2; }
            MIN_PHOTON_CONTROLS="$2"
            shift 2
            ;;
        --photon-fallback)
            ALLOW_MODEL_FALLBACK=1
            shift
            ;;
        --strict-photon-controls)
            ALLOW_MODEL_FALLBACK=0
            shift
            ;;
        --merge-city)
            MERGE_CITY=1
            shift
            ;;
        --city-output-tif)
            [ "$#" -ge 2 ] || { echo "ERROR: --city-output-tif requires a value" >&2; exit 2; }
            CITY_OUTPUT_TIF="$2"
            MERGE_CITY=1
            shift 2
            ;;
        --city-output-geojson)
            [ "$#" -ge 2 ] || { echo "ERROR: --city-output-geojson requires a value" >&2; exit 2; }
            CITY_OUTPUT_GEOJSON="$2"
            MERGE_CITY=1
            shift 2
            ;;
        --global-optimize)
            GLOBAL_OPTIMIZE=1
            shift
            ;;
        --export-synrs3d-raw-city)
            EXPORT_SYNRS3D_RAW_CITY=1
            shift
            ;;
        --global-patch-optimize)
            GLOBAL_PATCH_OPTIMIZE=1
            shift
            ;;
        --global-patch-hab-prior)
            GLOBAL_PATCH_OPTIMIZE=1
            GLOBAL_PATCH_HAB_PRIOR=1
            shift
            ;;
        --global-patch-components)
            GLOBAL_PATCH_OPTIMIZE=1
            GLOBAL_PATCH_COMPONENTS=1
            shift
            ;;
        --global-control-root)
            [ "$#" -ge 2 ] || { echo "ERROR: --global-control-root requires a value" >&2; exit 2; }
            GLOBAL_CONTROL_ROOT="$2"
            shift 2
            ;;
        --global-only)
            GLOBAL_ONLY=1
            shift
            ;;
        --show-progress)
            SHOW_PROGRESS=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --check)
            CHECK_ONLY=1
            shift
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
if [ "${DRY_RUN}" -eq 1 ] && [ "${CHECK_ONLY}" -eq 1 ]; then
    echo "ERROR: choose either --dry-run or --check." >&2
    exit 2
fi

window_options="$("${PYTHON_BIN}" - "${WS}" "${OL}" <<'PYWINDOW'
import sys
from utils.window_grid import window_step
try:
    size, overlap = int(sys.argv[1]), float(sys.argv[2])
    window_step(size, overlap)
except ValueError as error:
    raise SystemExit(f"ERROR: invalid --window-size or --overlap: {error}")
print(size, overlap)
PYWINDOW
)" || exit 2
if [ "${WINDOW_OPTIONS_SET}" -eq 1 ]; then
    read -r WS OL <<< "${window_options}"
fi

if ! [[ "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] || [ "${MIN_FREE_GIB}" -lt 1 ]; then
    echo "ERROR: MIN_FREE_GIB must be a positive integer" >&2
    exit 2
fi
if [ "${COMPUTE_UNCERTAINTY}" != "0" ] && [ "${COMPUTE_UNCERTAINTY}" != "1" ]; then
    echo "ERROR: COMPUTE_UNCERTAINTY must be 0 or 1" >&2
    exit 2
fi
if [ "${SHOW_PROGRESS}" != "0" ] && [ "${SHOW_PROGRESS}" != "1" ]; then
    echo "ERROR: SHOW_PROGRESS must be 0 or 1" >&2
    exit 2
fi
if [ "${MERGE_CITY}" != "0" ] && [ "${MERGE_CITY}" != "1" ]; then
    echo "ERROR: MERGE_CITY must be 0 or 1" >&2
    exit 2
fi
if [ "${GROUND_FILTER}" != "auto" ] && [ "${GROUND_FILTER}" != "0" ] && [ "${GROUND_FILTER}" != "1" ]; then
    echo "ERROR: GROUND_FILTER must be auto, 0, or 1" >&2
    exit 2
fi
if ! [[ "${PHOTON_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --photon-workers must be a positive integer" >&2
    exit 2
fi
if ! [[ "${MIN_PHOTON_CONTROLS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --min-photon-controls must be a positive integer" >&2
    exit 2
fi
if [ "${ALLOW_MODEL_FALLBACK}" = "auto" ]; then
    if [ "${MODEL_NAME}" = "SynRS3D" ]; then
        ALLOW_MODEL_FALLBACK=1
    else
        ALLOW_MODEL_FALLBACK=0
    fi
fi
if [ "${GROUND_FILTER}" = "auto" ]; then
    if [ "${MODEL_NAME}" = "SynRS3D" ]; then
        GROUND_FILTER=0
    else
        GROUND_FILTER=1
    fi
fi
if [ "${ALLOW_MODEL_FALLBACK}" != "0" ] && [ "${ALLOW_MODEL_FALLBACK}" != "1" ]; then
    echo "ERROR: ALLOW_MODEL_FALLBACK must be auto, 0, or 1" >&2
    exit 2
fi
case "${CONTROL_SOURCE}" in
    photon|model) ;;
    *) echo "ERROR: --control-source must be photon or model" >&2; exit 2 ;;
esac
case "${FUSION_MODE}" in
    mean|uncertainty) ;;
    *) echo "ERROR: --fusion must be mean or uncertainty" >&2; exit 2 ;;
esac
case "${ATL03_SCOPE}" in
    tile|shared) ;;
    *) echo "ERROR: --atl03-scope must be tile or shared" >&2; exit 2 ;;
esac
if [ "${FUSION_MODE}" = "uncertainty" ] && [ "${COMPUTE_UNCERTAINTY}" != "1" ]; then
    echo "ERROR: uncertainty fusion requires uncertainty inference" >&2
    exit 2
fi
if [ "${CONTROL_SOURCE}" = "model" ] && [ "${MODEL_NAME}" != "SynRS3D" ]; then
    echo "ERROR: --control-source model currently requires --model SynRS3D" >&2
    exit 2
fi
if [ "${CONTROL_SOURCE}" = "model" ] && [ "${HEIGHT_SOURCE}" != "absolute" ]; then
    echo "ERROR: model-scale optimization requires HEIGHT_SOURCE=absolute" >&2
    exit 2
fi
if [ "${ALLOW_MODEL_FALLBACK}" = "1" ] && [ "${MODEL_NAME}" != "SynRS3D" ]; then
    echo "ERROR: photon fallback requires --model SynRS3D" >&2
    exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "ERROR: --gpus must contain at least one GPU ID" >&2
    exit 2
fi
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
    if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    if [ -n "${SEEN_GPU_IDS[${gpu_id}]:-}" ]; then
        echo "ERROR: duplicate GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    SEEN_GPU_IDS["${gpu_id}"]=1
done

INPUT_DIR="$(readlink -m "${INPUT_DIR}")"
OUTPUT_ROOT="$(readlink -m "${OUTPUT_ROOT}")"
if [ -z "${CITY_OUTPUT_TIF}" ]; then
    CITY_OUTPUT_TIF="${OUTPUT_ROOT}/city_absolute_height_1m_uint16.tif"
else
    CITY_OUTPUT_TIF="$(readlink -m "${CITY_OUTPUT_TIF}")"
fi
if [ -z "${CITY_OUTPUT_GEOJSON}" ]; then
    CITY_OUTPUT_GEOJSON="${OUTPUT_ROOT}/city_building_heights.geojson"
else
    CITY_OUTPUT_GEOJSON="$(readlink -m "${CITY_OUTPUT_GEOJSON}")"
fi
if [ -z "${SHARED_ATL03_DIR}" ]; then
    SHARED_ATL03_DIR="${OUTPUT_ROOT}/_shared_atl03"
else
    SHARED_ATL03_DIR="$(readlink -m "${SHARED_ATL03_DIR}")"
fi
if [ -n "${LOCAL_ATL03_DIR}" ]; then
    LOCAL_ATL03_DIR="$(readlink -m "${LOCAL_ATL03_DIR}")"
    if [ ! -d "${LOCAL_ATL03_DIR}" ] || ! find -L "${LOCAL_ATL03_DIR}" -maxdepth 1 -type f -name '*.h5' -size +0c -print -quit | grep -q .; then
        echo "ERROR: local ATL03 directory must contain HDF5 granules: ${LOCAL_ATL03_DIR}" >&2
        exit 2
    fi
fi
if [ -z "${SHARED_PHOTON_TABLE}" ]; then
    SHARED_PHOTON_TABLE="${SHARED_ATL03_DIR}/city_photons.csv.gz"
else
    SHARED_PHOTON_TABLE="$(readlink -m "${SHARED_PHOTON_TABLE}")"
fi

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: input directory does not exist: ${INPUT_DIR}" >&2
    exit 1
fi

mapfile -d '' ALL_TIFS < <(
    find "${INPUT_DIR}" -maxdepth 1 -type f \( -iname '*.tif' -o -iname '*.tiff' \) -print0 | sort -zV
)

TIFS=()
for tif_path in "${ALL_TIFS[@]}"; do
    tile_stem="$(basename "${tif_path}")"
    tile_stem="${tile_stem%.*}"
    if [ -z "${ONLY_TILE}" ] || [ "${tile_stem}" = "${ONLY_TILE}" ]; then
        TIFS+=("${tif_path}")
    fi
done

if [ "${#TIFS[@]}" -eq 0 ]; then
    echo "ERROR: no matching GeoTIFF inputs found" >&2
    exit 1
fi
if [ "${MERGE_CITY}" -eq 1 ] && [ -n "${ONLY_TILE}" ]; then
    echo "ERROR: --merge-city cannot be combined with --tile; run the complete city batch" >&2
    exit 2
fi

if [ "${GLOBAL_OPTIMIZE}" -eq 1 ] && [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ]; then
    echo "ERROR: tile-level --global-optimize and --global-patch-optimize are mutually exclusive" >&2
    exit 2
fi
if [ "${EXPORT_SYNRS3D_RAW_CITY}" -eq 1 ] || [ "${GLOBAL_PATCH_HAB_PRIOR}" -eq 1 ]; then
    if [ "${MODEL_NAME}" != "SynRS3D" ] || [ "${WS}" != "1024" ] || [ "${OL}" != "0.5" ] || [ "${DR}" != "1" ] \
        || [ "${COMPUTE_UNCERTAINTY}" != "0" ] || [ "${FUSION_MODE}" != "mean" ] || [ "${GROUND_FILTER}" != "0" ]; then
        echo "ERROR: patch outputs require SynRS3D, WS=1024, OL=0.5, DR=1, --no-uncertainty, --fusion mean, --no-ground-filter" >&2
        exit 2
    fi
    if [ -n "${ONLY_TILE}" ] || [ "${GLOBAL_ONLY}" -eq 1 ]; then
        echo "ERROR: patch city outputs require the complete normal tile batch" >&2
        exit 2
    fi
fi
EXPORT_CALIBRATED_WINDOWS=0
if [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ] && [ "${GLOBAL_PATCH_HAB_PRIOR}" -eq 0 ]; then
    if [ "${DR}" != "1" ] || [ "${FUSION_MODE}" != "mean" ] || [ "${HEIGHT_SOURCE}" != "absolute" ] \
        || [[ ! "${BS}" =~ ^-1(,-1)*$ ]]; then
        echo "ERROR: calibrated patch optimization requires a 1m output grid, whole-tile RH2AH, mean fusion and absolute heights" >&2
        exit 2
    fi
    if [ "${#TIFS[@]}" -lt 2 ] || [ -n "${ONLY_TILE}" ] || [ "${GLOBAL_ONLY}" -eq 1 ]; then
        echo "ERROR: calibrated patch optimization requires a complete batch with at least two tiles" >&2
        exit 2
    fi
    EXPORT_CALIBRATED_WINDOWS=1
fi
if [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ] && [ "${CONTROL_SOURCE}" != "photon" ]; then
    echo "ERROR: --global-patch-optimize requires --control-source photon" >&2
    exit 2
fi
if [ "${GLOBAL_ONLY}" -eq 1 ] && [ "${GLOBAL_OPTIMIZE}" -ne 1 ]; then
    echo "ERROR: --global-only requires --global-optimize" >&2
    exit 2
fi
if [ "${GLOBAL_OPTIMIZE}" -eq 1 ] && [ "${MODEL_NAME}" != "SynRS3D" ]; then
    echo "ERROR: --global-optimize requires --model SynRS3D" >&2
    exit 2
fi
if [ "${GLOBAL_OPTIMIZE}" -eq 1 ] && [ -n "${ONLY_TILE}" ]; then
    echo "ERROR: --global-optimize requires all city tiles" >&2
    exit 2
fi
if [ "${GLOBAL_ONLY}" -eq 1 ] && [ "${MERGE_CITY}" -eq 1 ]; then
    echo "ERROR: --global-only produces its own city result; omit --merge-city" >&2
    exit 2
fi
if [ -n "${GLOBAL_CONTROL_ROOT}" ] && [ "${GLOBAL_OPTIMIZE}" -ne 1 ]; then
    echo "ERROR: --global-control-root requires --global-optimize" >&2
    exit 2
fi
if [ -z "${GLOBAL_CONTROL_ROOT}" ]; then
    GLOBAL_CONTROL_ROOT="${OUTPUT_ROOT}"
else
    GLOBAL_CONTROL_ROOT="$(readlink -m "${GLOBAL_CONTROL_ROOT}")"
fi

require_file() {
    if [ ! -s "$1" ]; then
        echo "ERROR: required file is missing or empty: $1" >&2
        return 1
    fi
}

preflight() {
    [ -x "${CONDA_BIN}" ] || { echo "ERROR: conda executable not found: ${CONDA_BIN}" >&2; return 1; }
    [ -x "${PYTHON_BIN}" ] || { echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2; return 1; }
    if [ "${MERGE_CITY}" -eq 1 ]; then
        require_file "tools/merge_city_height_results.py" || return 1
    fi
    if [ "${EXPORT_SYNRS3D_RAW_CITY}" -eq 1 ] || [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ]; then
        require_file "tools/global_patch_height.py" || return 1
        require_file "tools/infer_global_seam_windows.py" || return 1
        require_file "tools/assign_city_building_heights.py" || return 1
    fi
    local seam_args=()
    if [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ]; then seam_args+=(--seam); fi
    CONDA_BIN="${CONDA_BIN}" GEO_ENV="${GEO_ENV}" "${PYTHON_BIN}" scripts/runtime/check_runtime.py \
        --model "${MODEL_NAME}" --gpus "${GPU_IDS_CSV}" "${seam_args[@]}" || return 1

    "${CONDA_BIN}" run -n "${GEO_ENV}" python -c '
import sys
import rasterio
for filename in sys.argv[2:]:
    with rasterio.open(filename) as src:
        actual = (src.width, src.height, src.count, tuple(src.dtypes), str(src.crs))
        if src.width < int(sys.argv[1]) or src.height < int(sys.argv[1]):
            raise SystemExit(f"input is smaller than the window size: {filename}: {actual}")
        if src.count != 3 or tuple(src.dtypes) != ("uint8", "uint8", "uint8") or src.crs is None:
            raise SystemExit(f"invalid input metadata: {filename}: {actual}")
print(f"validated_input_count={len(sys.argv)-2}")
' "${WS}" "${TIFS[@]}" || return 1
}

global_preflight() {
    [ -x "${CONDA_BIN}" ] || { echo "ERROR: conda executable not found: ${CONDA_BIN}" >&2; return 1; }
    require_file "tools/global_optimize_height.py" || return 1
    require_file "tools/infer_global_seam_windows.py" || return 1
    require_file "${DEPTH_CHECKPOINT}" || return 1
    [ -d "${GLOBAL_CONTROL_ROOT}" ] || { echo "ERROR: global control root does not exist: ${GLOBAL_CONTROL_ROOT}" >&2; return 1; }
    local gpu_id
    for gpu_id in "${GPU_IDS[@]}"; do
        nvidia-smi --query-gpu=index --format=csv,noheader | grep -qx "${gpu_id}" || {
            echo "ERROR: GPU ${gpu_id} is unavailable" >&2
            return 1
        }
    done
    "${CONDA_BIN}" run -n "${GEO_ENV}" python -c "import geopandas, numpy, rasterio, scipy, shapely" >/dev/null || return 1
    NO_ALBUMENTATIONS_UPDATE=1 "${CONDA_BIN}" run -n synrs3d python -c "import cv2, numpy, torch, rasterio, albumentations; assert torch.cuda.is_available()" >/dev/null || return 1
}

run_global_patch_optimization() {
    local extra_args=()
    if [ "${EXPORT_SYNRS3D_RAW_CITY}" -eq 1 ]; then extra_args+=(--export-raw); fi
    if [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ]; then extra_args+=(--global-optimize); fi
    if [ "${GLOBAL_PATCH_HAB_PRIOR}" -eq 1 ]; then extra_args+=(--hab-prior); fi
    if [ "${GLOBAL_PATCH_COMPONENTS}" -eq 1 ]; then extra_args+=(--by-component); fi
    if [ "${EXPORT_CALIBRATED_WINDOWS}" -eq 1 ]; then
        extra_args+=(--calibrated-work-directory "${MODEL_NAME}-windowsize-${WS}-overlap-${OL}-downratio-${DR}-depth"
                     --window-size "${WS}" --overlap "${OL}")
    fi
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
        python tools/global_patch_height.py \
        --input-dir "${INPUT_DIR}" --results-root "${OUTPUT_ROOT}" \
        --cache-root "${OUTPUT_ROOT}/_global_patch_cache" \
        --expected-tiles "${#TIFS[@]}" \
        --gpus "${GPU_IDS_CSV}" --conda-bin "${CONDA_BIN}" \
        --checkpoint "${SEAM_CHECKPOINT}" \
        "${extra_args[@]}" "$@"
}

archive_synrs3d_patches() {
    local job_dir="$1" tile_stem="$2" tif_path="$3"
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
        python tools/global_patch_height.py --archive-tile \
        --input-tif "${tif_path}" --tile-name "${tile_stem}" \
        --depth-dir "${job_dir}/${MODEL_NAME}-windowsize-${WS}-overlap-${OL}-downratio-${DR}-depth" \
        --cache-root "${OUTPUT_ROOT}/_global_patch_cache"
}

run_global_optimization() {
    "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
        python tools/global_optimize_height.py \
        --input-dir "${INPUT_DIR}" \
        --results-root "${OUTPUT_ROOT}" \
        --control-root "${GLOBAL_CONTROL_ROOT}" \
        --gpus "${GPU_IDS_CSV}" \
        --conda-bin "${CONDA_BIN}" \
        --checkpoint "${DEPTH_CHECKPOINT}" \
        "$@"
}

write_status() {
    local gpu_id="${10:-}"
    local gpu_args=()
    if [ -n "${gpu_id}" ]; then gpu_args=(--gpu-id "${gpu_id}"); fi
    "${PYTHON_BIN}" scripts/runtime/run_status.py tile "${@:1:9}" "${gpu_args[@]}" \
        --model "${MODEL_NAME}" --control-source "${CONTROL_SOURCE}" \
        --compute-uncertainty "${COMPUTE_UNCERTAINTY}" --fusion-mode "${FUSION_MODE}" \
        --ground-filter "${GROUND_FILTER}"
}

rebuild_summary() {
    "${PYTHON_BIN}" scripts/runtime/run_status.py summary "${OUTPUT_ROOT}" "${TIFS[@]}"
}

validate_full_tile() {
    local job_dir="$1"
    local input_tif="$2"
    local tile_stem="$3"
    local atl03_dir="$4"
    local validation_args=(
        --job-dir "${job_dir}"
        --input-tif "${input_tif}"
        --tile-stem "${tile_stem}"
        --window-size "${WS}"
        --overlap "${OL}"
        --down-ratio "${DR}"
        --model-name "${MODEL_NAME}"
        --control-source "${CONTROL_SOURCE}"
    )
    if [ "${CONTROL_SOURCE}" = "photon" ]; then
        validation_args+=(--min-controls "${MIN_PHOTON_CONTROLS}")
        if [ "${ALLOW_MODEL_FALLBACK}" = "1" ]; then
            validation_args+=(--allow-model-fallback)
        fi
        validation_args+=(--atl03-dir "${atl03_dir}")
        if [ "${ATL03_SCOPE}" = "shared" ]; then
            validation_args+=(--photon-table "${SHARED_PHOTON_TABLE}")
        fi
    fi
    if [ "${COMPUTE_UNCERTAINTY}" = "1" ]; then
        validation_args+=(--compute-uncertainty)
    else
        validation_args+=(--no-uncertainty)
    fi
    "${CONDA_BIN}" run -n "${GEO_ENV}" python tools/validate_batch_tile.py "${validation_args[@]}"
}

validate_retained_tile() {
    local job_dir="$1"
    local input_tif="$2"
    local tile_stem="$3"
    local atl03_dir="$4"
    local validation_args=(
        --job-dir "${job_dir}"
        --input-tif "${input_tif}"
        --tile-stem "${tile_stem}"
        --model-name "${MODEL_NAME}"
        --control-source "${CONTROL_SOURCE}"
        --retained-only
    )
    if [ "${CONTROL_SOURCE}" = "photon" ]; then
        validation_args+=(--min-controls "${MIN_PHOTON_CONTROLS}")
        if [ "${ALLOW_MODEL_FALLBACK}" = "1" ]; then
            validation_args+=(--allow-model-fallback)
        fi
        validation_args+=(--atl03-dir "${atl03_dir}")
        if [ "${ATL03_SCOPE}" = "shared" ]; then
            validation_args+=(--photon-table "${SHARED_PHOTON_TABLE}")
        fi
    fi
    "${CONDA_BIN}" run -n "${GEO_ENV}" python tools/validate_batch_tile.py "${validation_args[@]}"
}

echo "========== Building-height batch =========="
echo "Input directory: ${INPUT_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Selected tiles: ${#TIFS[@]}"
echo "Model: ${MODEL_NAME}; GPUs: ${GPU_IDS_CSV}; WS/OL/DR/PR: ${WS}/${OL}/${DR}/${PR}"
echo "Ground filter: ${GROUND_FILTER}"
echo "Scale/control source: ${CONTROL_SOURCE}"
echo "Uncertainty: ${COMPUTE_UNCERTAINTY}; fusion: ${FUSION_MODE}; live progress: ${SHOW_PROGRESS}"
if [ "${GLOBAL_ONLY}" -eq 1 ]; then
    echo "ATL03 and per-tile HAB: skipped; using completed tile outputs"
elif [ "${CONTROL_SOURCE}" = "photon" ]; then
    echo "ATL03: V007, ${ATL03_START_DATE} through ${ATL03_END_DATE}, Harmony subset; scope=${ATL03_SCOPE}"
    if [ "${ATL03_SCOPE}" = "shared" ]; then
        echo "Shared ATL03 directory: ${SHARED_ATL03_DIR}"
        echo "Shared photon table: ${SHARED_PHOTON_TABLE}; extraction workers: ${PHOTON_WORKERS}"
    fi
    echo "Photon controls: minimum=${MIN_PHOTON_CONTROLS}; metric fallback=${ALLOW_MODEL_FALLBACK}"
else
    echo "ATL03: disabled; SynRS3D metric scale is retained"
fi
echo "Parallel workers: ${#GPU_IDS[@]} (one per GPU)"
echo "Failure policy: record and continue"
if [ "${MERGE_CITY}" -eq 1 ]; then
    echo "City merge: enabled after all ${#TIFS[@]} tiles succeed"
    echo "City height raster: ${CITY_OUTPUT_TIF}"
    echo "City building GeoJSON: ${CITY_OUTPUT_GEOJSON}"
else
    echo "City merge: disabled"
fi

if [ "${GLOBAL_OPTIMIZE}" -eq 1 ]; then
    echo "Global optimization: enabled; controls=${GLOBAL_CONTROL_ROOT}; existing tiles only=${GLOBAL_ONLY}"
fi
if [ "${EXPORT_SYNRS3D_RAW_CITY}" -eq 1 ] || [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ]; then
    echo "Raw SynRS3D city export=${EXPORT_SYNRS3D_RAW_CITY}; global patch optimization=${GLOBAL_PATCH_OPTIMIZE}; HAB patch prior=${GLOBAL_PATCH_HAB_PRIOR}"
fi
[ -x "${PYTHON_BIN}" ] || { echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }
model_config="$("${PYTHON_BIN}" scripts/runtime/model_config.py "${MODEL_NAME}")" || exit 2
mapfile -t model_values <<< "${model_config}"
DEPTH_ENV="${model_values[1]}"
DEPTH_CHECKPOINT="${model_values[2]}"
echo "Conda environments: geospatial=${GEO_ENV}; depth=${DEPTH_ENV}; footprints=${MLSBRN_ENV:-mlsbrn}"
SEAM_CHECKPOINT="${DEPTH_CHECKPOINT}"
if [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ] && [ "${MODEL_NAME}" != "SynRS3D" ]; then
    seam_config="$("${PYTHON_BIN}" scripts/runtime/model_config.py SynRS3D)" || exit 2
    mapfile -t seam_values <<< "${seam_config}"
    SEAM_CHECKPOINT="${seam_values[2]}"
fi
if [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ]; then
    echo "Cross-tile model plan: interior=${MODEL_NAME}; seam=SynRS3D"
fi
if [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ] || [ "${GLOBAL_OPTIMIZE}" -eq 1 ]; then
    echo "Seam inference env: synrs3d"
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "========== Dry-run tile plan =========="
    if [ "${GLOBAL_ONLY}" -eq 1 ]; then
        echo "Existing HAB tiles: ${#TIFS[@]}; no per-tile jobs or ATL03 extraction"
    else
        for tile_index in "${!TIFS[@]}"; do
            tif_path="${TIFS[${tile_index}]}"
            tile_stem="$(basename "${tif_path}")"
            tile_stem="${tile_stem%.*}"
            gpu_id="${GPU_IDS[$((tile_index % ${#GPU_IDS[@]}))]}"
            echo "${tile_stem}: GPU ${gpu_id}; ${tif_path} -> ${OUTPUT_ROOT}/${tile_stem}"
        done
    fi
    if [ "${MERGE_CITY}" -eq 1 ]; then
        echo "City merge plan: ${#TIFS[@]} tiles -> ${CITY_OUTPUT_TIF}; ${CITY_OUTPUT_GEOJSON}"
    fi
    echo "Dry-run complete; no output files were created."
    exit 0
fi

if [ "${GLOBAL_ONLY}" -eq 1 ]; then
    global_preflight || exit 1
else
    preflight || exit 1
fi

if [ "${CHECK_ONLY}" -eq 1 ]; then
    echo "Runtime check passed for the selected batch."
    exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/.batch.lock"
if ! flock -n 9; then
    echo "ERROR: another batch is already using output root: ${OUTPUT_ROOT}" >&2
    exit 1
fi
batch_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "${GLOBAL_ONLY}" -eq 1 ]; then
    run_global_optimization
    exit $?
fi

download_atl03() {
    CONDA_BIN="${CONDA_BIN}" GEO_ENV="${GEO_ENV}" \
        START_DATE="${ATL03_START_DATE}" END_DATE="${ATL03_END_DATE}" \
        VERSION="007" SHORT_NAME=ATL03 DOWNLOAD=true HARMONY_SUBSET=true \
        DOWNLOAD_FULL_GRANULE=false OVERWRITE=false \
        "${PYTHON_BIN}" scripts/runtime/pipeline_cache.py download-atl03 "$@"
}

prepare_shared_photon_table() {
    local key
    key="$("${PYTHON_BIN}" scripts/runtime/pipeline_cache.py key --value shared-photons-v1 \
        --files "${INPUT_DIR}" '*.[tT][iI][fF]' --files "${INPUT_DIR}" '*.[tT][iI][fF][fF]' \
        --files "${SHARED_ATL03_DIR}" '*.h5' --files PLiDAR_processing '**/*.py' \
        --file tools/extract_atl03_parallel.py)" || return 1
    local parent
    parent="$(dirname "${SHARED_PHOTON_TABLE}")/photons-${key:0:20}"
    SHARED_PHOTON_TABLE="${parent}/$(basename "${SHARED_PHOTON_TABLE}")"
    mkdir -p "${parent}" || return 1
    local photon_log="${parent}/photon_extract.log"
    local record="${parent}/.complete.json"
    exec 7>"${parent}/.lock"
    flock 7 || return 1
    if "${PYTHON_BIN}" scripts/runtime/pipeline_cache.py matches "${record}" "${key}" --file "${SHARED_PHOTON_TABLE}"; then
        echo "[REUSE] City-wide photon table: ${SHARED_PHOTON_TABLE}"
    else
        rm -f -- "${record}"
        echo "[RUN] Extract city-wide ATL03 photons with ${PHOTON_WORKERS} CPU workers"
        if ! "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" python tools/extract_atl03_parallel.py \
            --bound-dir "${INPUT_DIR}" --atl03-dir "${SHARED_ATL03_DIR}" \
            --output "${SHARED_PHOTON_TABLE}" --workers "${PHOTON_WORKERS}" >"${photon_log}" 2>&1 \
            || ! "${PYTHON_BIN}" scripts/runtime/pipeline_cache.py commit "${record}" "${key}" --file "${SHARED_PHOTON_TABLE}"; then
            echo "ERROR: city-wide photon extraction failed; see ${photon_log}" >&2
            return 1
        fi
        echo "[OK] City-wide photon table ready: ${SHARED_PHOTON_TABLE}"
    fi
    exec 7>&-
}

if [ "${CONTROL_SOURCE}" = "photon" ] && [ "${ATL03_SCOPE}" = "shared" ]; then
    if [ -n "${LOCAL_ATL03_DIR}" ]; then
        SHARED_ATL03_DIR="${LOCAL_ATL03_DIR}"
        echo "[USE] Shared local ATL03: ${SHARED_ATL03_DIR}"
    else
        SHARED_ATL03_DIR="$(download_atl03 "${SHARED_ATL03_DIR}" --tif-dir "${INPUT_DIR}")" || exit 1
    fi
    SHARED_ATL03_DIR="$("${PYTHON_BIN}" scripts/runtime/pipeline_cache.py select-atl03 "${OUTPUT_ROOT}" \
        "${SHARED_ATL03_DIR}" --directory "${SHARED_ATL03_DIR}")" || exit 1
    prepare_shared_photon_table || exit 1
fi

publish_retained() {
    local source="$1" target="$2"
    if cmp -s -- "${source}" "${target}"; then
        return
    fi
    local temporary="${target}.tmp"
    cp --reflink=auto -- "${source}" "${temporary}" && mv -f -- "${temporary}" "${target}"
}

run_tile() {
    local tif_path="$1"
    local gpu_id="$2"
    local worker_index="$3"
    local tile_filename
    local tile_stem
    local job_dir
    local status_file
    local log_path
    local started_at
    local finished_at
    local initial_stage="full_pipeline"
    local atl03_dir

    tile_filename="$(basename "${tif_path}")"
    tile_stem="${tile_filename%.*}"
    job_dir="${OUTPUT_ROOT}/${tile_stem}"
    atl03_dir="${job_dir}/ICESat-2"
    if [ "${CONTROL_SOURCE}" = "photon" ] && [ "${ATL03_SCOPE}" = "shared" ]; then
        atl03_dir="${SHARED_ATL03_DIR}"
    fi
    status_file="${job_dir}/status.json"
    log_path="${job_dir}/run.log"
    mkdir -p "${job_dir}"

    free_kib="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR==2 {print $4}')"
    min_free_kib=$((MIN_FREE_GIB * 1024 * 1024))
    if [ "${free_kib}" -lt "${min_free_kib}" ]; then
        started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
            "failed" "disk_preflight" 1 "${started_at}" "${started_at}" "${gpu_id}"
        echo "[STOP] ${tile_stem}: less than ${MIN_FREE_GIB} GiB free; worker ${worker_index} stopped" >&2
        return 2
    fi

    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    : >"${log_path}"
    if [ "${CONTROL_SOURCE}" = "photon" ]; then
        initial_stage="atl03_download"
    fi
    write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
        "running" "${initial_stage}" 0 "${started_at}" "" "${gpu_id}"

    if [ "${CONTROL_SOURCE}" = "photon" ]; then
        if [ "${ATL03_SCOPE}" = "shared" ]; then
            echo "[USE] ${tile_stem}: shared ATL03 ${atl03_dir}"
        elif [ -n "${LOCAL_ATL03_DIR}" ]; then
            atl03_dir="${LOCAL_ATL03_DIR}"
        elif ! atl03_dir="$(download_atl03 "${atl03_dir}" --tif "${tif_path}" 2>>"${log_path}")"; then
            finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
                "failed" "atl03_download" 1 "${started_at}" "${finished_at}" "${gpu_id}"
            echo "[FAIL] ${tile_stem}: ATL03 download; see ${log_path}" >&2
            return 1
        fi

        if ! find -L "${atl03_dir}" -maxdepth 1 -type f -name '*.h5' -size +0c -print -quit | grep -q .; then
            finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
                "failed" "atl03_no_hdf5" 1 "${started_at}" "${finished_at}" "${gpu_id}"
            echo "[FAIL] ${tile_stem}: downloader produced no HDF5; see ${log_path}" >&2
            return 1
        fi
    else
        echo "[SKIP] ${tile_stem}: ATL03 disabled (using SynRS3D metric scale)"
    fi

    write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
        "running" "full_pipeline" 0 "${started_at}" "" "${gpu_id}"
    echo "[RUN] ${tile_stem} on GPU ${gpu_id}: run_full_pipeline.sh"

    local pipeline_args=()
    if [ "${FORCE}" -eq 1 ]; then
        pipeline_args+=(--force)
    fi
    local pipeline_run_id="${BASHPID}-${started_at}-${tile_stem}"
    local pipeline_env=(
        "HAB_PIPELINE_RUN_ID=${pipeline_run_id}"
        "MPLBACKEND=Agg"
        "CONDA_BIN=${CONDA_BIN}" "GEO_ENV=${GEO_ENV}"
        "NO_ALBUMENTATIONS_UPDATE=1"
        "JOB_DIR=${job_dir}"
        "TIF_IMG=${tile_filename}"
        "INPUT_TIF=${tif_path}"
        "ICESAT_DIR=ICESat-2"
        "ICESAT_PATH=${atl03_dir}"
        "MLS_OUTPUT_DIR=${job_dir}/mlsbrn"
        "WS=${WS}" "OL=${OL}" "DR=${DR}" "PR=${PR}" "BS=${BS}"
        "HEIGHT_SOURCE=${HEIGHT_SOURCE}" "MODEL_NAME=${MODEL_NAME}" "GPU_ID=${gpu_id}"
        "CONTROL_SOURCE=${CONTROL_SOURCE}" "COMPUTE_UNCERTAINTY=${COMPUTE_UNCERTAINTY}"
        "FUSION_MODE=${FUSION_MODE}" "GROUND_FILTER=${GROUND_FILTER}" "PORT=$((29705 + worker_index))"
        "PHOTON_TABLE=$([ "${ATL03_SCOPE}" = "shared" ] && printf '%s' "${SHARED_PHOTON_TABLE}")"
        "PHOTON_TABLE_SHARED=$([ "${ATL03_SCOPE}" = "shared" ] && printf 1 || printf 0)"
        "ALLOW_MODEL_FALLBACK=${ALLOW_MODEL_FALLBACK}"
        "MIN_PHOTON_CONTROLS=${MIN_PHOTON_CONTROLS}"
        "EXPORT_CALIBRATED_WINDOWS=${EXPORT_CALIBRATED_WINDOWS}"
    )
    local pipeline_command=(bash scripts/run_full_pipeline.sh "${pipeline_args[@]}")
    local pipeline_exit
    local -a stream_exit_codes
    if [ "${SHOW_PROGRESS}" = "1" ]; then
        env "${pipeline_env[@]}" "${pipeline_command[@]}" 2>&1 | tee -a "${log_path}"
        stream_exit_codes=("${PIPESTATUS[@]}")
        pipeline_exit=${stream_exit_codes[0]}
        if [ "${pipeline_exit}" -eq 0 ]; then pipeline_exit=${stream_exit_codes[1]}; fi
    else
        env "${pipeline_env[@]}" "${pipeline_command[@]}" >>"${log_path}" 2>&1
        pipeline_exit=$?
    fi
    if [ "${pipeline_exit}" -ne 0 ]; then
        finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        failed_stage="$("${PYTHON_BIN}" scripts/runtime/run_status.py failed-stage \
            "${job_dir}/stage_status.json" "${pipeline_run_id}")" || failed_stage=full_pipeline
        write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
            "failed" "${failed_stage}" "${pipeline_exit}" "${started_at}" "${finished_at}" "${gpu_id}"
        echo "[FAIL] ${tile_stem}: ${failed_stage}; see ${log_path}" >&2
        return 1
    fi

    echo "[RUN] ${tile_stem}: stage validation"
    if ! validate_full_tile "${job_dir}" "${tif_path}" "${tile_stem}" "${atl03_dir}" >>"${log_path}" 2>&1; then
        finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
            "failed" "stage_validation" 1 "${started_at}" "${finished_at}" "${gpu_id}"
        echo "[FAIL] ${tile_stem}: stage validation; see ${log_path}" >&2
        return 1
    fi

    source_height="${job_dir}/${MODEL_NAME}-windowsize-${WS}-overlap-${OL}-downratio-${DR}-depth/AbsoluteHeight/output.tif"
    source_geojson="${job_dir}/footprint_direct_nms_absolute_height.geojson"
    stable_height="${job_dir}/${tile_stem}_absolute_height.tif"
    stable_geojson="${job_dir}/${tile_stem}_building_heights.geojson"
    if ! publish_retained "${source_height}" "${stable_height}" \
        || ! publish_retained "${source_geojson}" "${stable_geojson}"; then
        finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
            "failed" "stable_result_copy" 1 "${started_at}" "${finished_at}" "${gpu_id}"
        echo "[FAIL] ${tile_stem}: stable result copy; see ${log_path}" >&2
        return 1
    fi

    if [ "${EXPORT_SYNRS3D_RAW_CITY}" -eq 1 ] || [ "${GLOBAL_PATCH_HAB_PRIOR}" -eq 1 ]; then
        if ! archive_synrs3d_patches "${job_dir}" "${tile_stem}" "${tif_path}" >>"${log_path}" 2>&1; then
            finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
                "failed" "raw_patch_archive" 1 "${started_at}" "${finished_at}" "${gpu_id}"
            echo "[FAIL] ${tile_stem}: raw patch archive; see ${log_path}" >&2
            return 1
        fi
    fi
    if ! validate_retained_tile "${job_dir}" "${tif_path}" "${tile_stem}" "${atl03_dir}" >>"${log_path}" 2>&1; then
        finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
            "failed" "retained_result_validation" 1 "${started_at}" "${finished_at}" "${gpu_id}"
        echo "[FAIL] ${tile_stem}: retained result validation; see ${log_path}" >&2
        return 1
    fi

    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    write_status "${status_file}" "${tile_stem}" "${tif_path}" "${job_dir}" \
        "success" "complete" 0 "${started_at}" "${finished_at}" "${gpu_id}"
    echo "[OK] ${tile_stem} on GPU ${gpu_id}: ${stable_height}; ${stable_geojson}"
}

worker_loop() {
    local worker_index="$1"
    local gpu_id="$2"
    local worker_failed=0
    local tile_index
    local return_code
    for ((tile_index=worker_index; tile_index<${#TIFS[@]}; tile_index+=${#GPU_IDS[@]})); do
        run_tile "${TIFS[${tile_index}]}" "${gpu_id}" "${worker_index}"
        return_code=$?
        if [ "${return_code}" -ne 0 ]; then
            worker_failed=1
        fi
        if [ "${return_code}" -eq 2 ]; then
            break
        fi
    done
    return "${worker_failed}"
}

worker_pids=()
for worker_index in "${!GPU_IDS[@]}"; do
    worker_loop "${worker_index}" "${GPU_IDS[${worker_index}]}" &
    worker_pids+=("$!")
done

batch_failed=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
        batch_failed=1
    fi
done

if ! summary_counts="$(rebuild_summary)"; then
    echo "ERROR: could not write batch summary." >&2
    exit 1
fi

echo "========== Batch finished =========="
echo "Started: ${batch_started_at}"
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Summary: ${OUTPUT_ROOT}/batch_status.csv"
echo "Counts: ${summary_counts}"

if [ "${batch_failed}" -ne 0 ] || grep -q ',failed,' "${OUTPUT_ROOT}/batch_status.csv"; then
    exit 1
fi

if [ "${MERGE_CITY}" -eq 1 ]; then
    echo "========== City merge =========="
    if ! "${CONDA_BIN}" run --no-capture-output -n "${GEO_ENV}" \
        python tools/merge_city_height_results.py \
        --input-dir "${INPUT_DIR}" \
        --results-root "${OUTPUT_ROOT}" \
        --output-tif "${CITY_OUTPUT_TIF}" \
        --output-geojson "${CITY_OUTPUT_GEOJSON}" \
        --expected-tiles "${#TIFS[@]}" \
        --overwrite; then
        echo "ERROR: city merge failed" >&2
        exit 1
    fi
    echo "[OK] City height raster: ${CITY_OUTPUT_TIF}"
    echo "[OK] City building GeoJSON: ${CITY_OUTPUT_GEOJSON}"
fi

if [ "${GLOBAL_OPTIMIZE}" -eq 1 ]; then
    echo "========== Global seam optimization =========="
    run_global_optimization || exit 1
fi

if [ "${EXPORT_SYNRS3D_RAW_CITY}" -eq 1 ] || [ "${GLOBAL_PATCH_OPTIMIZE}" -eq 1 ]; then
    echo "========== Global patch results =========="
    run_global_patch_optimization || exit 1
fi
