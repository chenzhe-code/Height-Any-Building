# Installation

Run commands from the release root on Linux x86-64. Use
[`environment.yml`](environment.yml) for a compatible GDAL, PROJ, Rasterio,
Pyogrio and PyProj stack from Conda Forge. This combination passed the full
CPU regression suite. The GPU commands use the corresponding PyTorch CUDA
builds.

| Stage | Default environment | Handoff |
| --- | --- | --- |
| Prepare imagery and ATL03, optimize and assign heights | `hab-core` | GeoTIFF prediction patches and controls, GeoJSON footprints |
| Distill-Any-Depth | `distill-any-depth` | Writes GeoTIFF prediction patches |
| Depth Anything V1 | `depth-anything-v1` | Writes GeoTIFF prediction patches |
| Depth Anything V2 | `depth-anything-v2` | Writes GeoTIFF prediction patches |
| SynRS3D depth and crossing-window inference | `synrs3d` | Writes metric predictions |
| MLS-BRN footprint detection and CSV export | `mlsbrn` | Writes detections for footprint georeferencing |

Launch the CLI or browser interface from `hab-core`. It starts depth and
footprint inference as subprocesses in the corresponding environments. Install
the model environments you intend to use. Cross-tile optimization always uses
`synrs3d` for the seam predictions. MLS-BRN uses Python 3.8 for its older
PyTorch, MMCV and compiled extensions.
The Conda recipes disable user-site imports to keep packages in
`~/.local` from leaking into these environments.

## Geospatial core and CPU demo

```bash
conda env create -f environments/environment.yml
conda activate hab-core
python -m pip install -r environments/requirements.txt
python scripts/run_cpu_demo.py --work-dir work_dir/cpu-demo
```

The demo runs without PyTorch, CUDA, model weights or an Earthdata account. It
starts from generated prediction patches, controls and polygons.

## Depth and height model environments

Each model uses the same base Conda recipe for Python, NumPy, and GeoTIFF I/O,
with a separate environment and its own Python dependencies. Set `depth_env`
to `distill-any-depth`, `depth-anything-v1`, `depth-anything-v2`, or
`synrs3d`. Run the block once for each model you plan to use.

```bash
depth_env=distill-any-depth
conda env create -n "$depth_env" -f environments/environment-depth.yml
conda activate "$depth_env"
python -m pip install torch==2.2.1 torchvision==0.17.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "environments/requirements-$depth_env.txt"
python -m pip check
python -c 'import cv2, rasterio, torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Choose a PyTorch CUDA build compatible with your NVIDIA driver. The commands
above use the CUDA 11.8 wheels. For a `CXXABI` error from the system
`libstdc++.so.6`, load the active Conda environment's libraries first.

```bash
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
conda deactivate
conda activate "$depth_env"
```

DINOv2 is pinned to `b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8` for SynRS3D and
the V1 adapter. Torch Hub can fetch it on the first run. For a local checkout, run the commands below.

```bash
git clone https://github.com/facebookresearch/dinov2.git /path/to/dinov2
git -C /path/to/dinov2 checkout b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8
export DINOV2_REPO_DIR=/path/to/dinov2
```

After installing the relevant environments and checkpoints, run the end-to-end
environment check from `hab-core`.

```bash
conda activate hab-core
python tools/run_image.py --image /path/to/image.tif --work-dir work_dir/check \
  --model Distill-Any-Depth --gpus 0 --check
```

To use an existing `DAB` setup with Distill-Any-Depth or Depth Anything V1/V2,
activate `DAB` and set `GEO_ENV=DAB` and `HAB_DEPTH_ENV=DAB` when launching
the job. `HAB_DEPTH_ENV` selects the primary model environment. Seam inference
continues to use `synrs3d`.

## External MLS-BRN backend

The release includes inference adapters and production fixes, rather than the
whole framework. Obtain the upstream projects outside this release directory:

```bash
git clone https://github.com/opendatalab/MLS-BRN.git /path/to/upstream-mlsbrn
git -C /path/to/upstream-mlsbrn checkout 1b05c07d2a8dd88586937ce39ff232f01cd5391a
git clone --branch modify_for_loft-foa-fro https://github.com/Hoteryoung/bstool.git /path/to/upstream-bstool
git -C /path/to/upstream-bstool checkout eb85efd15c4f33c147ddfec3cda68cf34aed5cea
git clone https://github.com/jwwangchn/wwtool.git /path/to/upstream-wwtool
git -C /path/to/upstream-wwtool checkout 2f462a3d028b766234d62a3ef706a0f08f10680a

python scripts/setup_mlsbrn.py \
  --mlsbrn-source /path/to/upstream-mlsbrn \
  --bstool-source /path/to/upstream-bstool \
  --wwtool-source /path/to/upstream-wwtool
```

The maintained changes live at their original relative paths under
`MLS-BRN/mmdet/`, `MLS-BRN/tools/bonai/`, `MLS-BRN/bstool/` and `MLS-BRN/wwtool/`.
The preparation command fills in the upstream runtime packages in that same
layout and preserves the maintained files when run again. The installed upstream
files are ignored by Git. A local JSON record captures the upstream revisions.

The upstream MLS-BRN recipe uses Python 3.8, PyTorch 1.11.0, torchvision 0.12.0,
CUDA 11.3 and mmcv-full 1.7.0. A starting setup is:

```bash
conda create -n mlsbrn python=3.8 pip
conda activate mlsbrn
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install mmcv-full==1.7.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
python -m pip install 'numpy==1.23.5' 'Cython==3.0.11' 'setuptools<70' \
  'yapf==0.40.1' 'opencv-python-headless==4.9.0.80' \
  'pandas==1.5.3' 'geopandas==0.13.2' 'shapely==1.8.5.post1' 'rasterio==1.3.11' \
  matplotlib seaborn scikit-image scipy pycocotools geojson networkx \
  terminaltables tqdm six
python -m pip install --no-build-isolation -e ./MLS-BRN
python -m pip install --no-build-isolation -e ./MLS-BRN/wwtool
python -m pip install --no-build-isolation -e ./MLS-BRN/bstool
```

Native extensions require a compatible compiler and CUDA development toolkit.
Check that `nvcc --version` is available and that `CUDA_HOME` points to the
toolkit used to compile the MLS-BRN extensions before running the editable
installs.
Follow the pinned upstream repositories if platform-specific build adjustments
are necessary. Review their licensing separately; their current snapshots do
not provide root LICENSE files.

## Weights

See the [README download table](../README.md#model-downloads) for the
providers and preferred paths. Existing local setups can also use the
following paths.

| Backend | Legacy local path |
| --- | --- |
| Distill-Any-Depth Large | `MDE/Distill-Any-Depth/checkpoint/Dav2-large.safetensors` |
| SynRS3D | `MDE/SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth` |
| Depth Anything V1 Large | `MDE/DEPTH-ANYTHING-V1/checkpoints/depth_anything_vitl14.pth` |
| Depth Anything V2 Large | `MDE/DEPTH-ANYTHING-V2/checkpoints/depth_anything_v2_vitl.pth` |
| MLS-BRN | `checkpoints/MLS-BRN/latest.pth` |
| MLS-BRN training output | `MLS-BRN/work_dirs/loft_foahfm_r50_fpn_2x_bonai_ssl[test]/latest.pth` or `epoch_24.pth` in the same directory |

The original project-trained MLS-BRN checkpoint includes training state.
The inference export is 1,839,134,854 bytes and keeps the model tensors and
class names. Download `mlsbrn-inference.pth` from the repository's Release
assets and save it as `MLS-BRN/checkpoints/mlsbrn-inference.pth`.

The validated file is named `mlsbrn-inference.pth`. Its SHA-256 is
`e44b6e608bb608e74f9c99fa066701e682cb9228078fe9e105db03ff92bedc10`.
Verify the checksum for that specific file before loading it. Maintainers
with the original checkpoint can export an inference file with PyTorch 2.2.

```bash
python tools/maintenance/export_mlsbrn_checkpoint.py /path/to/epoch_24.pth /path/to/mlsbrn-inference.pth
sha256sum /path/to/mlsbrn-inference.pth
```

V2 Large weights use the provider's non-commercial terms. V1's adapter needs
`COMPUTE_UNCERTAINTY=0` or `--no-uncertainty` with the low-level scripts.
`tools/run_image.py` uses single-prediction inference for V1.

## Data and credentials

Use a three-band RGB UInt8 GeoTIFF with a correct CRS and transform. The global
pipeline expects a north-up 1 m metre grid; `tools/run_image.py` prepares it.
No example urban imagery or measured heights are redistributed. The tests create
synthetic inputs.

Configure Earthdata authentication outside source control. The downloader reads
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or the private `.netrc` used by
requests. Do not put these values in shared scripts, screenshots or commits.
`PROMPT_EARTHDATA=true` enables an interactive prompt in a terminal.

For an existing prediction-only workflow, use the core algorithm without model
weights or an Earthdata account. For the full workflow, ATL03 availability,
network access and successful ground/building photon extraction are required.

## Reusing a work directory

Each completed stage saves an atomic record in `.pipeline-cache/` with its
input and settings key and its output file fingerprints. The fingerprints
use paths, sizes and timestamps. For the same prepared image grid, reuse the
work directory when changing model or ATL03 settings. Examples of what reruns
are shown below.

| Change | Recomputed stages |
| --- | --- |
| Depth checkpoint, processing size or uncertainty | Depth, RH2AH and building heights |
| MLS-BRN checkpoint | Footprint inference and export, control heights, RH2AH and building heights |
| ATL03 download date range or selected local HDF5 files | ATL03 inputs and photon table as needed, controls, RH2AH and building heights |
| Graph weights or overlap mask | RH2AH and building heights |

Downloaded granules are separated by request, including dates, product version
and input coverage. Shared photon tables are scoped to the granules and tile
set. Use `--local-atl03-dir DIR` for an explicitly selected set of local
HDF5 files. Download date options apply to downloads. For offline runs,
select the HDF5 files for the desired period in that directory.
`--shared-atl03-dir` sets the root of managed downloads.

The photon stage records the resolved source in `atl03_selection.json`.
When rerunning `scripts/sparse_height_extraction.sh`, keep the same `JOB_DIR`
to reuse that selection. Set `ICESAT_PATH` to the specific request directory
to switch sources or resolve a work directory with multiple historical
requests and no recorded selection.

Harmony runs use strict error handling by default. To use successful subsets
from a partial job, explicitly set `HARMONY_IGNORE_ERRORS=true`. The request
directory contains `atl03_download_partial.json` with the service errors and
available files, and the work directory selection is marked `partial`.
Partial results do not get a complete download cache record.

When all prepared input tiles match the recorded grid, resuming skips image
reprojection. If a tile is missing, preparation fills it from the source
image.

Batch runs keep reusable intermediate files. Plan storage for the image
patches, inference outputs and request-specific ATL03 data. Older jobs created
before stage records were introduced recompute unverified stages on the first
normal run. `--force` reruns the selected stages. Use `--steps N` or
`--steps N-M` for a standalone stage or range with normal cache reuse.
Use `--from-step N` to force a restart through stage 10 when the earlier
artifacts are ready.
