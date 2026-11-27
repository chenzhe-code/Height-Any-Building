# Third-party components

The root research license covers only original HeightAnyBuilding code and
contributions. Preserve all applicable third-party rights and notices. Code
licenses do not automatically cover pretrained weights or datasets.

| Component | Included material | Local evidence / terms |
| --- | --- | --- |
| [Distill-Any-Depth](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth) | Selected inference/model dependency files and the GeoTIFF adapter | Upstream snapshot's MIT text is retained at `MDE/Distill-Any-Depth/LICENSE`; copied backbone files retain their Meta/MiDaS provenance |
| [Depth Anything V1](https://github.com/LiheYoung/Depth-Anything) | Inference model files and GeoTIFF adapter | Apache-2.0 text retained at `MDE/DEPTH-ANYTHING-V1/LICENSE` |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Inference model files and GeoTIFF adapter | Apache-2.0 code text retained; provider README states Small weights are Apache-2.0 and Base/Large/Giant weights CC-BY-NC-4.0 |
| [SynRS3D](https://github.com/JTRNEO/SynRS3D) | DPT inference model, GeoTIFF I/O and augmentation adapter | MIT text retained at `MDE/SynRS3D/LICENSE` |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Notices and dependent/adapted model layers; backbone otherwise obtained externally | Apache-2.0 text retained in `third_party/licenses/DINOv2-LICENSE`; V1/SynRS3D runtime reference pinned to `b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8` |
| [CSF](https://github.com/jianboqi/CSF) | Python integration wrapper; native code installed separately | Original CSF Apache-2.0 license retained in `third_party/licenses/CSF-LICENSE.txt` |
| [MLS-BRN](https://github.com/opendatalab/MLS-BRN), [bstool](https://github.com/Hoteryoung/bstool), [wwtool](https://github.com/jwwangchn/wwtool) | Only inference integration overlays and configuration; backend runtime prepared separately | Root LICENSE files absent from the examined local/upstream snapshots; MLS-BRN README contains an Apache-2.0 badge, which is not a substitute for the full license text |

The maintained MLS-BRN inference and dumping changes retain their original
upstream paths within `MLS-BRN/`. Redistribution of these derived files depends
on permission from the relevant upstream projects. The root inventories of the
pinned snapshots were rechecked on 2026-11-25. They omit root license files,
so confirm that permission before public redistribution. Install the full
frameworks separately.

The external runtime preparation was checked against these exact public snapshots:

| Backend | Revision |
| --- | --- |
| MLS-BRN | `1b05c07d2a8dd88586937ce39ff232f01cd5391a` |
| bstool, `modify_for_loft-foa-fro` | `eb85efd15c4f33c147ddfec3cda68cf34aed5cea` |
| wwtool | `2f462a3d028b766234d62a3ef706a0f08f10680a` |

The maintainers retain the project source revision and per-file source hashes.
The release retains the project's MLS-BRN inference fixes at their native paths.

The Git source archive excludes pretrained model files, satellite imagery, ATL03
granules, reference LiDAR and derived city datasets. Obtain these under the
respective provider's terms. The project-trained MLS-BRN checkpoint is available
for non-commercial research as the Release asset `mlsbrn-inference.pth`.
Its distribution terms are separate from the code license.

The research-only root license restricts fields of use and is therefore not an
OSI open-source license. See the [Open Source Definition](https://opensource.org/osd),
especially its field-of-endeavor requirement.
