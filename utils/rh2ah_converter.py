import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

from utils.ground_filter import ground_filter
from utils.io_utils import list_files_in_directory, mosaic_patches
from utils.patch_positions import parse_patch_position
from utils.raster_io import read_geotiff, save_geotiff


class RH2AHConverter:
    """
    一个用于将相对高度(RH)地图转换为绝对高度(AH)地图的类。
    该类封装了整个处理流程，包括算法作为静态方法。
    """

    def __init__(self, input_dir, output_dir, ch_img_path, window_size=1024, overlap=0.5,
                 use_conf=False, metric_input=False, apply_ground_filter=True,
                 graph_control_lambda=1.0, graph_overlap_lambda=1.0,
                 overlap_mask="all", export_calibrated=False):
        """
        使用必要的参数和路径初始化转换器。
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ch_img_path = ch_img_path
        self.window_size = window_size
        self.overlap = overlap
        self.use_conf = use_conf
        self.metric_input = metric_input
        self.apply_ground_filter = apply_ground_filter
        self.export_calibrated = export_calibrated
        graph_lambdas = np.asarray([graph_control_lambda, graph_overlap_lambda], dtype=float)
        if not np.all(np.isfinite(graph_lambdas)) or np.any(graph_lambdas < 0) or not np.any(graph_lambdas > 0):
            raise ValueError("Graph coefficients must be finite, nonnegative, and not both zero")
        self.graph_control_lambda = float(graph_control_lambda)
        self.graph_overlap_lambda = float(graph_overlap_lambda)
        if overlap_mask not in ("all", "nonground"):
            raise ValueError("overlap_mask must be all or nonground")
        self.overlap_mask = overlap_mask
        
        # 初始化状态变量
        self.img_dir_path = None
        self.imgs = None
        self.metas = None
        self.position = None
        self.rh_norm_imgs = None
        self.norm_rh = None
        self.CH_img = None
        self.RH2CH_model = None
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def construct_overlaps(imgs, positions, window_size, overlap, ground_weight=0, non_ground_weight=1,
                           valid_mask_mode="all"):
        if len(imgs) != len(positions):
            raise ValueError("images and positions must have the same length")
        if valid_mask_mode not in ("all", "nonground"):
            raise ValueError("valid_mask_mode must be all or nonground")

        x_rank = {value: rank for rank, value in enumerate(sorted({x for x, _ in positions}))}
        y_rank = {value: rank for rank, value in enumerate(sorted({y for _, y in positions}))}
        overlaps = []
        for i, (x_i, y_i) in enumerate(positions):
            height_i, width_i = imgs[i].shape[-2:]
            for j in range(i + 1, len(positions)):
                x_j, y_j = positions[j]

                # Preserve the original 8-neighbour topology while allowing
                # the edge-anchored coordinate to use a shorter final stride.
                rank_dx = abs(x_rank[x_i] - x_rank[x_j])
                rank_dy = abs(y_rank[y_i] - y_rank[y_j])
                if rank_dx > 1 or rank_dy > 1 or (rank_dx == 0 and rank_dy == 0):
                    continue

                height_j, width_j = imgs[j].shape[-2:]
                left = max(x_i, x_j)
                top = max(y_i, y_j)
                right = min(x_i + width_i, x_j + width_j)
                bottom = min(y_i + height_i, y_j + height_j)
                if left >= right or top >= bottom:
                    continue

                overlap_region_i = imgs[i][top - y_i:bottom - y_i, left - x_i:right - x_i]
                overlap_region_j = imgs[j][top - y_j:bottom - y_j, left - x_j:right - x_j]
                if overlap_region_i.shape != overlap_region_j.shape:
                    raise ValueError(
                        f"overlap shape mismatch for patches {i} and {j}: "
                        f"{overlap_region_i.shape} != {overlap_region_j.shape}"
                    )

                height_attention_mask1 = np.where(
                    (overlap_region_i >= -3) & (overlap_region_i <= 3),
                    ground_weight,
                    non_ground_weight,
                )
                height_attention_mask2 = np.where(
                    (overlap_region_j >= -3) & (overlap_region_j <= 3),
                    ground_weight,
                    non_ground_weight,
                )
                height_attention_mask = np.where(
                    (height_attention_mask1 == ground_weight)
                    | (height_attention_mask2 == ground_weight),
                    ground_weight,
                    non_ground_weight,
                )
                valid_mask = height_attention_mask > 0 if valid_mask_mode == "nonground" else height_attention_mask > -1
                if not np.any(valid_mask):
                    continue
                avgheight_i = np.mean(overlap_region_i[valid_mask])
                avgheight_j = np.mean(overlap_region_j[valid_mask])
                overlaps.append((i, j, avgheight_i, avgheight_j))

        print("Construct {} overlaps.".format(len(overlaps)))
        return overlaps

    @staticmethod
    def construct_RH2CH(CH_img, RH_imgs, return_coord=False):
        def find_valid_points(CH_img_data):
            CH_img_data, _ = CH_img
            valid_points = np.argwhere(CH_img_data >= 5)
            print("The number of valid points is %d" % (len(valid_points)))
            return valid_points

        def construct_matches(i, CH_img, RH_img, valid_points):
            matches = []
            RH_img_data, RH_img_meta = RH_img
            RH_img_transform = RH_img_meta['transform']
            CH_img_data, CH_img_meta = CH_img
            CH_img_transform = CH_img_meta['transform']
            vis_RHCH_list = []
            for j, point in enumerate(valid_points):
                row_CH, col_CH = point
                x_geo, y_geo = CH_img_transform * (col_CH + 0.5, row_CH + 0.5)
                col_RH, row_RH = ~RH_img_transform * (x_geo, y_geo)
                row_RH, col_RH = int(np.floor(row_RH)), int(np.floor(col_RH))
                if 0 <= row_RH < RH_img_data.shape[0] and 0 <= col_RH < RH_img_data.shape[1]:
                    if return_coord:
                        matches.append((i, -1, RH_img_data[row_RH, col_RH].astype(np.float16), CH_img_data[row_CH, col_CH].astype(np.float16), (row_RH, col_RH), (row_CH, col_CH)))
                    else:
                        matches.append((i, -1, RH_img_data[row_RH, col_RH].astype(np.float16), CH_img_data[row_CH, col_CH].astype(np.float16)))
                    vis_RHCH_list.append({'RH_row': row_RH, 'RH_col': col_RH, 'RH_val': RH_img_data[row_RH, col_RH], 'CH_row': row_CH, 'CH_col': col_CH, 'CH_val': CH_img_data[row_CH, col_CH]})
            # if vis_RHCH_list:
            #     plt.figure(figsize=(12, 10))
            #     img_show = plt.imshow(RH_img_data, cmap='viridis')
            #     rh_rows = [item['RH_row'] for item in vis_RHCH_list]
            #     rh_cols = [item['RH_col'] for item in vis_RHCH_list]
            #     plt.plot(rh_cols, rh_rows, 'ro', markersize=5, label='Match Points')
            #     for item in vis_RHCH_list:
            #         plt.text(item['RH_col'], item['RH_row'], f"R:{item['RH_val']:.2f}\nC:{item['CH_val']:.2f}", color='yellow', fontsize=8, ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5, pad=1))
            #     plt.title('RH image (with RH & CH values)')
            #     plt.axis('off')
            #     plt.colorbar(img_show, fraction=0.046, pad=0.04)
            #     plt.tight_layout()
            #     plt.show()
                    
            # # --- 新增的过滤逻辑开始 ---
            # final_matches = []
            # zero_matches = []

            # for m in matches:
            #     # m[3] 是 CH_img_data 的值
            #     if m[3] != 0:
            #         final_matches.append(m)
            #     else:
            #         zero_matches.append(m)
            
            # # 处理 0 值列表
            # if len(zero_matches) > 2:
            #     # 随机抽取 n 个，random.sample 不会重复抽取同一个元素
            #     final_matches.extend(random.sample(zero_matches, 2))
            # else:
            #     # 如果不够 n 个，就全部放进去
            #     final_matches.extend(zero_matches)
            
            # return final_matches
            # # --- 新增的过滤逻辑结束 ---
            return matches

        overlaps = []
        valid_points = find_valid_points(CH_img)
        for i, RH_img in enumerate(RH_imgs):
            matches = construct_matches(i, CH_img, RH_img, valid_points)
            overlaps.extend(matches)
        print("Construct {} controls.".format(len(overlaps)))
        return overlaps

    @staticmethod
    def joint_RHCH_scale(images, overlaps, weights):
        # weights are coefficients of squared residuals, not residual multipliers.
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(overlaps),) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("Expected one finite nonnegative graph coefficient per constraint, with at least one positive coefficient")
        residual_weights = np.sqrt(weights)

        def RHCH_residuals(params, overlaps, images, ground_weight=0, non_ground_weight=1):
            num_images = len(images)
            scale = params[:num_images]
            
            residuals = []
            init_residuals = []
            for constraint_idx, (i, j, overlap_region_i, overlap_region_j) in enumerate(overlaps):
                if j == -1:
                    diff = ((overlap_region_i * scale[i]) - (overlap_region_j))
                    diff_init = (overlap_region_i - overlap_region_j)
                else:
                    diff = (overlap_region_i * scale[i] - overlap_region_j * scale[j])
                    diff_init = (overlap_region_i - overlap_region_j)
                    
                residuals.append(residual_weights[constraint_idx] * diff)
                init_residuals.append(diff_init)

            print("init_rmse", np.sqrt(np.mean(np.square(init_residuals))))
            print("rmse:", np.sqrt(np.mean(np.square(residuals))))

            return np.array(residuals)
        
        initial_scale_params = np.ones(len(images))
        lower_bounds = np.zeros(len(images))
        upper_bounds = np.full(len(images), np.inf)

        result = least_squares(RHCH_residuals, initial_scale_params, args=(overlaps, images), 
                                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1000000, 
                                method='trf', 
                                diff_step=1e-2,
                                bounds=(lower_bounds, upper_bounds),
                                # x_scale=param_scales,
                                # x_scale='jac',
                               )
        
        optimized_s = result.x[:len(images)]
        print("Optimized scales:", optimized_s)
        aligned_images = [optimized_s[i] * images[i] for i in range(len(images))]

        return aligned_images

    @staticmethod
    def RH_align_scale(images, overlaps, weights, base_idx=None):
        def RH_residuals_scale(params, overlaps, images, base_idx=0, ground_weight=0, non_ground_weight=1):
            num_images = len(images)
            scale = np.insert(params[:num_images-1], base_idx, 1)
            residuals = []
            init_residuals = []
            for (i, j, overlap_region_i, overlap_region_j) in overlaps:
                diff = (overlap_region_i * scale[i] - overlap_region_j * scale[j])
                diff_init = (overlap_region_i - overlap_region_j)

                residuals.append(diff)
                init_residuals.append(diff_init)

            print("init_rmse", np.sqrt(np.mean(np.square(init_residuals))))
            print("rmse:", np.sqrt(np.mean(np.square(residuals))))

            return np.array(residuals)
        
        initial_scale_params = np.ones(len(images)-1)
        lower_bounds = np.zeros(len(images)-1)
        upper_bounds = np.full(len(images)-1, np.inf)

        base_img_idx = random.randint(0, len(images)-1) if base_idx is None else base_idx

        result = least_squares(RH_residuals_scale, initial_scale_params, args=(overlaps, images, base_img_idx), 
                                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1000000, 
                                method='trf', 
                                diff_step=2e-2,
                                bounds=(lower_bounds, upper_bounds),
                                # x_scale=param_scales,
                                # x_scale='jac',
                               )
        
        optimized_s = np.insert(result.x[:len(images)-1], base_img_idx, 1)
        print("Optimized scales:", optimized_s)
        aligned_images = [optimized_s[i] * images[i] for i in range(len(images))]

        return aligned_images
        
    @staticmethod
    def coarse_RH2CH(RH_img, overlaps, weights, output_dir, save_out=True, num_poly=1, return_model=True):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        output_dir = os.path.join(output_dir, 'coarse_AH')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        RH_height = np.empty((len(overlaps), 1))
        CH_height = np.empty(len(overlaps))
        for k, (i, j, overlap_region_i, overlap_region_j) in enumerate(overlaps):
            RH_height[k, 0] = overlap_region_i
            CH_height[k] = overlap_region_j

        print('start RH2CH fit.')
        poly = PolynomialFeatures(degree=num_poly)
        X_poly = poly.fit_transform(RH_height)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly, CH_height, sample_weight=weights)
        print("多项式系数:", model.coef_)

        rmse = np.mean((model.predict(poly.transform(RH_height)) - CH_height) ** 2) ** 0.5
        print(f'The RH2CH RMSE is {rmse}')
        if save_out:
            CH_pred = model.predict(poly.transform(RH_img[0].reshape(-1,1))).reshape(RH_img[0].shape)
            CH_pred_out = np.repeat(CH_pred[..., np.newaxis], 3, axis=-1)
            CH_output_path = os.path.join(output_dir, 'CH_final.tif')
            save_geotiff(CH_output_path, CH_pred_out.transpose(2,0,1), RH_img[1])

        return [poly, model] if return_model else None


    @staticmethod
    def coarse_RH2CH_ransac(RH_img, CH_img, overlaps, weights, output_dir, save_out=True, num_poly=1, return_model=True):
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        output_dir = os.path.join(output_dir, 'coarse_AH')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) 

        RH_height = np.empty((len(overlaps), 1))
        CH_height = np.empty(len(overlaps))
        for k, (i, j, overlap_region_i, overlap_region_j, _, _) in enumerate(overlaps):
            RH_height[k, 0] = overlap_region_i
            CH_height[k] = overlap_region_j

        print('start RH2CH RANSAC fit.')
        poly = PolynomialFeatures(degree=num_poly, include_bias=False)
        X_poly = poly.fit_transform(RH_height)
        base_model = LinearRegression(fit_intercept=False)
        model = RANSACRegressor(estimator=base_model, min_samples=max(2, int(0.9 * len(X_poly))), residual_threshold=100.0, max_trials=100000)
        model.fit(X_poly, CH_height, sample_weight=weights)
        print("RANSAC多项式系数:", model.estimator_.coef_)

        inlier_mask = model.inlier_mask_
        print(inlier_mask)
        rmse = np.mean((model.predict(X_poly[inlier_mask]) - CH_height[inlier_mask]) ** 2) ** 0.5
        print(f'The RH2CH RANSAC RMSE (inliers) is {rmse}')

        if save_out:
            CH_pred = model.predict(poly.transform(RH_img[0].reshape(-1,1))).reshape(RH_img[0].shape)
            CH_pred_out = np.repeat(CH_pred[..., np.newaxis], 3, axis=-1)
            CH_output_path = os.path.join(output_dir, 'CH_final_ransac.tif')
            save_geotiff(CH_output_path, CH_pred_out.transpose(2,0,1), RH_img[1])

        # 将 CH_img 中非内点位置的像素设置为 -10
        # mask_img = np.full(CH_img[0].shape, False, dtype=bool)
        # for k, (i, j, _, _, rh_coord, ch_coord) in enumerate(overlaps):
        #     if not inlier_mask[k]:
        #         row_CH, col_CH = ch_coord
        #         if 0 <= row_CH < CH_img[0].shape[0] and 0 <= col_CH < CH_img[0].shape[1]:
        #             mask_img[row_CH, col_CH] = True
        # CH_img[0][mask_img] = -10

        return [poly, model] if return_model else None


    def _preprocess_and_filter(self):
        """步骤1:加载数据并应用地面点滤波器以转换为相对高度。"""
        print("--- 步骤1: 预处理与滤波 ---")
        self.img_dir_path = list_files_in_directory(self.input_dir)
        imgs_tif = [read_geotiff(img_path, band=1) for img_path in self.img_dir_path]
        self.imgs, self.metas = map(list, zip(*imgs_tif))
        self.imgs = [img.astype(np.float64) for img in self.imgs]
        if not self.apply_ground_filter:
            print("跳过地面滤波；直接使用模型输出高度。")
            return
        for i, img_tif in enumerate(tqdm(imgs_tif, desc="应用地面点滤波")):
            ndsm_patch, _ = ground_filter(img_tif, self.output_dir, window_size=100., 
                                          down_ratio=4, ground_ratio=0.0001, save_result=False, 
                                          dir_tag='rh_ndsm', img_name=os.path.basename(self.img_dir_path[i]))['ndsm']
            ndsm_patch[ndsm_patch < 0] = 0
            self.imgs[i] = ndsm_patch

    def _align_relative_heights(self):
        """步骤2:对齐相对高度切片，以获得全局一致的尺度。"""
        print("\n--- 步骤2: 对齐相对高度 ---")
        self.position = [parse_patch_position(img_path) for img_path in self.img_dir_path]
        if self.metric_input:
            # SynRS3D patches already share an absolute metric scale. Anchoring
            # only one patch at 1 while freely fitting every other patch can
            # collapse low-height/coastal tiles toward zero. A metric fallback
            # must therefore preserve every patch at scale=1 exactly.
            self.rh_norm_imgs = self.imgs
            print(
                "Metric input: skipped patch scale fitting; all patch scales "
                "are fixed at 1."
            )
        elif len(self.imgs) == 1:
            self.rh_norm_imgs = self.imgs
            print("Single patch: no relative scale alignment needed.")
        else:
            overlap_options = {"valid_mask_mode": self.overlap_mask} if self.overlap_mask != "all" else {}
            overlaps_RH2RH = self.construct_overlaps(
                self.imgs, self.position, self.window_size, self.overlap, **overlap_options
            )
            print(f'相对高度重叠区域数量: {len(overlaps_RH2RH)}')
            rh_weights = np.ones(len(overlaps_RH2RH))
            self.rh_norm_imgs = self.RH_align_scale(
                self.imgs, overlaps_RH2RH, rh_weights
            )
        self.norm_rh = mosaic_patches(
            self.rh_norm_imgs, self.metas, self.img_dir_path, self.window_size,
            self.output_dir, "RelativeHeight", use_conf=self.use_conf,
        )
        # plt.imshow(self.norm_rh[0])
        # plt.title('Mosaicked Normalized Relative Height')
        # plt.axis('off')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()

    def _create_coarse_absolute_model(self):
        """步骤3:创建一个从相对高度到绝对高度的粗略转换模型。"""
        print("\n--- 步骤3: 创建粗略的RH-to-AH模型 ---")
        self.CH_img = read_geotiff(self.ch_img_path, band=1)
        overlaps_NRH2CH = self.construct_RH2CH(self.CH_img, [self.norm_rh], return_coord=True)
        print(f'控制点高度重叠区域数量: {len(overlaps_NRH2CH)}')
        ch_weights = np.ones(len(overlaps_NRH2CH))
        self.RH2CH_model = self.coarse_RH2CH_ransac(self.norm_rh, self.CH_img, overlaps_NRH2CH, ch_weights, self.output_dir, save_out=True)
        norm_ch_show = self.RH2CH_model[1].predict(self.RH2CH_model[0].transform(self.norm_rh[0].reshape(-1,1))).reshape(self.norm_rh[0].shape)
        # # norm_ch_show = self.RH2CH_model['a'] * (self.norm_rh[0] ** self.RH2CH_model['b'])
        # plt.imshow(norm_ch_show)
        # plt.title('Coarse Absolute Height Preview')
        # plt.axis('off')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()

    def _save_calibrated_windows(self, images):
        directory = Path(self.output_dir) / "CalibratedWindows"
        directory.mkdir(exist_ok=True)
        manifest = directory / "manifest.json"
        manifest.unlink(missing_ok=True)
        records = []
        for (x, y), image, meta in zip(self.position, images, self.metas, strict=True):
            prediction = np.asarray(image, dtype=np.float32)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite calibrated height window at ({x}, {y})")
            filename = f"x{x}_y{y}.npz"
            temporary = directory / f"{filename}.tmp"
            with temporary.open("wb") as output:
                np.savez_compressed(output, prediction=prediction)
            temporary.replace(directory / filename)
            records.append({"x": x, "y": y, "file": filename, "shape": list(prediction.shape),
                            "transform": list(meta["transform"])[:6], "crs": meta["crs"].to_string()})
        names = {record["file"] for record in records}
        for path in directory.glob("*.npz"):
            if path.name not in names:
                path.unlink()
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 1, "units": "m", "patches": records}), encoding="utf-8")
        temporary.replace(manifest)

    def _joint_optimization_and_finalize(self):
        """步骤4:执行联合优化并生成最终的绝对高度图。"""
        print("\n--- 步骤4: 联合优化与最终拼接 ---")
        corse_CH_imgs = []
        for rh_img in tqdm(self.rh_norm_imgs, desc="生成粗糙AH切片"):
            ch_img = self.RH2CH_model[1].predict(self.RH2CH_model[0].transform(rh_img.reshape(-1,1))).reshape(rh_img.shape)
            # ch_img = self.RH2CH_model['a'] * (rh_img.reshape(-1, 1) ** self.RH2CH_model['b']).reshape(rh_img.shape)
            corse_CH_imgs.append(ch_img)
        overlaps_CCH2CH = self.construct_RH2CH(self.CH_img, list(zip(corse_CH_imgs, self.metas)), return_coord=False)
        overlap_options = {"valid_mask_mode": self.overlap_mask} if self.overlap_mask != "all" else {}
        overlaps_CCH2CCH = self.construct_overlaps(corse_CH_imgs, self.position, self.window_size, self.overlap, **overlap_options)
        overlaps_RHCH = [*overlaps_CCH2CCH, *overlaps_CCH2CH]
        print(f'用于联合优化的总重叠区域数量: {len(overlaps_RHCH)}')
        rhch_weights = np.asarray([
            self.graph_control_lambda if j == -1 else self.graph_overlap_lambda
            for _, j, _, _ in overlaps_RHCH
        ], dtype=np.float64)
        refined_imgs = self.joint_RHCH_scale(corse_CH_imgs, overlaps_RHCH, rhch_weights)
        if self.export_calibrated:
            self._save_calibrated_windows(refined_imgs)
        refined_ch = mosaic_patches(
            refined_imgs, self.metas, self.img_dir_path, self.window_size,
            self.output_dir, "AbsoluteHeight", use_conf=self.use_conf,
        )
        # plt.imshow(refined_ch[0])
        # plt.title('Final Refined Absolute Height')
        # plt.axis('off')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()
        print("\n--- 最终验证 ---")
        overlaps_test = self.construct_RH2CH(self.CH_img, [refined_ch])
        ch_weights_test = np.ones(len(overlaps_test))
        test_model = self.coarse_RH2CH(refined_ch, overlaps_test, ch_weights_test, self.output_dir, save_out=False)

    def run(self):
        """
        执行完整的 RH 到 AH 转换流程。
        """
        self._preprocess_and_filter()
        self._align_relative_heights()
        self._create_coarse_absolute_model()
        self._joint_optimization_and_finalize()
        print("\n处理完成。")

    def run_metric(self):
        """Fuse metric height patches without changing their model scale."""
        self._preprocess_and_filter()
        self._align_relative_heights()
        if self.export_calibrated:
            self._save_calibrated_windows(self.rh_norm_imgs)
        relative_path = os.path.join(self.output_dir, "RelativeHeight", "output.tif")
        absolute_dir = os.path.join(self.output_dir, "AbsoluteHeight")
        os.makedirs(absolute_dir, exist_ok=True)
        shutil.copy2(relative_path, os.path.join(absolute_dir, "output.tif"))
        print("\nMetric-height fusion complete; every SynRS3D patch scale remained 1.")
