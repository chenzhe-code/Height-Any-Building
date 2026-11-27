import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. 设置 Nature 风格全局绘图参数 ---
def set_nature_style():
    """配置 Matplotlib 以符合 Nature/Science 期刊标准"""
    # 字体设置：优先使用 Arial 或 Helvetica
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    rcParams['font.size'] = 9
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 11
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9

    # 线条与刻度
    rcParams['axes.linewidth'] = 0.8       # 坐标轴线宽
    rcParams['lines.linewidth'] = 1.5      # 线宽
    rcParams['xtick.direction'] = 'out'    # 刻度向外
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.size'] = 4
    rcParams['ytick.major.size'] = 4
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['axes.grid'] = False

    # 颜色与图例
    rcParams['legend.frameon'] = False     # 图例去框，更简洁
    rcParams['legend.fontsize'] = 8
    rcParams['figure.facecolor'] = 'white'
    rcParams['axes.facecolor'] = 'white'

def _load_geojson_table(geojson_path: str) -> pd.DataFrame:
    """仅提取 properties 到 DataFrame"""
    if not os.path.isfile(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    feats = data.get("features", [])
    rows = [ (feat.get("properties") or {}) for feat in feats ]
    return pd.DataFrame(rows)

def _resolve_output_path(geojson_path: str, output_image_path: str) -> str:
    if os.path.isabs(output_image_path):
        return output_image_path
    return os.path.join(os.path.dirname(os.path.abspath(geojson_path)), output_image_path)

def plot_height_scatter(geojson_path, truth_col, pred_col, output_image_path, show_plot=False):
    set_nature_style() # 应用样式

    # 1) 读数据
    df = _load_geojson_table(geojson_path)
    missing_cols = [col for col in (truth_col, pred_col) if col not in df.columns]
    if missing_cols:
        available_cols = ", ".join(df.columns.astype(str))
        raise ValueError(
            f"Missing column(s): {', '.join(missing_cols)}. "
            f"Available columns: {available_cols}"
        )

    # 2) 取列并转为数值
    # 修正逻辑：通常 X 轴为真值 (Ground Truth)，Y 轴为预测值 (Prediction)
    x_truth = pd.to_numeric(df[truth_col], errors='coerce')
    y_pred = pd.to_numeric(df[pred_col], errors='coerce')

    # 3) 过滤无效值。Reference<=0 是当前项目评估中已排除的无效参考高度。
    mask = (
        ~(x_truth.isna() | y_pred.isna())
        & np.isfinite(x_truth)
        & np.isfinite(y_pred)
        & (x_truth > 0)
    )
    x = x_truth[mask].to_numpy()
    y = y_pred[mask].to_numpy()

    if x.size == 0 or y.size == 0:
        raise ValueError("有效样本数为 0,无法绘图。")

    # 4) =========== 统计计算 ===========
    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    r2 = r2_score(x, y)

    # 线性回归仅用于绘制拟合线 (拟合 y = kx + b)
    X_mat = x.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X_mat, y)
    k = reg.coef_[0]
    b = reg.intercept_

    # 5) =========== 密度计算 (核心美化步骤) ===========
    # 计算点密度，用于给散点上色。降采样拟合 KDE，避免整体结果图过慢。
    try:
        rng = np.random.default_rng(0)
        if x.size > 12000:
            kde_idx = rng.choice(x.size, size=12000, replace=False)
            kde_xy = np.vstack([x[kde_idx], y[kde_idx]])
        else:
            kde_xy = np.vstack([x, y])
        z = gaussian_kde(kde_xy)(np.vstack([x, y]))
        # 按密度排序，保证高密度点绘制在最上层
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    except Exception:
        z = None

    # 6) =========== 画图 ===========
    # 尺寸：Nature 单栏宽度约 89mm (3.5 inch)，双栏 183mm。
    # 这里设定为 5.2x5.2 inch，适合作为半页图或展示图。
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=300)

    # A. 绘制散点。保留原有 Spectral_r 密度色图，其他样式保持轻量化。
    if z is None:
        scatter = ax.scatter(
            x, y, s=7, alpha=0.55, color='#356AA0', edgecolor='none',
            rasterized=True, label='_nolegend_'
        )
    else:
        norm = colors.PowerNorm(gamma=0.55, vmin=np.nanmin(z), vmax=np.nanmax(z))
        scatter = ax.scatter(
            x, y, c=z, s=7, alpha=0.72, cmap='Spectral_r', norm=norm,
            edgecolor='none', rasterized=True, label='_nolegend_'
        )

    # B. 绘制 1:1 参考线 (灰色虚线，置于底层)
    lo = np.min([x.min(), y.min()])
    hi = np.max([x.max(), y.max()])
    # 稍微扩展一点范围，以免点压在轴上
    pad = (hi - lo) * 0.05
    ax.plot(
        [lo-pad, hi+pad], [lo-pad, hi+pad],
        linestyle=(0, (4, 3)), linewidth=1.0, color='#7A7A7A',
        alpha=0.75, zorder=1, label='1:1'
    )

    # C. 绘制回归线
    # 仅在数据范围内绘制
    x_range = np.linspace(x.min(), x.max(), 100)
    y_reg_line = k * x_range + b
    ax.plot(x_range, y_reg_line, color='#D55E00', linewidth=1.8, label='Linear fit')

    # 7) =========== 装饰与标注 ===========

    # 轴标签 (支持 LaTeX 格式)
    ax.set_xlabel(f'Reference (m)')
    ax.set_ylabel(f'Prediction (m)')

    # 去除上方和右侧的边框 (Spines)，这是 Nature 风格的标志
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#2F2F2F')
    ax.spines['bottom'].set_color('#2F2F2F')

    # 坐标范围一致 (Parity Plot 关键)
    limit_min = min(lo, 0) # 假设高度不小于0
    limit_max = np.ceil((hi + pad) / 10) * 10
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)
    ax.set_aspect('equal', adjustable='box')

    # 构建统计文本
    # 使用 LaTeX 数学字体渲染公式
    stats_text = (
        f"$y = {k:.2f}x + {b:.2f}$\n"
        f"$R^2 = {r2:.3f}$\n"
        f"$RMSE = {rmse:.2f}$ m\n"
        f"$MAE = {mae:.2f}$ m\n"
        f"$N = {len(x):,}$"
    )

    # 将统计文本放在左上角，无边框背景
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=8.8, linespacing=1.45,
            bbox=dict(
                boxstyle='round,pad=0.35',
                facecolor='white',
                edgecolor='#D1D5DB',
                linewidth=0.7,
                alpha=0.92,
            ))

    ax.legend(loc='lower right', handlelength=2.6)
    # 优化布局
    plt.tight_layout()

    # 保存与显示
    output_image_path = _resolve_output_path(geojson_path, output_image_path)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    fig.savefig(output_image_path, bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"Done. Image saved to {output_image_path}")
    print(f"Stats: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a reference-vs-prediction height scatter chart from GeoJSON properties."
    )
    parser.add_argument(
        "--geojson",
        default="work_dir/test/footprint_direct_nms_absolute_height_with_lidar.geojson",
        help="Input GeoJSON path.",
    )
    parser.add_argument(
        "--truth-col",
        default="LiDAR_Height",
        help="Property column used as reference height.",
    )
    parser.add_argument(
        "--pred-col",
        default="Extracted_Height",
        help="Property column used as predicted height.",
    )
    parser.add_argument(
        "--output",
        default="height_scatter_nature_style_test.png",
        help="Output image path. Relative paths are saved next to the input GeoJSON.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window after saving.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        plot_height_scatter(
            args.geojson,
            truth_col=args.truth_col,
            pred_col=args.pred_col,
            output_image_path=args.output,
            show_plot=args.show,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
