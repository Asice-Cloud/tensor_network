#!/usr/bin/env python3
"""
绘制辫子群示意图：上下两个平面（top/bottom），点列为 1..n。
绘制生成元 σ_k （第 k 条与 k+1 条交换）的示意：一条弧越过另一条弧。
保存为 braid_sigma_k.png

用法: python3 braid_plot.py [n] [k] [over|under]
  n: 点数量 (>=2)
  k: 交换索引（1-based），表示 σ_k 交换 k 和 k+1
  over|under: 指定哪条弧在上面（默认为 over）

示例: python3 braid_plot.py 6 3 over

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


def make_curve(x0, y0, x1, y1, control_height=0.4):
    """用三点贝塞尔近似（用 Path.CURVE3）生成一段平滑曲线控制点位于中间的上/下方。
    返回 PathPatch 对象的 Path 和 verts 用于绘制。"""
    # 使用一个二阶贝塞尔（CURVE3）需要一个控制点
    midx = 0.5 * (x0 + x1)
    midy = 0.5 * (y0 + y1)
    # control point 移动到 mid + (0, control_height)
    ctrl = (midx, midy + control_height)
    verts = [(x0, y0), ctrl, (x1, y1)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    return Path(verts, codes)


def draw_braid(n=6, k=3, over=True, fname='braid_sigma_k.png'):
    # positions
    xs = np.arange(n)
    y_top = 1.0
    y_bot = 0.0

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-0.6, n-1+0.6)
    ax.set_ylim(-0.6, 1.6)
    ax.axis('off')

    # draw top and bottom horizontal planes as faint lines
    ax.hlines(y_top, -0.6, n-1+0.6, colors='#dddddd')
    ax.hlines(y_bot, -0.6, n-1+0.6, colors='#dddddd')

    # label top/bottom points
    for j, x in enumerate(xs, start=1):
        ax.plot(x, y_top, 'ko')
        ax.text(x, y_top+0.08, str(j), ha='center', va='bottom')
        ax.plot(x, y_bot, 'ko')
        ax.text(x, y_bot-0.08, str(j), ha='center', va='top')

    idx = k-1
    # build mapping from top index -> bottom index (0-based)
    bottom_target = list(range(n))
    if idx < 0 or idx >= n-1:
        raise ValueError('k must be in 1..n-1')
    bottom_target[idx] = idx+1
    bottom_target[idx+1] = idx

    # draw strands
    for j in range(n):
        x_top = xs[j]
        x_bot = xs[bottom_target[j]]
        # if strand is part of swapped pair, draw curved path
        if j == idx or j == idx+1:
            # determine which one should go "over"
            if over:
                over_idx = idx  # by convention, the left one goes over; can swap
            else:
                over_idx = idx+1
            if j == over_idx:
                # over strand: arc that goes up above top plane
                path = make_curve(x_top, y_top, x_bot, y_bot, control_height=0.7)
                patch = patches.PathPatch(path, lw=3, edgecolor='C0', facecolor='none', zorder=3)
                ax.add_patch(patch)
            else:
                # under strand: arc that dips below bottom plane
                path = make_curve(x_top, y_top, x_bot, y_bot, control_height=-0.6)
                patch = patches.PathPatch(path, lw=3, edgecolor='C1', facecolor='none', alpha=0.9, zorder=2)
                ax.add_patch(patch)
        else:
            # straight-ish curve small control
            path = make_curve(x_top, y_top, x_bot, y_bot, control_height=0.0)
            patch = patches.PathPatch(path, lw=2, edgecolor='k', facecolor='none', zorder=1)
            ax.add_patch(patch)

    # annotate which generator and which crossing
    ax.text(0.02, 1.48, f'Braid generator σ_{k} (swap {k}↔{k+1}), over={over}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    print('Saved', fname)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n', nargs='?', type=int, default=6, help='number of strands (>=2)')
    parser.add_argument('k', nargs='?', type=int, default=3, help='generator index σ_k (1-based)')
    parser.add_argument('mode', nargs='?', choices=['over', 'under'], default='over', help='which strand passes over')
    args = parser.parse_args()
    draw_braid(n=args.n, k=args.k, over=(args.mode=='over'))
