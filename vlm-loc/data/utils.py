import numpy as np 


def project_points_to_pixels(xy: np.ndarray, x_min, y_min, x_scale, y_scale, img_size_minus_1):
    """
    世界→像素（分箱基于像素左边界）。返回:
      x_img(int32), y_img_flipped(int32), idx(int64)
    约定:
      - W = img_size_minus_1 + 1
      - x_scale, y_scale 没再直接使用；仅依赖 bev_range = W/x_scale = H/y_scale
    """
    W = int(img_size_minus_1) + 1
    H = W  # 你使用的是正方形 BEV

    # 用“格子左边界分箱”：i = floor( (x-x_min) * W / range )
    # 其中 range = W / x_scale
    bev_range_x = W / float(x_scale)
    bev_range_y = H / float(y_scale)

    x = xy[:, 0]
    y = xy[:, 1]

    # 为避免恰落在右边界的浮点误差，减去极小 eps
    eps = 1e-9

    # 未翻转的图像 y 轴向下，先得到行号 j0，再做翻转
    i = np.floor(((x - x_min) * W / bev_range_x) - eps).astype(np.int64, copy=False)
    j0 = np.floor(((y - y_min) * H / bev_range_y) - eps).astype(np.int64, copy=False)

    # 裁剪到合法索引
    np.clip(i, 0, W - 1, out=i)
    np.clip(j0, 0, H - 1, out=j0)

    # 翻转 y：图像行号自上而下 -> BEV y 向上
    j = (H - 1) - j0

    # 转回 int32 以保持你的返回类型
    x_img = i.astype(np.int32, copy=False)
    y_img_flipped = j.astype(np.int32, copy=False)

    idx = (j * W + i).astype(np.int64, copy=False)
    return x_img, y_img_flipped, idx


def pixels_to_world_from_center(
    x_pix, y_pix,
    center_world,   # (cx_w, cy_w)
    bev_range,      # 覆盖边长(米) —— 这里与上面一致: bev_range = W / x_scale
    image_size,     # H=W
    use_center=True # 总是以像素中心反投影（保证 pixel->world->pixel 恒等）
):
    """
    像素→世界（固定到像素中心）。
    对 project_points_to_pixels 的分箱方案严格对偶：
      给定 (x_pix, y_pix)，反投影到该像素中心的世界坐标；
      再前向投影回去，必回到同一像素。
    """
    W = int(image_size)
    H = W
    cx_w, cy_w = float(center_world[0]), float(center_world[1])
    half = float(bev_range) / 2.0

    x_min = cx_w - half
    y_min = cy_w - half

    # 像素中心
    off = 0.5 if use_center else 0.0

    # 翻转回未翻转的行号 j0
    x_pix = np.asarray(x_pix, dtype=np.float64)
    y_pix = np.asarray(y_pix, dtype=np.float64)

    j = y_pix
    j0 = (H - 1) - j

    # 对应像素中心的世界坐标（严格中心：+0.5）
    xw = x_min + ((x_pix + off) * bev_range) / W
    yw = y_min + ((j0    + off) * bev_range) / H
    return xw, yw


