import cv2
import numpy as np

# ウィンドウ内に収まる最大表示サイズ (GUI不要なので未使用)
MAX_DISPLAY_WIDTH = 800
MAX_DISPLAY_HEIGHT = 600

def _order_points(pts):
    # pts: (4,2) の numpy 配列
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]        # 左上  = x+y 最小
    rect[2] = pts[np.argmax(s)]        # 右下  = x+y 最大
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # 右上  = x−y 最小
    rect[3] = pts[np.argmax(diff)]     # 左下  = x−y 最大
    return rect

def compute_transform_with_canvas(src_pts, image_shape):
    # ① まずクリック順に関係なくソート
    src_pts = _order_points(src_pts)
    # 以下は既存の処理…
    h, w = image_shape[:2]
    dst_pts = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # …キャンバスサイズ計算→出力…
    corners = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype=np.float32).reshape(-1,1,2)
    warped  = cv2.perspectiveTransform(corners, H).reshape(-1,2)
    xs, ys  = warped[:,0], warped[:,1]
    min_x, min_y = np.floor(xs.min()), np.floor(ys.min())
    max_x, max_y = np.ceil(xs.max()),  np.ceil(ys.max())
    tx, ty = -min_x, -min_y
    out_w, out_h = int(max_x-min_x), int(max_y-min_y)
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
    H2 = T @ H
    return H2, (out_w, out_h)

def warp_image_with_canvas(image, H2, out_size):
    """
    ホモグラフィ H2 とキャンバスサイズ out_size で画像をワープ
    borderValue=(255,255,255) で欠損領域を白で埋める
    """
    out_w, out_h = out_size
    return cv2.warpPerspective(
        image, H2, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

def transform_boxes(boxes, H2):
    """
    バウンディングボックス座標リストをホモグラフィ H2 で変換
    boxes: list of tuples/lists [x1, y1, x2, y2, ...]
    Returns: list of tuples (x1p, y1p, x2p, y2p, ...rest)
    """
    transformed = []
    for box in boxes:
        # box は可変長: 最低 4 要素 (x1,y1,x2,y2)、残りは rest
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        rest = box[4:] if len(box) > 4 else []
        # ４点をホモグラフィ変換
        pts = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, H2).reshape(-1, 2)
        xs, ys = warped[:, 0], warped[:, 1]
        x1p, y1p = int(np.floor(xs.min())), int(np.floor(ys.min()))
        x2p, y2p = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
        transformed.append((x1p, y1p, x2p, y2p, *rest))
    return transformed
