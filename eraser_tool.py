import cv2
import numpy as np

def launch_eraser_tool(image):
    brush_radius = 20
    drawing = False
    h, w = image.shape[:2]

    # 縮小スケール（画面内に収める）
    screen_h = 800
    screen_w = 1200
    scale = min(screen_w / w, screen_h / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)

    mask = np.zeros((h, w), dtype=np.uint8)

    def draw_circle(event, x, y, flags, param):
        nonlocal drawing
        gx = int(x / scale)
        gy = int(y / scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(mask, (gx, gy), brush_radius, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(mask, (gx, gy), brush_radius, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    window_name = "EraserTool (drag to mark, 'r'=repair, ESC=skip)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, draw_circle)

    while True:
        # 縮小画像とマスクも縮小
        resized_image = cv2.resize(image, (disp_w, disp_h))
        resized_mask = cv2.resize(mask, (disp_w, disp_h))

        # 半透明赤マスクを作成
        overlay = resized_image.copy()
        overlay[resized_mask > 0] = (0, 0, 255)  # 赤にする

        alpha = 0.4
        display = cv2.addWeighted(overlay, alpha, resized_image, 1 - alpha, 0)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1)

        if key == ord('r'):
            result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            cv2.destroyAllWindows()

            # 白黒化
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result_bw = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            return result_bw

        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return image
