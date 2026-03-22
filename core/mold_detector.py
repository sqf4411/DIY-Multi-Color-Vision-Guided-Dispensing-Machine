# mold_detector.py
import cv2
import numpy as np
import os
import json

# ====== Global Cache ======
# These variables store data across calls to avoid re-computation.
_TEMPLATES = None  # (4,4,2) Stores coordinates of 4 user-drawn yellow boxes
_WARP_SIZE = None  # (2,)    [W, H] Stores the size of the golden template after perspective correction

_PATH_POLY = None  # (N,2)   Dispensing path polyline data for Cell 0
_PATH_DOTS = None  # (K,2)   Dispensing dots data for Cell 0

# Cell 0 [Core Template]
# This is the core of the algorithm: we use this single cropped, shadow-free template to match all positions.
_CORE_TEMPLATE = None
# Records the absolute coordinate (x, y) of the core template in the golden image, as the reference point for all relative offsets.
_CORE_REF_TL = None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Helper function: Sorts the four points of a quadrilateral in a fixed order: TL, TR, BR, BL.
    This is a prerequisite for perspective transformation.
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]  # Smallest x+y is Top-Left
    br = pts[np.argmax(s)]  # Largest x+y is Bottom-Right
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]  # Smallest y-x is Top-Right
    bl = pts[np.argmax(diff)]  # Largest y-x is Bottom-Left
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_big_mold(gray: np.ndarray):
    """
    Finds the largest rectangular contour (the big outer frame of the mold) in the input image.
    Steps: Downscale -> Gaussian Blur -> Threshold -> Morphology Close -> Find Contours -> Filter Max -> Restore Coords.
    """
    h_img, w_img = gray.shape
    img_area = h_img * w_img

    # ====== Optimization: Downscaling ======
    scale = 0.25  # Downscale to 1/4
    small_w = int(w_img * scale)
    small_h = int(h_img * scale)
    small_gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    # 1. Preprocessing: Denoise and Smooth (on small image)
    blur = cv2.GaussianBlur(small_gray, (7, 7), 0)
    # 2. Binarization
    _, th_bg = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    # 3. Morphology Close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # Kernel scaled down
    closed = cv2.morphologyEx(th_bg, cv2.MORPH_CLOSE, kernel)
    # 4. Find Contours
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    big_cnt = None
    max_area = 0
    min_area_thresh = (img_area * scale * scale) * 0.15 # Threshold scaled down

    # 5. Filter: Find the largest area
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area_thresh and area > max_area:
            max_area = area
            big_cnt = c

    if big_cnt is None: return None, None
    
    # 6. Calculate Min Area Rect
    rect = cv2.minAreaRect(big_cnt)
    # Get 4 corner points (small coords)
    box_small = cv2.boxPoints(rect)
    
    # ====== Restore Coords to Original Image ======
    box = (box_small / scale).astype(np.int32)
    
    # Recalculate rect for full size (center, size, angle)
    center = (rect[0][0] / scale, rect[0][1] / scale)
    size = (rect[1][0] / scale, rect[1][1] / scale)
    rect_full = (center, size, rect[2])
    
    return rect_full, box


def _get_perspective_matrix(box, w, h):
    """
    Calculates the Perspective Matrix.
    Maps an arbitrary quad (box) to a standard rectangle (w x h).
    """
    pts = box.astype(np.float32)
    # Sort 4 points
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Source Points
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    # Destination Points
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    # Calculate Matrix
    return cv2.getPerspectiveTransform(rect_pts, dst_pts)


def _load_golden_and_paths():
    """
    [Init Function]
    1. Load golden template data (slot_templates.npy).
    2. Load dispensing path data (disp_poly_cell0.npy).
    3. Key Step: Extract 'Core Template' (_CORE_TEMPLATE) from mold2.png, removing side wall interference.
    """
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS, _CORE_TEMPLATE, _CORE_REF_TL

    # Define absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "config")
    
    path_templates = os.path.join(config_dir, "slot_templates.npy")
    path_warp = os.path.join(config_dir, "warp_size.npy")
    path_poly_json = os.path.join(config_dir, "disp_poly_cell0.json")
    path_poly = os.path.join(config_dir, "disp_poly_cell0.npy")
    path_dots = os.path.join(config_dir, "disp_dots_cell0.npy")
    path_mold2 = os.path.join(config_dir, "mold2.png")

    # --- 1. Load Basic Data ---
    if _TEMPLATES is None:
        try:
            _TEMPLATES = np.load(path_templates)
            _WARP_SIZE = np.load(path_warp).astype(int)
            print(f"[mold_detector] Golden data loaded: {_TEMPLATES.shape}")
        except:
            print("[mold_detector] Missing config/slot_templates.npy")
            return

    # --- 2. Load Path Files ---
    # IMPORTANT: Use global golden-coordinate path files.
    if _PATH_POLY is None:
        if os.path.exists(path_poly_json):
            try:
                with open(path_poly_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                segments = data.get("segments", [])
                parsed = []
                for seg in segments:
                    arr = np.asarray(seg, dtype=np.float32)
                    if arr.size == 0:
                        continue
                    if arr.ndim == 1:
                        if arr.shape[0] != 2:
                            continue
                        arr = arr.reshape(1, 2)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        parsed.append(arr)
                _PATH_POLY = parsed if parsed else None
            except Exception:
                pass
        elif os.path.exists(path_poly):
            try:
                # Legacy fallback: numeric npy only (pickle disabled).
                arr = np.load(path_poly, allow_pickle=False)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.size > 0:
                    _PATH_POLY = [arr]
                else:
                    _PATH_POLY = None
            except Exception:
                pass

    if _PATH_DOTS is None and os.path.exists(path_dots):
        try:
            arr = np.load(path_dots).astype(np.float32)
            _PATH_DOTS = arr if arr.size > 0 else None
        except:
            pass

    # --- 3. Key Logic: Extract Core Template ---
    if _CORE_TEMPLATE is None and os.path.exists(path_mold2):
        try:
            gold_raw = cv2.imread(path_mold2, cv2.IMREAD_GRAYSCALE)
            if gold_raw is None: return

            # 3.1 Find big frame on golden image and correct perspective
            _, gold_big_box = _find_big_mold(gold_raw)
            if gold_big_box is None: return

            goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])
            M_gold = _get_perspective_matrix(gold_big_box, goldW, goldH)
            gold_warped = cv2.warpPerspective(gold_raw, M_gold, (goldW, goldH))

            # 3.2 Get user drawn box 0
            box0 = _order_points(_TEMPLATES[0])
            x0, y0 = int(min(box0[:, 0])), int(min(box0[:, 1]))
            x1, y1 = int(max(box0[:, 0])), int(max(box0[:, 1]))

            # 3.3 "Inner Crop"
            # Calculate margin to crop 15% from sides.
            # Purpose: remove side wall shadows, keep only the flat bottom pattern.
            w_box = x1 - x0
            h_box = y1 - y0
            crop_ratio = 0.15

            margin_x = int(w_box * crop_ratio)
            margin_y = int(h_box * crop_ratio)

            core_x0 = max(0, x0 + margin_x)
            core_y0 = max(0, y0 + margin_y)
            core_x1 = min(goldW, x1 - margin_x)
            core_y1 = min(goldH, y1 - margin_y)

            # Save this clean template
            _CORE_TEMPLATE = gold_warped[core_y0:core_y1, core_x0:core_x1].copy()
            # Record its absolute coord in golden image (Reference point for offsets)
            _CORE_REF_TL = np.array([core_x0, core_y0], dtype=np.float32)

            print(f"[mold_detector] Core template 0 extracted: {_CORE_TEMPLATE.shape}")

        except Exception as e:
            print(f"[mold_detector] Template generation error: {e}")



def reload_golden_templates():
    """
    Reset all caches, force reload.
    Usually called after re-teaching.
    """
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS, _CORE_TEMPLATE
    _TEMPLATES = None
    _WARP_SIZE = None
    _PATH_POLY = None
    _PATH_DOTS = None
    _CORE_TEMPLATE = None
    _load_golden_and_paths()

def set_override_paths(polys=None, dots=None):
    """
    [NEW] Force set the paths from external source (e.g. JSON from ProductManager).
    This bypasses the default .npy loading.
    """
    global _PATH_POLY, _PATH_DOTS
    _PATH_POLY = polys
    _PATH_DOTS = dots
    print(f"[mold_detector] Manual path override: {len(polys) if polys else 0} polys, {len(dots) if dots else 0} dots")



def detect_mold_and_cells(gray: np.ndarray):
    """
    [Main Processing Function] Called every frame.
    1. Find big mold position.
    2. Correct perspective.
    3. Match each cell using core template.
    4. Calculate offset based on position, draw dispensing path.
    """
    _load_golden_and_paths()
    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- 1. Find Big Frame ---
    big_rect, big_box = _find_big_mold(gray)
    if big_box is None:
        return color_img, None, [], []

    # Draw big frame (Green)
    cv2.drawContours(color_img, [big_box], 0, (0, 255, 0), 3)

    if _TEMPLATES is None or _WARP_SIZE is None:
        return color_img, big_box, [], []

    # --- 2. Correct Current Frame ---
    # Calculate perspective matrix from current frame to golden size
    goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])
    pts = big_box.astype(np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0, 0], [goldW - 1, 0], [goldW - 1, goldH - 1], [0, goldH - 1]], dtype=np.float32)

    # M_cur: Current -> Golden
    M_cur = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    # Minv_cur: Golden -> Current (for projecting results back)
    Minv_cur = cv2.getPerspectiveTransform(dst_pts, rect_pts)

    # Generate corrected current view (Template matching is most accurate here)
    curr_warped = cv2.warpPerspective(gray, M_cur, (goldW, goldH))

    cell_boxes = []
    cell_centers = []

    # --- UI Display: Draw user drawn yellow boxes (Green thick line) ---
    for idx in range(_TEMPLATES.shape[0]):
        box_w = _TEMPLATES[idx].astype(np.float32)
        pts_w = box_w.reshape(-1, 1, 2)
        if pts_w.size > 0:
            # Use Minv to project golden coords back to current frame coords
            pts_o = cv2.perspectiveTransform(pts_w, Minv_cur)
            if pts_o is not None:
                box_o = pts_o.reshape(-1, 2).astype(np.int32)
                cell_boxes.append(box_o)
                cx, cy = int(box_o[:, 0].mean()), int(box_o[:, 1].mean())
                cell_centers.append((cx, cy))
                # Draw box, thickness 5
                cv2.drawContours(color_img, [box_o], 0, (0, 255, 0), 5)

    # --- 3. Core Template Matching & Path Generation ---
    if (_PATH_POLY is not None or _PATH_DOTS is not None) and _CORE_TEMPLATE is not None:

        th, tw = _CORE_TEMPLATE.shape[:2]
        search_pad = 60  # Expand search range by 60 pixels

        for idx in range(_TEMPLATES.shape[0]):
            # Get search center (center of user drawn box)
            # We only use it to determine a rough range, not relying on its precise position
            guess_box = _order_points(_TEMPLATES[idx])
            bx = int((min(guess_box[:, 0]) + max(guess_box[:, 0])) / 2)
            by = int((min(guess_box[:, 1]) + max(guess_box[:, 1])) / 2)

            # Determine ROI (Region of Interest)
            rx = max(0, bx - search_pad - tw // 2)
            ry = max(0, by - search_pad - th // 2)
            rw = tw + 2 * search_pad
            rh = th + 2 * search_pad

            # Boundary check
            if rx + rw > goldW: rw = goldW - rx
            if ry + rh > goldH: rh = goldH - ry

            # Crop ROI
            roi = curr_warped[ry:ry + rh, rx:rx + rw]

            match_success = False
            offset = np.array([0, 0], dtype=np.float32)

            # --- Execute Template Matching ---
            if roi.shape[0] >= th and roi.shape[1] >= tw:
                res = cv2.matchTemplate(roi, _CORE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                # Match threshold 0.3
                if max_val > 0.3:
                    match_success = True
                    # Restore match point to full image coords (Template Top-Left)
                    match_x = rx + max_loc[0]
                    match_y = ry + max_loc[1]
                    current_live_tl = np.array([match_x, match_y], dtype=np.float32)

                    # ⭐ Core Formula: Calculate Physical Offset ⭐
                    # Offset = (Current Found Coord) - (Ref Coord 0)
                    # This implements true "position snapping", completely independent of hand-drawn box errors.
                    offset = current_live_tl - _CORE_REF_TL
                else:
                    # Match failed
                    pass

            # --- Fallback Logic ---
            # If really not matched (rare), fallback to using hand-drawn box relative displacement
            if not match_success:
                box0 = _order_points(_TEMPLATES[0])
                c0_x = (min(box0[:, 0]) + max(box0[:, 0])) / 2
                c0_y = (min(box0[:, 1]) + max(box0[:, 1])) / 2
                offset = np.array([bx - c0_x, by - c0_y], dtype=np.float32)

            # --- 4. Draw Paths ---
            # Add offset to Path 0, then project back to screen

            # Draw Polylines (Thickness 5, Green)
            if _PATH_POLY is not None:
                # Iterate each segment
                for seg in _PATH_POLY:
                    pts_gold = seg + offset
                    pts_w = pts_gold.reshape(-1, 1, 2)
                    res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                    if res_o is not None:
                        pts = res_o.reshape(-1, 2).astype(np.int32)
                        for i in range(len(pts) - 1):
                            cv2.line(color_img, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 0), 5, cv2.LINE_AA)

            # Draw Dots (Radius 10, Green)
            if _PATH_DOTS is not None and len(_PATH_DOTS) > 0:
                pts_gold = _PATH_DOTS + offset
                pts_w = pts_gold.reshape(-1, 1, 2)
                res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                if res_o is not None:
                    pts = res_o.reshape(-1, 2).astype(np.int32)
                    for px, py in pts:
                        cv2.circle(color_img, (int(px), int(py)), 10, (0, 255, 0), -1)

    return color_img, big_box, cell_boxes, cell_centers


def get_calculated_paths(gray: np.ndarray):
    """
    Interface for UI:
    Returns dispensing paths (pixel coords) for all detected cells in current frame.
    Format: [ path1, path2, ... ]
              where path1 is (N, 2) numpy array
    """
    # Ensure env loaded
    _load_golden_and_paths()

    # 1. Find big frame & matrix
    rect, big_box = _find_big_mold(gray)
    if big_box is None or _TEMPLATES is None or _WARP_SIZE is None:
        return []

    goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])

    # Calculate Current -> Golden matrix
    pts = big_box.astype(np.float32)
    s = pts.sum(axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0, 0], [goldW - 1, 0], [goldW - 1, goldH - 1], [0, goldH - 1]], dtype=np.float32)

    # We need Minv: Golden -> Current
    Minv_cur = cv2.getPerspectiveTransform(dst_pts, rect_pts)

    # 2. Prepare corrected image for matching
    M_cur = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    curr_warped = cv2.warpPerspective(gray, M_cur, (goldW, goldH))

    all_paths_pixels = []

    # 3. Iterate each cell, calculate path
    if (_PATH_POLY is not None) and (_CORE_TEMPLATE is not None):
        th, tw = _CORE_TEMPLATE.shape[:2]
        search_pad = 60

        for idx in range(_TEMPLATES.shape[0]):
            product_paths = [] # Paths for this specific product

            # --- A. Template Match for Offset ---
            guess_box = _order_points(_TEMPLATES[idx])
            bx = int((min(guess_box[:, 0]) + max(guess_box[:, 0])) / 2)
            by = int((min(guess_box[:, 1]) + max(guess_box[:, 1])) / 2)

            rx = max(0, bx - search_pad - tw // 2)
            ry = max(0, by - search_pad - th // 2)
            rw, rh = tw + 2 * search_pad, th + 2 * search_pad
            if rx + rw > goldW: rw = goldW - rx
            if ry + rh > goldH: rh = goldH - ry

            roi = curr_warped[ry:ry + rh, rx:rx + rw]

            offset = np.array([0, 0], dtype=np.float32)
            match_success = False

            if roi.shape[0] >= th and roi.shape[1] >= tw:
                res = cv2.matchTemplate(roi, _CORE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.3:
                    match_success = True
                    match_x = rx + max_loc[0]
                    match_y = ry + max_loc[1]
                    current_live_tl = np.array([match_x, match_y], dtype=np.float32)
                    offset = current_live_tl - _CORE_REF_TL

            # Fallback
            if not match_success:
                box0 = _order_points(_TEMPLATES[0])
                c0_x = (min(box0[:, 0]) + max(box0[:, 0])) / 2
                c0_y = (min(box0[:, 1]) + max(box0[:, 1])) / 2
                offset = np.array([bx - c0_x, by - c0_y], dtype=np.float32)

            # --- B. Generate Pixel Paths ---
            # Iterate each segment
            for seg in _PATH_POLY:
                pts_gold = seg + offset
                # Project back to current pixel coords
                pts_w = pts_gold.reshape(-1, 1, 2)
                res_o = cv2.perspectiveTransform(pts_w, Minv_cur)  # (N, 1, 2)

                if res_o is not None:
                    product_paths.append(res_o.reshape(-1, 2))  # (N, 2)

            # --- C. Generate Dot Paths ---
            if _PATH_DOTS is not None and len(_PATH_DOTS) > 0:
                pts_gold = _PATH_DOTS + offset
                pts_w = pts_gold.reshape(-1, 1, 2)
                res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                if res_o is not None:
                    # res_o is (N, 1, 2). Each point is a dot.
                    # We want to add each dot as a separate path of shape (1, 2)
                    dots = res_o.reshape(-1, 2)
                    for dot in dots:
                        product_paths.append(dot.reshape(1, 2))
            
            if product_paths:
                all_paths_pixels.append(product_paths)

    return all_paths_pixels


def get_paths_and_image(gray: np.ndarray):
    """
    Returns both the calculated paths and the visualized debug image.
    """
    # Reuse the logic of get_calculated_paths but also return the image
    # Note: get_calculated_paths creates its own local variables. 
    # To avoid code duplication, we can copy the logic or refactor.
    # For safety and speed now, I will copy the logic of get_calculated_paths 
    # and add the image drawing from detect_mold_and_cells.
    
    _load_golden_and_paths()
    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 1. Find big frame
    rect, big_box = _find_big_mold(gray)
    if big_box is None or _TEMPLATES is None or _WARP_SIZE is None:
        return [], color_img

    # Draw big frame
    cv2.drawContours(color_img, [big_box], 0, (0, 255, 0), 3)

    goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])

    # Matrix
    pts = big_box.astype(np.float32)
    s = pts.sum(axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0, 0], [goldW - 1, 0], [goldW - 1, goldH - 1], [0, goldH - 1]], dtype=np.float32)

    Minv_cur = cv2.getPerspectiveTransform(dst_pts, rect_pts)
    M_cur = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    curr_warped = cv2.warpPerspective(gray, M_cur, (goldW, goldH))

    all_paths_pixels = []

    if (_PATH_POLY is not None) and (_CORE_TEMPLATE is not None):
        th, tw = _CORE_TEMPLATE.shape[:2]
        search_pad = 60

        for idx in range(_TEMPLATES.shape[0]):
            product_paths = [] 
            
            # Draw User Box (Green Thick)
            box_w = _TEMPLATES[idx].astype(np.float32)
            pts_w = box_w.reshape(-1, 1, 2)
            pts_o = cv2.perspectiveTransform(pts_w, Minv_cur)
            if pts_o is not None:
                box_o = pts_o.reshape(-1, 2).astype(np.int32)
                cv2.drawContours(color_img, [box_o], 0, (0, 255, 0), 5)

            # Template Match
            guess_box = _order_points(_TEMPLATES[idx])
            bx = int((min(guess_box[:, 0]) + max(guess_box[:, 0])) / 2)
            by = int((min(guess_box[:, 1]) + max(guess_box[:, 1])) / 2)

            rx = max(0, bx - search_pad - tw // 2)
            ry = max(0, by - search_pad - th // 2)
            rw, rh = tw + 2 * search_pad, th + 2 * search_pad
            if rx + rw > goldW: rw = goldW - rx
            if ry + rh > goldH: rh = goldH - ry

            roi = curr_warped[ry:ry + rh, rx:rx + rw]
            offset = np.array([0, 0], dtype=np.float32)
            match_success = False

            if roi.shape[0] >= th and roi.shape[1] >= tw:
                res = cv2.matchTemplate(roi, _CORE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.3:
                    match_success = True
                    match_x = rx + max_loc[0]
                    match_y = ry + max_loc[1]
                    current_live_tl = np.array([match_x, match_y], dtype=np.float32)
                    offset = current_live_tl - _CORE_REF_TL

            if not match_success:
                box0 = _order_points(_TEMPLATES[0])
                c0_x = (min(box0[:, 0]) + max(box0[:, 0])) / 2
                c0_y = (min(box0[:, 1]) + max(box0[:, 1])) / 2
                offset = np.array([bx - c0_x, by - c0_y], dtype=np.float32)

            # Generate Paths
            for seg in _PATH_POLY:
                pts_gold = seg + offset
                pts_w = pts_gold.reshape(-1, 1, 2)
                res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                if res_o is not None:
                    product_paths.append(res_o.reshape(-1, 2))
                    
                    # Draw Path on Image (Green Thin)
                    pts = res_o.reshape(-1, 2).astype(np.int32)
                    cv2.polylines(color_img, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)

            if _PATH_DOTS is not None and len(_PATH_DOTS) > 0:
                pts_gold = _PATH_DOTS + offset
                pts_w = pts_gold.reshape(-1, 1, 2)
                res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                if res_o is not None:
                    dots = res_o.reshape(-1, 2)
                    for dot in dots:
                        product_paths.append(dot.reshape(1, 2))
                        # Draw Dot
                        cv2.circle(color_img, (int(dot[0]), int(dot[1])), 5, (0, 255, 0), -1)

            if product_paths:
                all_paths_pixels.append(product_paths)

    return all_paths_pixels, color_img
