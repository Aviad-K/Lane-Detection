"""
Lane Preprocessor Module
=========================
Converts raw video frames into a clean binary mask.
Logic: "Locally Bright + Structural" (Top-Hat Approximation) with Median Denoising
"""

import cv2
import numpy as np


class LanePreprocessor:
    """
    Produces a clean binary mask from a raw video frame.
    """
    
    def __init__(self, roi_config: dict, 
                 blur_kernel: int = 5, 
                 edge_threshold: int = 30, 
                 local_average_kernel: int = 25, 
                 lightness_sensitivity: float = 0.05,
                 min_diff: int = 3,
                 mask_median_kernel: int = 5,
                 min_area_close: int = 500,  
                 min_area_far: int = 50,     
                 area_threshold_exponent: float = 4.0):
        self.roi = roi_config
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.local_avg_kernel = local_average_kernel if local_average_kernel % 2 == 1 else local_average_kernel + 1
        
        self.sensitivity = lightness_sensitivity
        self.min_diff = min_diff
        
        # Area thresholds
        self.min_area_close = min_area_close
        self.min_area_far = min_area_far
        self.area_exponent = area_threshold_exponent
        
        # Median Filter Kernel Setup
        self.mask_median_size = mask_median_kernel
        if self.mask_median_size > 0:
            k = self.mask_median_size if self.mask_median_size % 2 == 1 else self.mask_median_size + 1
            self.mask_median_size = max(3, k) 
        
        self.canny_low = edge_threshold
        self.canny_high = edge_threshold * 3
        
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    def process(self, frame: np.ndarray, detect_edges: bool = True) -> tuple:
        if frame is None:
            return None, None, None, None, None
        
        height, width = frame.shape[:2]
        
        # 1. Extract Lightness
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        
        # 2. Initial Blur
        blurred = cv2.medianBlur(l_channel, self.blur_kernel)
        
        # 3. Local Brightness (Top-Hat)
        local_avg = cv2.blur(blurred, (self.local_avg_kernel, self.local_avg_kernel))
        diff = cv2.subtract(blurred, local_avg)
        
        # 4. Adaptive Thresholding
        adaptive_threshold = (local_avg.astype(np.float32) * self.sensitivity).astype(np.uint8)
        final_threshold = np.maximum(adaptive_threshold, self.min_diff)
        brightness_mask = (diff > final_threshold).astype(np.uint8) * 255
        
        # 5. Median Filter
        if self.mask_median_size > 0:
            brightness_mask = cv2.medianBlur(brightness_mask, self.mask_median_size)
            
        # 6. Standard Cleanup
        brightness_mask = cv2.morphologyEx(brightness_mask, cv2.MORPH_OPEN, self.morph_kernel)
        brightness_mask = cv2.dilate(brightness_mask, self.morph_kernel, iterations=1)
        
        # 7. PERSPECTIVE AREA FILTER
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(brightness_mask, connectivity=8)
        
        filtered_mask = np.zeros_like(brightness_mask)
        
        # Prepare Visualization
        diff_vis_gray = np.clip(diff, 0, 255).astype(np.uint8)
        diff_vis_gray = cv2.normalize(diff_vis_gray, None, 0, 255, cv2.NORM_MINMAX)
        diff_vis_raw = cv2.cvtColor(diff_vis_gray, cv2.COLOR_GRAY2BGR)
        diff_vis_tuned = diff_vis_raw.copy()

        if num_labels > 1:
            roi_top_y = int(height * self.roi['crop_height'])
            roi_height = height - roi_top_y
            
            # Extract Data
            areas = stats[1:, cv2.CC_STAT_AREA]
            center_ys = centroids[1:, 1]
            
            # Vectorized Ratio
            ratios = (center_ys - roi_top_y) / roi_height
            ratios = np.clip(ratios, 0.0, 1.0) 
            ratios = np.power(ratios, self.area_exponent)
            
            # Vectorized Threshold
            required_areas = self.min_area_far + (self.min_area_close - self.min_area_far) * ratios
            
            # Determine Winners
            is_valid = areas >= required_areas
            valid_indices = np.where(is_valid)[0] + 1
            invalid_indices = np.where(~is_valid)[0] + 1
            
            # Mask Generation
            mask_lut = np.zeros(num_labels, dtype=np.uint8)
            mask_lut[valid_indices] = 255
            filtered_mask = mask_lut[labels]
            
            # Visualization Coloring
            diff_vis_tuned[filtered_mask == 255] = [0, 255, 0] # Green
            
            red_lut = np.zeros(num_labels, dtype=np.uint8)
            red_lut[invalid_indices] = 255
            red_mask = red_lut[labels]
            diff_vis_tuned[red_mask == 255] = [0, 0, 255] # Red
            
        brightness_mask = filtered_mask

        # 8. Edge Detection & Intersection
        if detect_edges:
            # ORIGINAL PATH: Force Canny Edges
            raw_edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            clean_output = cv2.bitwise_and(raw_edges, raw_edges, mask=brightness_mask)
        else:
            # NEW PATH: Return the Solid mask directly
            raw_edges = None # Not used
            clean_output = brightness_mask
        
        # 9. Geometry (UPDATED: Simple Rectangular Crop)
        roi_masked, roi_vertices = self._apply_roi_mask(clean_output)
        final_binary = self._remove_center_noise(roi_masked)
        
        return final_binary, raw_edges, roi_vertices, diff_vis_raw, diff_vis_tuned

    # -----------------------------------------------------------
    # Geometric Helpers
    # -----------------------------------------------------------
    def _apply_roi_mask(self, img: np.ndarray) -> tuple:
        """
        [UPDATED] Applies a simple rectangular crop defined by 'crop_height'.
        Ignores complex trapezoid parameters to allow wide-view tracking.
        """
        height, width = img.shape[:2]
        
        # Calculate cut-off Y
        y_top = int(height * self.roi['crop_height'])
        
        mask = np.zeros_like(img)
        # Draw a white rectangle from y_top to the bottom of the screen
        cv2.rectangle(mask, (0, y_top), (width, height), 255, -1)
        
        # Return vertices for visualization
        pts = np.array([[
            (0, height), (0, y_top), (width, y_top), (width, height)
        ]], dtype=np.int32)
        
        return cv2.bitwise_and(img, mask), pts
    
    def _remove_center_noise(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        pts = self.get_center_noise_vertices((h, w))
        
        if pts is None: 
            return img
            
        res = img.copy()
        cv2.fillPoly(res, pts, 0)
        return res
    
    def get_center_noise_vertices(self, shape: tuple) -> np.ndarray:
        h, w = shape[:2]
        cx = w // 2
        y_top = int(h * self.roi['crop_height'])
        tw, bw = self.roi['top_w'], self.roi['bottom_w']
        ts, bs = self.roi['top_s'], self.roi['bottom_s']
        tnr, bnr = self.roi['top_noise_ratio'], self.roi['bottom_noise_ratio']
        
        nw_top, nw_bot = int(tw * tnr), int(bw * bnr)
        
        # [UPDATED] If ratios are zero, disable the feature (Fixes upside-down triangle)
        if nw_top <= 1 and nw_bot <= 1:
            return None
            
        ts_cx, bs_cx = cx + ts, cx + bs
        
        return np.array([[
            (int(bs_cx - nw_bot/2), h), (int(ts_cx - nw_top/2), y_top),
            (int(ts_cx + nw_top/2), y_top), (int(bs_cx + nw_bot/2), h)
        ]], dtype=np.int32)

    def get_lane_exclusion_mask(self, shape: tuple, roi_top=None, roi_bottom=None) -> np.ndarray:
        height, width = shape[:2]
        if roi_top is None: roi_top = int(height * self.roi['crop_height'])
        if roi_bottom is None: roi_bottom = height
        return np.zeros((height, width), dtype=np.uint8)