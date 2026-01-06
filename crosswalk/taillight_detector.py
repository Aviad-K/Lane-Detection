import cv2
import numpy as np
from lane_detector import LaneDetector

class TaillightDetector:
    def __init__(self, config: dict, detector: LaneDetector):
        self.cfg = config
        self.focal_length = config.get('focal_length', 2800)
        self.horizon_y = config.get('horizon_y', 540) 
        
        self.min_area = config.get('min_light_area', 10)
        self.max_area = config.get('max_light_area', 800)
        self.y_tolerance = config.get('y_tolerance', 50)
        self.max_pair_width = config.get('max_pair_width', 600)
        
        # Lane centering
        self.lane_sample_ratio = config.get('lane_sample_ratio', 0.5)
        self.morph_kernel = config.get('morph_kernel_size', (3,3))

        # NEW: Magic Number Replacement
        # Default to 0.7 if not found in config
        self.lights_height_ratio = config.get('lights_height_ratio', 0.7)

        self.detector = detector

    def detect(self, frame, left_lane=None, right_lane=None):
        h, w = frame.shape[:2]
        
        # --- 1. SETUP ---
        tx_min, tx_max = self.cfg['top_x']
        bx_min, bx_max = self.cfg['bottom_x']
        top_y = self.cfg['search_top']
        
        ref_w = self.cfg.get('car_width_at_bottom', 900)
        ref_h = self.cfg.get('car_height_at_bottom', 600)
        
        c_btm = self.cfg.get('center_bottom_x', w // 2)
        c_top = self.cfg.get('center_top_x', w // 2)
        
        if left_lane is not None and right_lane is not None:
            lane_top_y = left_lane[1][1] 
            sample_y = int(h - (h - lane_top_y) * self.lane_sample_ratio)
            lx = self._get_x_on_line(left_lane, sample_y)
            rx = self._get_x_on_line(right_lane, sample_y)
            if lx is not None and rx is not None:
                c_top = int((lx + rx) / 2)

        # --- 2. MASKS ---
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[tx_min, top_y], [tx_max, top_y], [bx_max, h], [bx_min, h]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l1, u1 = np.array(self.cfg['red_low_1']), np.array(self.cfg['red_high_1'])
        l2, u2 = np.array(self.cfg['red_low_2']), np.array(self.cfg['red_high_2'])
        raw_red = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
        clean_red = cv2.morphologyEx(cv2.bitwise_and(raw_red, mask), cv2.MORPH_OPEN, np.ones(self.morph_kernel, np.uint8))
        clean_red = cv2.morphologyEx(clean_red, cv2.MORPH_DILATE, np.ones(self.morph_kernel, np.uint8))

        # Force Split (Black Line)
        cv2.line(clean_red, (c_top, top_y), (c_btm, h), 0, 6)

        # --- 3. CONTOURS ---
        contours, _ = cv2.findContours(clean_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        roi_debug = frame.copy()
        cv2.polylines(roi_debug, [pts], True, (0, 0, 255), 2)
        cv2.line(roi_debug, (c_top, top_y), (c_btm, h), (0, 255, 255), 2)

        line_h = h - top_y
        slope = (c_btm - c_top) / line_h if line_h != 0 else 0
        
        best_left = None
        best_right = None
        max_left_x = -float('inf') 
        min_right_x = float('inf')

        for cnt in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            cx = x + w_rect // 2
            cy = y + h_rect // 2
            
            if not (self.min_area < area < self.max_area):
                cv2.rectangle(roi_debug, (x, y), (x+w_rect, y+h_rect), (100, 100, 100), 1)
                continue
            
            cv2.rectangle(roi_debug, (x, y), (x+w_rect, y+h_rect), (255, 100, 0), 1)
            
            ref_x = int(c_top + slope * (cy - top_y))
            if cx < ref_x:
                if cx > max_left_x:
                    max_left_x = cx
                    best_left = (x, y, w_rect, h_rect)
            else:
                if cx < min_right_x:
                    min_right_x = cx
                    best_right = (x, y, w_rect, h_rect)

        # --- 4. CONSTRUCT BOX ---
        detected_cars = []
        if best_left and best_right:
            lx, ly, lw, lh = best_left
            rx, ry, rw, rh = best_right
            
            raw_w = (rx + rw) - lx
            lights_cy = int((ly + lh/2 + ry + rh/2) / 2)
            
            is_vert_aligned = abs((ly + lh/2) - (ry + rh/2)) < self.y_tolerance
            is_width_ok = raw_w < self.max_pair_width
            
            if is_vert_aligned and is_width_ok:
                
                center_x = int((lx + rx + rw) / 2)
                
                # --- SCALING LOGIC ---
                denom = max(1, h - self.horizon_y)
                curr_h = max(0, lights_cy - self.horizon_y)
                scale = curr_h / denom
                
                # 1. Calculate Ideal Math Width
                math_w = int(ref_w * scale)
                
                # 2. Apply "Safety Floor" (The Fix)
                # Ensure width is at least 'min_box_width' defined in config
                # Also ensure it's at least as wide as the lights themselves (raw_w)
                min_limit = self.cfg.get('min_box_width', 100)
                final_w = max(math_w, min_limit, int(raw_w * 1.1))
                
                # 3. Recalculate Height to maintain Aspect Ratio
                # If we forced the width to stay big, we must force the height to match
                effective_scale = final_w / ref_w
                final_h = int(ref_h * effective_scale)
                
                # --- POSITIONING ---
                # Position box so lights are near the bottom (according to ratio)
                offset_from_top = int(final_h * self.lights_height_ratio)
                final_y = int(lights_cy - offset_from_top)
                final_x = int(center_x - final_w / 2)
                
                cv2.rectangle(roi_debug, (final_x, final_y), (final_x+final_w, final_y+final_h), (0, 255, 0), 2)
                
                dist_m = self._calculate_distance(final_y + final_h)
                detected_cars.append((final_x, final_y, final_w, final_h, dist_m))
                
            else:
                if not is_width_ok:
                    cv2.line(roi_debug, (lx+lw, ly+lh//2), (rx, ry+rh//2), (255, 0, 255), 2)

        return detected_cars, pts, clean_red, roi_debug

    def _get_x_on_line(self, lane_segment, y):
        if not lane_segment: return None
        (x1, y1), (x2, y2) = lane_segment
        if y1 == y2: return x1 
        return int(x1 + (x2 - x1) * (y - y1) / (y2 - y1))

    def _calculate_distance(self, contact_y):
        """
        Calculates distance using the dynamic horizon.
        
        Args:
            contact_y (float): The Y-coordinate of the tire/road contact point (or bottom of lights).
        """
        # 1. Safety Check: 
        # If the object is above the horizon, it's mathematically at infinity or invalid.
        # We add a small buffer (+1) to avoid DivisionByZero errors.
        current_horizon_y = self.detector.get_vanishing_point()[1]
        if contact_y <= current_horizon_y + 1:
            return 999.0
            
        # 2. Perspective Projection Formula:
        # Distance = K / (y_screen - y_horizon)
        # K is your calibrated constant (1333)
        pixel_offset = contact_y - current_horizon_y
        dist_meters = self.focal_length / pixel_offset
        
        return round(dist_meters, 1)

    def draw(self, frame, cars):
        for (x, y, w, h, dist) in cars:
            color = (0, 255, 0)
            if dist < 12: color = (0, 255, 255)
            if dist < 7: color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{dist}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame