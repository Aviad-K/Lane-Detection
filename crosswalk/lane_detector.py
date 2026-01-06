import cv2
import numpy as np
from collections import deque

class LaneDetector:
    """
    Detects lanes using Polar Coordinates with Geometric Coupling ("Buddy System").
    Fixes: 
    1. Directional Suppression: Prevents false 'return' warnings (e.g., warning Right 
       immediately after a Left swap) by waiting for the car to cross the lane center first.
    2. Dynamic Tolerance & Warning States preserved.
    """
    
    def __init__(
        self,
        # 1. Base Config
        roi_config: dict,
        initial_search_config: dict = None,
        
        # 2. Hough Transform
        hough_threshold: int = 15,
        min_line_length: int = 20,
        max_line_gap: int = 300,
        
        # 3. Filtering & Tracking Logic
        horizontal_rejection_deg: float = 74.5,
        track_match_angle_deg: float = 17.2,
        track_merge_angle_deg: float = 14.3,
        sleeve_width: int = 40,                  
        sleeve_expansion_factor: float = 2.0,    
        
        # 4. Smoothing & Inertia
        smoothing_frames: int = 10,
        anchor_weight: float = 5.0,              
        coasting_anchor_weight: float = 1000.0,  
        
        # 5. Lane Change / Reflection Logic
        lane_change_threshold: int = 60,         
        lane_change_vertical_angle_deg: float = 4.6,
        lane_change_center_tolerance_px: int = 250,  
        
        # 6. Stability Timers
        swap_cooldown_frames: int = 30,          
        anchor_suspension_frames: int = 10,      
        
        # 7. Visual Warnings
        vertical_warning_angle_deg: float = 12.0,
        safe_threshold_deg: float = 15.0,
        
        # 8. Calibration & Rejection
        calibration_frames: int = 30,
        avg_lane_width: int = 800,
        consistency_tolerance: int = 50, 
        
        # 9. Dynamic Tolerance Tuning
        drift_threshold_px: int = 100,            
        drift_tolerance_multiplier: float = 4.0,

        max_loss_frames: int = 50   
    ):
        self.roi_config = roi_config
        self.search_regions = initial_search_config

        # Hough
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        
        # Angle Thresholds
        self.horizontal_rejection_rad = np.deg2rad(horizontal_rejection_deg)
        self.track_match_angle_rad = np.deg2rad(track_match_angle_deg)
        self.track_merge_angle_rad = np.deg2rad(track_merge_angle_deg)
        self.vertical_swap_rad = np.deg2rad(lane_change_vertical_angle_deg)
        
        # Warning Thresholds
        self.warning_trigger_rad = np.deg2rad(vertical_warning_angle_deg)
        self.warning_clear_rad = np.deg2rad(safe_threshold_deg)
        
        # Tracking Params
        self.sleeve_width = sleeve_width
        self.sleeve_expansion_factor = sleeve_expansion_factor
        self.anchor_weight = anchor_weight
        self.coasting_anchor_weight = coasting_anchor_weight
        self.lane_change_threshold = lane_change_threshold
        self.center_tolerance_px = lane_change_center_tolerance_px
        
        # Timers
        self.swap_cooldown_target = swap_cooldown_frames
        self.anchor_suspension_target = anchor_suspension_frames
        
        # State
        self.left_fit = None   
        self.right_fit = None  
        self.left_history = deque(maxlen=smoothing_frames)
        self.right_history = deque(maxlen=smoothing_frames)
        self.avg_lane_width = avg_lane_width 
        self.consistency_tolerance = consistency_tolerance
        
        # Dynamic Tolerance Params
        self.drift_threshold_px = drift_threshold_px
        self.drift_tolerance_multiplier = drift_tolerance_multiplier
        
        self.calibration_buffer = [] 
        self.baseline_offset = 0.0
        self.is_calibrated = False
        self.calibration_target = calibration_frames
        self.current_offset = 9999.0 
        
        # NEW: Suppression State to block false alarms during recovery
        self.suppress_direction = None 
        
        self.last_search_mask = None 
        self.debug_segments_used = []
        self.debug_ghost_zones = [] 
        
        # Active Timers & State
        self.swap_cooldown = 0
        self.anchor_suspension_counter = 0 
        self.warning_state = None 

        self.max_loss_frames = max_loss_frames
        self.loss_counter = 0
        
        self.EPSILON = 1e-4

    def reset(self):
        self.left_fit = None
        self.right_fit = None
        self.left_history.clear()
        self.right_history.clear()
        self.swap_cooldown = 0
        self.anchor_suspension_counter = 0
        self.warning_state = None
        self.current_offset = 9999.0
        self.suppress_direction = None

    def detect(self, binary_mask: np.ndarray, roi_top_y_override: int = None) -> tuple:
        height, width = binary_mask.shape[:2]
        self.debug_segments_used = []
        self.debug_ghost_zones = [] 
        
        if self.swap_cooldown > 0: self.swap_cooldown -= 1
        if self.anchor_suspension_counter > 0: self.anchor_suspension_counter -= 1
        
        # 1. DETERMINE HORIZON (Same logic as process_video.py)
        # We use the override (crop_y) if provided to ensure we match the visualizer exactly.
        if roi_top_y_override is not None:
            roi_top = roi_top_y_override
        elif 'crop_height' in self.roi_config:
            roi_top = int(height * self.roi_config['crop_height'])
        else:
            roi_top = int(height * 0.6)
        
        roi_bottom = height 

        # 2. GENERATE SEARCH MASK 
        # This now draws the EXACT trapezoids you see in the "SEARCH" view.
        self.last_search_mask = np.zeros_like(binary_mask)
        self._get_search_mask(self.last_search_mask, height, width, roi_top, is_left=True)
        self._get_search_mask(self.last_search_mask, height, width, roi_top, is_left=False)
            
        # 3. HOUGH TRANSFORM (The Search)
        # We mask the image FIRST. This guarantees we only find lines inside your trapezoids.
        masked_binary = cv2.bitwise_and(binary_mask, self.last_search_mask)
        
        lines = cv2.HoughLinesP(
            masked_binary,
            rho=1, theta=np.pi/180, threshold=self.hough_threshold,
            minLineLength=self.min_line_length, maxLineGap=self.max_line_gap
        )
        
        # 4. COLLECT LINES
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2 and y1 == y2: continue
                
                # Sanity Check: Ignore lines fully above the crop line (should be handled by mask, but safe to keep)
                if min(y1, y2) < roi_top: continue

                rho, theta = self._cartesian_to_polar(x1, y1, x2, y2)
                
                # Global Angle Filter
                if abs(theta) > self.horizontal_rejection_rad: continue 

                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                x_bottom_candidate = self._get_x_at_y((rho, theta), height)

                is_left = False
                is_right = False
                
                # --- CHECK LEFT ---
                if self.left_fit:
                    # TRACKING MODE: Is it near the existing lane?
                    if self._is_near_track(self.left_fit, rho, theta): 
                        if self._is_consistent_with_partner(x_bottom_candidate, height, is_left=True):
                            is_left = True
                else:
                    # INITIAL SEARCH:
                    # We trust the Mask. If the line is on the left half of the screen, it is the Left Lane.
                    # We REMOVED '_is_in_search_zone' because the Mask (Step 2) already enforced the zone.
                    if x_bottom_candidate < (width // 2):
                        is_left = True
                    
                # --- CHECK RIGHT ---
                if self.right_fit:
                    # TRACKING MODE
                    if self._is_near_track(self.right_fit, rho, theta): 
                         if self._is_consistent_with_partner(x_bottom_candidate, height, is_left=False):
                            is_right = True
                else:
                    # INITIAL SEARCH
                    if x_bottom_candidate >= (width // 2):
                        is_right = True
                
                if is_left:
                    left_lines.append((rho, theta, length))
                    self.debug_segments_used.append(line[0])
                elif is_right:
                    right_lines.append((rho, theta, length))
                    self.debug_segments_used.append(line[0])

        # We check if we found ANY fresh data for either lane.
        # If we didn't find new lines, we might be coasting on old data. 
        # If we coast too long, we reset.
        if not left_lines or not right_lines:
            self.loss_counter += 1
            if self.loss_counter > self.max_loss_frames:
                self.reset()
                # We return immediately to avoid updating with stale data this frame
                return None, None, False, None, 0.0, None, None 
        else:
            self.loss_counter = 0

        # 5. UPDATE LANES
        self.left_fit = self._update_lane(left_lines, self.left_fit, self.left_history)
        self.right_fit = self._update_lane(right_lines, self.right_fit, self.right_history)
        
        # 6. OFFSET, SWAP, WARNINGS (Standard Logic)
        offset, is_lane_change, direction, lane_center, car_center = self._calculate_offset(width, roi_bottom)
        
        if self.swap_cooldown == 0 and car_center is not None and (self.left_fit or self.right_fit):
            self._check_and_swap_lanes(car_center, roi_bottom)
            
        self._update_warning_state()
        
        left_coords = self._polar_to_cartesian(self.left_fit, roi_bottom, roi_top)
        right_coords = self._polar_to_cartesian(self.right_fit, roi_bottom, roi_top)
        
        return left_coords, right_coords, is_lane_change, direction, offset, lane_center, car_center

    # =========================================================================
    # WARNING LOGIC
    # =========================================================================
    
    def _update_warning_state(self):
        current_trigger = None
        
        if self.left_fit and abs(self.left_fit[1]) < self.warning_trigger_rad:
            current_trigger = "LEFT"
        elif self.right_fit and abs(self.right_fit[1]) < self.warning_trigger_rad:
            current_trigger = "RIGHT"
            
        # SUPPRESSION CHECK:
        # If we just moved Left, suppress "RIGHT" warnings until we stabilize (cross center).
        if self.suppress_direction:
            if current_trigger == self.suppress_direction:
                current_trigger = None

        if self.warning_state:
            is_safe = False
            if self.warning_state == "LEFT":
                if not self.left_fit or abs(self.left_fit[1]) > self.warning_clear_rad: is_safe = True
            elif self.warning_state == "RIGHT":
                if not self.right_fit or abs(self.right_fit[1]) > self.warning_clear_rad: is_safe = True
            
            if is_safe: self.warning_state = None 
        else:
            if current_trigger: self.warning_state = current_trigger

    # =========================================================================
    # GEOMETRIC COUPLING
    # =========================================================================

    def _is_consistent_with_partner(self, x_bottom, height, is_left):
        if self.swap_cooldown > 0: return True
        if not self.is_calibrated or self.avg_lane_width == 0: return True
            
        active_tolerance = self.consistency_tolerance
        
        is_high_drift = abs(self.current_offset) > self.drift_threshold_px
        is_warning_active = (self.warning_state is not None)
        
        if is_high_drift or is_warning_active:
            active_tolerance = self.consistency_tolerance * self.drift_tolerance_multiplier 
            
        partner_fit = self.right_fit if is_left else self.left_fit
        if partner_fit is None: return True
            
        partner_x = self._get_x_at_y(partner_fit, height)
        expected_x = partner_x - self.avg_lane_width if is_left else partner_x + self.avg_lane_width
            
        zone_data = (int(expected_x), int(height), int(active_tolerance))
        if zone_data not in self.debug_ghost_zones: self.debug_ghost_zones.append(zone_data)
        
        return abs(x_bottom - expected_x) < active_tolerance

    # =========================================================================
    # LANE SWAP LOGIC
    # =========================================================================

    def _check_and_swap_lanes(self, car_center, height):
        if self.left_fit:
            rho_l, theta_l = self.left_fit
            lx = self._get_x_at_y(self.left_fit, height)
            if abs(theta_l) < self.vertical_swap_rad and abs(lx - car_center) < self.center_tolerance_px:
                print("[EVENT] Left Lane Vertical! Reflecting...")
                new_right_fit = self.left_fit
                if self.right_fit:
                    new_left_fit = self._reflect_lane_polar(self.right_fit, axis_x=lx, height=height)
                else:
                    width = self.avg_lane_width
                    new_rho, new_theta = self.left_fit
                    new_left_fit = (new_rho - (width * np.cos(new_theta)), new_theta)
                self.right_fit = new_right_fit
                self.left_fit = new_left_fit
                self._trigger_swap(is_left_swap=True)
                return 

        if self.right_fit:
            rho_r, theta_r = self.right_fit
            rx = self._get_x_at_y(self.right_fit, height)
            if abs(theta_r) < self.vertical_swap_rad and abs(rx - car_center) < self.center_tolerance_px:
                print("[EVENT] Right Lane Vertical! Reflecting...")
                new_left_fit = self.right_fit
                if self.left_fit:
                    new_right_fit = self._reflect_lane_polar(self.left_fit, axis_x=rx, height=height)
                else:
                    width = self.avg_lane_width
                    new_rho, new_theta = self.right_fit
                    new_right_fit = (new_rho + (width * np.cos(new_theta)), new_theta)
                self.left_fit = new_left_fit
                self.right_fit = new_right_fit
                self._trigger_swap(is_left_swap=False)
                return

    def _trigger_swap(self, is_left_swap):
        self._reset_history()
        self.swap_cooldown = self.swap_cooldown_target
        self.anchor_suspension_counter = self.anchor_suspension_target
        self.warning_state = None
        
        # Force Loose Mode for acquisition
        self.current_offset = 9999.0 
        
        # SUPPRESSION LOGIC:
        # If we moved Left, we are entering from the Right side.
        # We must suppress "MOVING RIGHT" warnings until we cross the center.
        if is_left_swap:
            self.suppress_direction = "RIGHT"
        else:
            self.suppress_direction = "LEFT"

    def _reflect_lane_polar(self, source_fit, axis_x, height):
        rho, theta = source_fit
        new_theta = -theta
        x_old = self._get_x_at_y(source_fit, height)
        x_new = (2 * axis_x) - x_old
        new_rho = x_new * np.cos(new_theta) + height * np.sin(new_theta)
        return (new_rho, new_theta)

    def _reset_history(self):
        self.right_history.clear()
        self.left_history.clear()
        if self.right_fit: self.right_history.append(self.right_fit)
        if self.left_fit: self.left_history.append(self.left_fit)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _update_lane(self, lines, current_fit, history):
        total_rho, total_theta, total_weight = 0, 0, 0
        for rho, theta, length in lines:
            if current_fit and abs(theta - current_fit[1]) > self.track_merge_angle_rad: continue
            total_rho += rho * length
            total_theta += theta * length
            total_weight += length
            
        found_real_lines = (total_weight > 0)
        
        use_anchor = (current_fit is not None) and (self.anchor_suspension_counter == 0)
        if use_anchor:
            w = (total_weight * self.anchor_weight) if total_weight > 0 else self.coasting_anchor_weight
            total_rho += current_fit[0] * w
            total_theta += current_fit[1] * w
            total_weight += w

        if total_weight == 0:
            return current_fit 
        
        avg_rho = total_rho / total_weight
        avg_theta = total_theta / total_weight
        history.append((avg_rho, avg_theta))
        return self._smooth(history)

    def _smooth(self, history):
        weights = np.arange(1, len(history) + 1)
        return (np.average([h[0] for h in history], weights=weights), np.average([h[1] for h in history], weights=weights))

    def _get_x_at_y(self, fit, y):
        rho, theta = fit
        if abs(np.cos(theta)) < self.EPSILON: return rho
        return (rho - y * np.sin(theta)) / np.cos(theta)

    def _cartesian_to_polar(self, x1, y1, x2, y2):
        if x2 == x1: theta_line = np.pi / 2
        else: theta_line = np.arctan((y2-y1) / (x2-x1))
        theta = theta_line - np.pi / 2
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        if theta < -np.pi / 2: theta += np.pi; rho = -rho
        elif theta > np.pi / 2: theta -= np.pi; rho = -rho
        return rho, theta

    def _polar_to_cartesian(self, fit, y_bottom, y_top):
        if fit is None: return None
        x_bottom = int(self._get_x_at_y(fit, y_bottom))
        x_top = int(self._get_x_at_y(fit, y_top))
        return ((x_bottom, y_bottom), (x_top, y_top))

    def _get_search_mask(self, mask, h, w, roi_top, is_left):
        """
        Draws the search area. 
        In Search Mode, this now matches the 'SEARCH' debug view in process_video.py EXACTLY.
        """
        fit = self.left_fit if is_left else self.right_fit
        
        # A. TRACKING MODE (Dynamic Sleeve)
        if fit:
            c = self._polar_to_cartesian(fit, h, 0)
            if c:
                pts = np.array([
                    [c[1][0]-self.sleeve_width, 0], 
                    [c[1][0]+self.sleeve_width, 0],
                    [c[0][0]+self.sleeve_width, h], 
                    [c[0][0]-self.sleeve_width, h]
                ], np.int32)
                cv2.fillPoly(mask, [pts], 255)
        
        # B. INITIAL SEARCH MODE (Static Trapezoids)
        else:
            cfg = self.search_regions['L' if is_left else 'R']
            
            # --- GEOMETRY FIX ---
            # We strictly use 'roi_top' as the top Y coordinate.
            # This matches "process_video.py" line: pts = np.array([[tx1, crop_y], ...])
            pts = np.array([
                [cfg['top_x'][0], roi_top], 
                [cfg['top_x'][1], roi_top], 
                [cfg['bottom_x'][1], h], 
                [cfg['bottom_x'][0], h]
            ], np.int32)
            
            cv2.fillPoly(mask, [pts], 255)

        # MANDATORY CROP:
        # Zero out everything above the horizon to ensure the mask doesn't leak upwards.
        if roi_top > 0:
            mask[0:roi_top, :] = 0

    def _is_in_search_zone(self, x1, y1, x2, y2, w, h, roi_top, side):
        """
        Checks if a line segment is within the trapezoid defined by 'roi_top'.
        """
        mx, my = (x1+x2)/2, (y1+y2)/2
        cfg = self.search_regions[side]
        
        # FIX: Calculate ratio based on the actual roi_top
        # Ratio 0.0 = Top (roi_top), Ratio 1.0 = Bottom (h)
        if h > roi_top:
            ratio = (my - roi_top) / (h - roi_top)
        else:
            ratio = 0
            
        min_x = cfg['top_x'][0] * (1-ratio) + cfg['bottom_x'][0] * ratio
        max_x = cfg['top_x'][1] * (1-ratio) + cfg['bottom_x'][1] * ratio
        
        return min_x < mx < max_x

    def _is_near_track(self, fit, rho, theta):
        if abs(theta - fit[1]) > self.track_match_angle_rad: return False
        rho_threshold = self.sleeve_width * self.sleeve_expansion_factor
        return abs(rho - fit[0]) < rho_threshold

    def _calculate_offset(self, width, height):
        bottom_shift = self.roi_config.get('bottom_s', 0) 
        car_center = (width // 2) + bottom_shift
        lx, rx = None, None
        if self.left_fit: lx = self._get_x_at_y(self.left_fit, height)
        if self.right_fit: rx = self._get_x_at_y(self.right_fit, height)
            
        if lx and rx:
            cw = rx - lx
            if 600 < cw < 1400: self.avg_lane_width = (self.avg_lane_width * 0.95) + (cw * 0.05)
        lane_center = None
        if lx and rx: lane_center = (lx + rx) / 2
        elif lx: lane_center = lx + (self.avg_lane_width / 2)
        elif rx: lane_center = rx - (self.avg_lane_width / 2)
        
        if lane_center is not None:
            raw = lane_center - car_center
            self.current_offset = raw 
            
            # CHECK: Did we cross the center? If so, lift suppression.
            # Moving Left: suppress_direction="RIGHT", offset changes from Neg to Pos.
            if self.suppress_direction == "RIGHT" and raw > 0:
                self.suppress_direction = None
            # Moving Right: suppress_direction="LEFT", offset changes from Pos to Neg.
            elif self.suppress_direction == "LEFT" and raw < 0:
                self.suppress_direction = None
                
        else:
            raw = 0.0 
        
        if lane_center is None: return 0.0, False, None, None, car_center
        
        if not self.is_calibrated:
            self.calibration_buffer.append(raw)
            if len(self.calibration_buffer) >= self.calibration_target:
                self.baseline_offset = np.mean(self.calibration_buffer)
                self.is_calibrated = True
            return raw, False, "CALIB", lane_center, car_center
        dev = raw - self.baseline_offset
        is_change = abs(dev) > self.lane_change_threshold
        direction = "LEFT" if dev > 0 else "RIGHT"
        return dev, is_change, direction, lane_center, car_center
    
    def get_vanishing_point(self):
        """
        Calculates the (x, y) intersection of the current left and right fits.
        Returns None if lines are parallel or fits are missing.
        """
        if self.left_fit is None or self.right_fit is None:
            return None
            
        rho1, theta1 = self.left_fit
        rho2, theta2 = self.right_fit
        
        # Formulate as matrix equation: A * [x, y] = B
        # Line 1: x cos(t1) + y sin(t1) = rho1
        # Line 2: x cos(t2) + y sin(t2) = rho2
        
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        
        B = np.array([rho1, rho2])
        
        # Check determinant to avoid singular matrix (parallel lines)
        det = np.linalg.det(A)
        if abs(det) < 1e-4:
            return None
            
        intersection = np.linalg.solve(A, B)
        return int(intersection[0]), int(intersection[1]) # (x, y)

    def create_debug_view(self, mask, left, right):
        debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if self.last_search_mask is not None:
            overlay = cv2.cvtColor(self.last_search_mask, cv2.COLOR_GRAY2BGR)
            debug = cv2.addWeighted(debug, 1.0, overlay, 0.3, 0)
        
        for s in self.debug_segments_used: 
            cv2.line(debug, (s[0], s[1]), (s[2], s[3]), (0,255,255), 2)
        
        for (gx, gy, g_tol) in self.debug_ghost_zones:
            color = (255, 255, 0) if g_tol <= self.consistency_tolerance else (0, 165, 255)
            top_left = (gx - g_tol, gy - 50)
            bot_right = (gx + g_tol, gy)
            cv2.rectangle(debug, top_left, bot_right, color, 2)
            cv2.putText(debug, f"GHOST {g_tol}", (gx - 20, gy - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if left: cv2.line(debug, left[0], left[1], (255,0,0), 3)
        if right: cv2.line(debug, right[0], right[1], (0,0,255), 3)
        
        return debug
        
    def draw_lanes(self, frame, left, right, is_lane_change=False, line_color=(0,255,255), fill_color=(0,100,0), line_thickness=5):
        output = frame.copy()
        if left and right:
            pts = np.array([left[0], left[1], right[1], right[0]], np.int32)
            cv2.fillPoly(output, [pts], fill_color)
            frame = cv2.addWeighted(output, 0.4, frame, 0.6, 0)
        if left: cv2.line(frame, left[0], left[1], line_color, line_thickness)
        if right: cv2.line(frame, right[0], right[1], line_color, line_thickness)
        
        if self.swap_cooldown == 0 and self.warning_state:
            text = "<<< MOVING LEFT" if self.warning_state == "LEFT" else "MOVING RIGHT >>>"
            self._draw_warning_text(frame, text)
        return frame

    def _draw_warning_text(self, frame, text):
        font_scale = 1.5
        thickness = 4
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = int(frame.shape[0] * 0.3)
        cv2.putText(frame, text, (text_x + 3, text_y + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)