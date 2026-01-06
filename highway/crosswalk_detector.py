import cv2
import numpy as np

class CrosswalkDetector:
    def __init__(self, config: dict):
        self.cfg = config
        
        # 1. ROI Config
        self.scan_top_y = config.get('scan_min_y', 800)
        self.scan_bottom_y = config.get('scan_max_y', 1080)
        self.scan_min_x = config.get('scan_min_x', 0)
        self.scan_max_x = config.get('scan_max_x', 1920)
        
        # 2. Blob Filtering
        self.min_area = config.get('min_blob_area', 1000)
        self.max_area = config.get('max_blob_area', 15000)
        self.min_solidity = config.get('min_solidity', 0.8)
        self.min_ar = config.get('min_aspect_ratio', 1.0)
        self.max_ar = config.get('max_aspect_ratio', 5.0)
        
        # 3. Clustering
        self.max_dist_x = config.get('max_neighbor_dist_x', 250)
        self.max_dist_y = config.get('max_neighbor_dist_y', 30)
        self.min_group_size = config.get('min_group_size', 3)

        # 4. ROBUST PERSISTENCE (The Anti-Flicker Logic)
        self.persistence_threshold = config.get('persistence_threshold', 3) # Frames to wait before showing
        self.max_loss_frames = config.get('max_loss_frames', 15)            # Frames to keep showing after loss
        
        # State
        self.detection_streak = 0       # Consecutive hits
        self.loss_counter = 0           # Consecutive misses
        self.is_active = False          # Are we currently alerting the user?
        
        self.smoothed_box = None        # The stable (y, h) we are tracking
        
        # Debug
        self.debug_candidates = [] 
        self.debug_connections = []

    def detect(self, frame, external_binary=None):
        # --- PHASE 1: Preprocessing ---
        if external_binary is not None:
            binary_mask = external_binary
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        self.debug_candidates = []
        self.debug_connections = []
        
        # --- PHASE 2: Blob Filtering ---
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_candidates = [] 
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area: continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Region Check
            if (y < self.scan_top_y) or (y + h > self.scan_bottom_y): continue
            
            # Shape Check
            aspect_ratio = float(w) / h
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            is_valid_shape = (self.min_ar <= aspect_ratio <= self.max_ar and solidity >= self.min_solidity)
            self.debug_candidates.append(((x,y,w,h), is_valid_shape))
            
            if is_valid_shape:
                valid_candidates.append({'rect': (x,y,w,h), 'center': (x + w//2, y + h//2)})

        # --- PHASE 3: Clustering ---
        # (Standard "Stepping Stone" Logic)
        candidate_box = None
        
        if len(valid_candidates) >= self.min_group_size:
            n = len(valid_candidates)
            adj = [[] for _ in range(n)]
            
            # Connect neighbors
            for i in range(n):
                for j in range(i + 1, n):
                    c1 = valid_candidates[i]
                    c2 = valid_candidates[j]
                    dx = abs(c1['center'][0] - c2['center'][0])
                    dy = abs(c1['center'][1] - c2['center'][1])
                    
                    if dx < self.max_dist_x and dy < self.max_dist_y:
                        adj[i].append(j)
                        adj[j].append(i)
                        self.debug_connections.append((c1['center'], c2['center']))
            
            # Find largest group
            visited = [False] * n
            best_group = []
            
            for i in range(n):
                if not visited[i]:
                    group = []
                    queue = [i]
                    visited[i] = True
                    while queue:
                        curr = queue.pop(0)
                        group.append(valid_candidates[curr])
                        for neighbor in adj[curr]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                queue.append(neighbor)
                    
                    if len(group) >= self.min_group_size and len(group) > len(best_group):
                        best_group = group
            
            # Calculate raw box from group
            if best_group:
                min_gy = min(c['rect'][1] for c in best_group)
                max_gy = max(c['rect'][1] + c['rect'][3] for c in best_group)
                
                # FORCE FULL WIDTH
                # We ignore the group's X and W. We only care about Y and Height.
                candidate_box = (self.scan_min_x, min_gy, self.scan_max_x - self.scan_min_x, max_gy - min_gy)

        # --- PHASE 4: Robust Persistence & Smoothing ---
        self._update_state(candidate_box)
        
        return self.is_active, self.smoothed_box, self._create_debug_view(frame), binary_mask

    def _update_state(self, candidate_box):
        """
        Manages the 'Active' state with hysteresis and smoothing.
        """
        # A. If we have a NEW detection
        if candidate_box is not None:
            self.loss_counter = 0
            self.detection_streak += 1
            
            # 1. Update Position
            new_y = candidate_box[1]
            new_h = candidate_box[3]
            
            if self.smoothed_box is None:
                # First hit: snap instantly
                self.smoothed_box = candidate_box
            else:
                # SMOOTHING: Blend with previous position to reduce jitter
                # Y usually only increases (car moves forward), so we can bias towards that if needed
                old_x, old_y, old_w, old_h = self.smoothed_box
                
                # Alpha blend (0.3 = fast adaptation, 0.1 = slow/smooth)
                alpha = 0.3
                smooth_y = int(old_y * (1 - alpha) + new_y * alpha)
                smooth_h = int(old_h * (1 - alpha) + new_h * alpha)
                
                self.smoothed_box = (old_x, smooth_y, old_w, smooth_h)

            # 2. Activate Alert (if streak is high enough)
            if self.detection_streak >= self.persistence_threshold:
                self.is_active = True

        # B. If we LOST detection
        else:
            self.detection_streak = 0
            
            if self.is_active:
                self.loss_counter += 1
                
                # MEMORY: Keep showing the box for N frames
                if self.loss_counter > self.max_loss_frames:
                    self.is_active = False
                    self.smoothed_box = None
                else:
                    # OPTIONAL: Move the box down slightly to simulate approach?
                    # For now, just holding it is stable enough.
                    pass

    def _create_debug_view(self, frame):
        view = frame.copy()
        
        # Draw ROI
        cv2.rectangle(view, (self.scan_min_x, self.scan_top_y), 
                      (self.scan_max_x, self.scan_bottom_y), (50, 50, 50), 2)
        
        # Draw Candidates
        for (rect, is_valid) in self.debug_candidates:
            x, y, w, h = rect
            color = (0, 255, 0) if is_valid else (0, 0, 255) 
            cv2.rectangle(view, (x, y), (x+w, y+h), color, 2)
            
        for (pt1, pt2) in self.debug_connections:
            cv2.line(view, pt1, pt2, (255, 255, 0), 2)

        return view

    def draw(self, frame):
        if self.is_active and self.smoothed_box:
            x, y, w, h = self.smoothed_box
            
            # 1. Semi-transparent Warning Zone (Full Width)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 165, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # 2. Solid Border
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 3)
            
            # 3. Label (Centered)
            text = "CROSSWALK DETECTED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Center X
            tx = x + (w - tw) // 2
            ty = y - 15
            
            # Text Background for readability
            cv2.rectangle(frame, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
            cv2.putText(frame, text, (tx, ty), font, font_scale, (0, 255, 255), thickness)
            
        return frame