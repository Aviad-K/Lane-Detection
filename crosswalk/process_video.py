"""
Lane Detection Video Processor
==============================
Main application that processes video frames for lane detection.
Cleaned: Strictly Lane Detection only.
"""

import cv2
import numpy as np
import os
import sys
from lane_preprocessor import LanePreprocessor
from lane_detector import LaneDetector
from taillight_detector import TaillightDetector
from crosswalk_detector import CrosswalkDetector

# =============================================================================
# 1. VISUAL CONFIGURATION (HARDCODED PIXELS)
# =============================================================================
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
HORIZON_Y = 780

# ROI Config (Simple Crop)
ROI_CONFIG = {
    'top_w': 1920,             
    'bottom_w': 1920,          
    'bottom_s': 0,           
    'top_s': 0,               
    'top_noise_ratio': 0,   
    'bottom_noise_ratio': 0, 
    'crop_height': HORIZON_Y / FRAME_HEIGHT,
    'roi_top_y_override': HORIZON_Y     
}

# INITIAL SEARCH ZONES (EXPLICIT PIXELS)
# Define the X-ranges for the trapezoids directly in pixels.
# top_x: (min_x, max_x) at the horizon line (HORIZON_Y)
# bottom_x: (min_x, max_x) at the bottom of the screen (FRAME_HEIGHT)
LANE_SEARCH_CONFIG = {
    'L': {
        'top_y':    HORIZON_Y,
        'top_x':    (775, 875),   # Hardcode these for your Tokyo video
        'bottom_x': (375, 575)    # Hardcode these for your Tokyo video
    },
    'R': {
        'top_y':    HORIZON_Y,
        'top_x':    (1025, 1125), # Hardcode these for your Tokyo video
        'bottom_x': (1400, 1600)  # Hardcode these for your Tokyo video
    }
}
# # Highway Video settings
# VIDEO_PATH = 'highway_short.mp4'
# START_FRAME = 0  # beginning plus car cutting in plus lane change
# # START_FRAME = 1100  # nearby car
# # START_FRAME = 5100 # lane change
# # START_FRAME = 5800 # car cutting in and nearby truck
# # START_FRAME = 7900 # curving
# # START_FRAME = 8880 # close car
# # START_FRAME = 11000  # close car in front of us 
# # START_FRAME = 14070  # car cutting in
# # START_FRAME = 17740  # bridge shadow
# # START_FRAME = 19530  # more shadows

# # Nightime Video settings
# VIDEO_PATH = 'night.mp4'
# START_FRAME = 0  # beginning

# # Nightime Video settings
# VIDEO_PATH = 'seoul_night.mp4'
# START_FRAME = 800  # beginning

# Crosswalk Video settings
VIDEO_PATH = 'crosswalk.mp4'
START_FRAME = 800  # beginning
# START_FRAME = 2000  # crosswalk appears

# =============================================================================
# INITIALIZATION
# =============================================================================

preprocessor = LanePreprocessor(
    ROI_CONFIG, 
    blur_kernel=5, 
    lightness_sensitivity=0.2,
    mask_median_kernel=5,
    min_area_close=600, 
    min_area_far=10,     
    area_threshold_exponent=4.0 
)

detector = LaneDetector(
    # 1. Base
    roi_config=ROI_CONFIG,
    initial_search_config=LANE_SEARCH_CONFIG,
    
    # 2. Line Finding
    hough_threshold=10,     
    min_line_length=20,     
    max_line_gap=350,       
    
    # 3. Filtering & Tracking
    horizontal_rejection_deg=80, 
    track_match_angle_deg=45,    
    track_merge_angle_deg=45,    
    sleeve_width=50,
    sleeve_expansion_factor=2.0,
    
    # 4. Smoothing
    smoothing_frames=1,      
    anchor_weight=1.0,
    coasting_anchor_weight=1000.0,
    
    # 5. Lane Change Logic
    lane_change_threshold=60,         
    lane_change_vertical_angle_deg=2,
    lane_change_center_tolerance_px=250,
    
    # 6. Stability
    swap_cooldown_frames=60,          
    anchor_suspension_frames=10,      
    
    # 7. Visuals
    calibration_frames=30,
    avg_lane_width=1000,
    vertical_warning_angle_deg=40, 
    safe_threshold_deg=42,          
    
    # 8. TUNING (Exposed)
    consistency_tolerance=50,
    drift_threshold_px=200,
    drift_tolerance_multiplier=4.0,
    
    max_loss_frames=30
)

TAILLIGHT_CONFIG = {
    # STATIC TRAPEZOID (The "Ego Tunnel")
    'top_x':    (810, 1110),    
    'bottom_x': (400, 1520),    
    'search_top': HORIZON_Y - 40,         
    
    # NEW: Adjustable Center Line
    # Adjust these to align the yellow line with the center of your lane
    'center_top_x': 960,     # Start at middle (1920 / 2)
    'center_bottom_x': 960,  # Start at middle
    'lane_sample_ratio': 0.62,
    
    'horizon_y': HORIZON_Y - 40,          
    
    # Standard Red
    'red_low_1':  [0, 150, 150],  
    'red_high_1': [10, 255, 255],
    'red_low_2':  [170, 125, 125],
    'red_high_2': [180, 255, 255],
    
    'min_light_area': 1,
    'max_light_area': 999999,
    'y_tolerance': 1000,         
    'focal_length': 550,
    # If the left and right blobs are further apart than this, they are ignored.
    'max_pair_width': 150,

    'morph_kernel_size': (7, 3),

    'car_width_at_bottom': 900, 
    'car_height_at_bottom': 700,
    # This leaves % of the box for the trunk/roof/window above the lights.
    'lights_height_ratio': 0.55,
    # Prevents the box from becoming too small at mid-distances.
    'min_box_width': 120,
}

# Initialize
car_tracker = TaillightDetector(TAILLIGHT_CONFIG, detector)

CW_HORIZON_Y = HORIZON_Y - 200

CROSSWALK_CONFIG = {
    'scan_min_y': CW_HORIZON_Y, 
    'scan_max_y': FRAME_HEIGHT,
    'scan_min_x': 0,             
    'scan_max_x': 1920,     

    # --- BLOB FILTERING (Is this a bar?) ---
    # Bars are solid paint. Adjust area based on resolution.
    'min_blob_area': 1000,           
    'max_blob_area': 15000,         
    
    # Solidity: Bars are rectangular. 
    # Use 0.7 to allow for some jagged edges from the preprocessor.
    'min_solidity': 0.9,            
    
    # Aspect Ratio (Width / Height)
    # Bars are usually wider than tall (ratio > 1.0)
    # BUT due to perspective, distant bars might look square (1.0) or slightly tall (0.5).
    # Lane lines are usually extremely tall (ratio < 0.2).
    'min_aspect_ratio': 1,        
    'max_aspect_ratio': 5,        
    
    # --- CLUSTERING (Are they neighbors?) ---
    # This is the "Gap Size" check.
    # If the gap between blobs is > 150px, they are not friends.
    'max_neighbor_dist_x': 250,     
    'max_neighbor_dist_y': 10,      # Allow some vertical drift (for angled approaches)
    'min_group_size': 3,            # 2 bars is enough to be suspicious
    
    # --- PERSISTENCE ---
    'persistence_threshold': 3,     # Needs 3 consecutive hits to show up
    'max_loss_frames': 15,          # KEY NEW PARAM: Keep showing for 0.5s (15 frames) after loss
}

CW_ROI_CONFIG = {
    'top_w': 1920,             
    'bottom_w': 1920,          
    'bottom_s': 0,           
    'top_s': 0,               
    'top_noise_ratio': 0,   
    'bottom_noise_ratio': 0, 
    'crop_height': CW_HORIZON_Y / FRAME_HEIGHT,
    'roi_top_y_override': CW_HORIZON_Y     
}

# 1. Dedicated Crosswalk Preprocessor
# We use a LARGE kernel (65) to ensure we see the full width of the bars
cw_preprocessor = LanePreprocessor(
    CW_ROI_CONFIG, 
    blur_kernel=5, 
    edge_threshold=30,
    local_average_kernel=151,      # KEY CHANGE: Big kernel for fat bars
    lightness_sensitivity=0.05,
    min_diff=5,                   # Higher diff to ignore faint road texture
    mask_median_kernel=5,
    min_area_close=200,           # KEY CHANGE: Bars are smaller than full lanes
    min_area_far=10,
    area_threshold_exponent=2.0 
)

crosswalk_detector = CrosswalkDetector(CROSSWALK_CONFIG)

# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def draw_search_zones(frame, search_config, roi_config):
    output = frame.copy()
    height, width = frame.shape[:2]
    crop_y = int(height * roi_config['crop_height'])
    
    cv2.line(output, (0, crop_y), (width, crop_y), (0, 165, 255), 2)
    colors = {'L': (255, 0, 0), 'R': (0, 0, 255)}
    
    for side, cfg in search_config.items():
        # DIRECT PIXEL ACCESS - NO RATIOS
        tx1, tx2 = cfg['top_x']
        bx1, bx2 = cfg['bottom_x']
        
        # Ensure integers
        tx1, tx2, bx1, bx2 = int(tx1), int(tx2), int(bx1), int(bx2)
        
        pts = np.array([[tx1, crop_y], [tx2, crop_y], [bx2, height], [bx1, height]], np.int32)
        sub = output.copy()
        cv2.fillPoly(sub, [pts], colors[side])
        cv2.addWeighted(sub, 0.2, output, 0.8, 0, output)
        cv2.polylines(output, [pts], True, colors[side], 2)
        cx = (bx1 + bx2) // 2
        cv2.putText(output, f"SEARCH {side}", (cx - 40, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[side], 2)

    return output

def draw_offset_bar(frame, offset, is_lane_change, direction, lane_center, car_center):
    if offset is None or lane_center is None or car_center is None: return frame
    h, w = frame.shape[:2]
    cx_ref = int(car_center) 
    bar_y = h - 40
    bar_width = 300
    
    cv2.line(frame, (cx_ref, h - 100), (cx_ref, h), (255, 255, 255), 2)
    cx_lane = int(lane_center)
    cx_lane = max(0, min(w, cx_lane)) 
    cv2.line(frame, (cx_lane, h - 100), (cx_lane, h), (255, 200, 0), 3)

    cv2.rectangle(frame, (cx_ref - bar_width//2, bar_y), (cx_ref + bar_width//2, bar_y + 10), (50, 50, 50), -1)
    cv2.line(frame, (cx_ref, bar_y - 10), (cx_ref, bar_y + 20), (255, 255, 255), 2)
    marker_x = int(lane_center)
    marker_x = max(cx_ref - bar_width//2, min(cx_ref + bar_width//2, marker_x))
    
    color = (0, 255, 0)
    if abs(offset) > 30: color = (0, 255, 255)
    if is_lane_change: color = (0, 0, 255)
        
    cv2.circle(frame, (marker_x, bar_y + 5), 8, color, -1)
    if is_lane_change:
        cv2.putText(frame, f"CHANGING: {direction}", (cx_ref - 100, bar_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

# =============================================================================
# FRAME PROCESSING (OPTIMIZED)
# =============================================================================

def process_frame(frame: np.ndarray, layer_mode: int = 0, minimal: bool = False) -> tuple:
    if frame is None: return (None,) * 13
    
    # 1. Lane Detection (MUST BE FIRST)
    binary_mask, raw_binary, roi_vertices, diff_vis_raw, diff_vis_tuned = preprocessor.process(frame)
    crop_y = int(frame.shape[0] * ROI_CONFIG['crop_height'])
    
    # Detect Lanes
    left_line, right_line, is_lane_change, lane_change_direction, offset, lane_center, car_center = detector.detect(binary_mask, crop_y)
    
    # 2. Crosswalk Detection
    # --- Run Preprocessor for Crosswalks ---
    # This handles the shadows/sunlight logic for you automatically
    cw_binary, _, _, _, _ = cw_preprocessor.process(frame, detect_edges=False)
    
    # Pass this CLEAN binary mask to the detector
    has_crosswalk, cw_box, cw_debug, cw_mask = crosswalk_detector.detect(frame, external_binary=cw_binary)

    # 2. Car Detection (Pass the lanes!)
    # The detector now handles the dynamic center calculation internally
    # detected_cars, search_poly, car_binary, car_roi_view = car_tracker.detect(frame, left_lane=left_line, right_lane=right_line)
    
    # 3. Final View
    view_final = frame.copy()
    view_final = detector.draw_lanes(view_final, left_line, right_line, is_lane_change=is_lane_change)
    # view_final = car_tracker.draw(view_final, detected_cars)
    view_final = crosswalk_detector.draw(view_final)
    
    if minimal:
        return view_final, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    # --- 4. Debug Rendering (Interactive Mode Only) ---
    view_offset = view_final.copy()
    view_offset = draw_offset_bar(view_offset, offset, is_lane_change, lane_change_direction, lane_center, car_center)
    view_search = draw_search_zones(frame, LANE_SEARCH_CONFIG, ROI_CONFIG)
    
    view_debug = detector.create_debug_view(binary_mask, left_line, right_line)
    
    mask_canvas = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    view_mask = detector.draw_lanes(mask_canvas, left_line, right_line)

    center_vertices = preprocessor.get_center_noise_vertices(frame.shape)
    
    # Return 15 elements (Added cw_mask at the end)
    # return (view_final, view_mask, view_debug, binary_mask, 
    #         roi_vertices, center_vertices, frame, 
    #         diff_vis_raw, diff_vis_tuned, view_offset, view_search,
    #         car_binary, car_roi_view, cw_debug, cw_mask)
    return (view_final, view_mask, view_debug, binary_mask, 
            roi_vertices, center_vertices, frame, 
            diff_vis_raw, diff_vis_tuned, view_offset, view_search,
            None, None, cw_debug, cw_mask)
# =============================================================================
# VIEW MODES
# =============================================================================

# Updated View List
VIEW_MODES = [
    'FINAL', 'MASK', 'DEBUG', 'OFFSET', 'SEARCH', 
    'DIFF-RAW', 'DIFF-TUNED', 
    'CAR-BINARY', 'CAR-ROI', 
    'CROSSWALK-DEBUG', 'CROSSWALK-MASK'
]

# ALLOWED_VIEWS = {
#     0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
# }
ALLOWED_VIEWS = {
    0: [0, 1, 2, 3, 4, 5, 6, 9, 10] 
}

def get_display_frame(view_mode: int, frame_data: tuple) -> tuple:
    # Unpack the 13 items
    (v_final, v_mask, v_debug, binary_mask, 
     _, _, _, 
     diff_vis_raw, diff_vis_tuned, 
     v_offset, v_search,
     car_binary, car_roi_view, cw_debug, cw_mask) = frame_data
    
    if view_mode == 0: return v_final, VIEW_MODES[0]
    elif view_mode == 1: return v_mask, VIEW_MODES[1]
    elif view_mode == 2: return v_debug, VIEW_MODES[2]
    elif view_mode == 3: return v_offset, VIEW_MODES[3]
    elif view_mode == 4: return v_search, VIEW_MODES[4]
    elif view_mode == 5: return diff_vis_raw, VIEW_MODES[5]
    elif view_mode == 6: return diff_vis_tuned, VIEW_MODES[6]
    elif view_mode == 7: return car_binary, VIEW_MODES[7] # PRE-PROCESSED RED
    elif view_mode == 8: return car_roi_view, VIEW_MODES[8] # RED TRAPEZOID
    if view_mode == 9: return cw_debug, VIEW_MODES[9]
    elif view_mode == 10: return cw_mask, VIEW_MODES[10] # The "Strict White" Mask

    return v_final, "UNKNOWN"

# =============================================================================
# BATCH PROCESSOR
# =============================================================================

def process_video_file(input_path, output_name=None):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    if output_name is None:
        base = os.path.basename(input_path)
        output_name = f"outputs/processed_{base}"
    else:
        output_name = f"outputs/{output_name}"

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    
    print(f"Processing: {input_path} -> {output_name}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print("-" * 40)

    detector.reset()
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # [OPTIMIZATION] Set minimal=True to skip debug/offset rendering
        frame_data = process_frame(frame, layer_mode=0, minimal=True)
        final_view = frame_data[0] # The first element is always view_final
        
        out.write(final_view)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            progress = int((frame_idx / total_frames) * 100)
            sys.stdout.write(f"\rProgress: [{progress}%] {frame_idx}/{total_frames} frames")
            sys.stdout.flush()
            
    cap.release()
    out.release()
    print(f"\nDone! Video saved to {output_name}")

# =============================================================================
# INTERACTIVE PLAYER
# =============================================================================

def interactive_player():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    current_frame = START_FRAME
    
    layer_mode = 0
    internal_view_index = 0
    paused = False
    manual_seek = False
    last_frame_data = None
    
    print("Controls: [Space] Pause, [E] Cycle View, [Q] Quit")

    cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lane Detection', 1280, 720)
    
    while cap.isOpened():
        if not paused or manual_seek:
            if manual_seek:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                manual_seek = False
            ret, frame = cap.read()
            if not ret: break
            
            # Interactive mode needs all debug data, so minimal=False
            last_frame_data = process_frame(frame, layer_mode, minimal=False)
            if not paused: current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if last_frame_data and last_frame_data[0] is not None:
            current_allowed_list = ALLOWED_VIEWS.get(layer_mode, [0])
            if internal_view_index >= len(current_allowed_list): internal_view_index = 0
            actual_view_mode = current_allowed_list[internal_view_index]
            
            display, mode_label = get_display_frame(actual_view_mode, last_frame_data)
            
            if len(display.shape) == 2: display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            
            text = f"{mode_label}"
            cv2.putText(display, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Lane Detection', display)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == ord('e'): internal_view_index = (internal_view_index + 1) % len(current_allowed_list)
        elif key == ord(' '): paused = not paused
        elif key == ord('a'): manual_seek = True; current_frame = max(0, current_frame - 30)
        elif key == ord('d'): manual_seek = True; current_frame = min(total_frames - 1, current_frame + 30)
        elif key == ord('r'): detector.reset()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    MODE = "SAVE" # Options: "INTERACTIVE", "SAVE"
    
    if MODE == "SAVE":
        process_video_file(VIDEO_PATH)
    else:
        interactive_player()