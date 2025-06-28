import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import deque
import math
import json
import os
import subprocess
import platform
from datetime import datetime

# Load YOLO model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

# For better person tracking, we'll track appearance histograms
appearance_memory = {}  # ID → appearance histogram
memory = {}  # ID → {entry_time, last_seen, box, loitering, positions, speeds, is_running, erratic_movement, in_group}
memory_timeout = 8  # Increased from 5 - longer memory for better tracking
loiter_threshold = 10
reliable_ids = set()  # Track reliable IDs instead of all unique IDs
appearance_match_threshold = 0.5  # Reduced from 0.7 - more lenient matching for webcam quality
id_frame_count = {}  # Track how many frames each ID has been seen

# Constants for suspicious behavior detection
SPEED_THRESHOLD = 80  # Increased from 50 - Speed above which is considered running (pixels per frame)
ERRATIC_THRESHOLD = 60  # Increased from 30 - Angle change threshold for erratic movement (degrees)
GROUP_DISTANCE_THRESHOLD = 150  # Distance threshold for group detection (pixels)
GROUP_SIZE_THRESHOLD = 3  # Number of people to consider a group
POSITION_HISTORY_SIZE = 30  # Number of positions to keep in history
SUSPICIOUS_MEMORY_TIMEOUT = 5  # Increased from 3 - Number of seconds to remember suspicious behavior

# NEW: Improved tracking parameters
MIN_DETECTION_CONFIDENCE = 0.3  # Minimum confidence for person detection
MIN_MOVEMENT_THRESHOLD = 15  # Minimum movement to consider for speed calculation
STABLE_TRACKING_FRAMES = 5  # Number of frames before considering tracking stable

suspicious_activities = []  # Log of suspicious activities with timestamps

# NEW: Zone-based monitoring configuration
ZONES_CONFIG_FILE = "zones_config.json"
AUDIO_ALERTS_ENABLED = True
DETERRENT_COOLDOWN = 30  # Seconds between audio deterrents for same zone
NOTIFICATION_COOLDOWN = 60  # Seconds between notifications for same zone/person

# Zone and deterrent tracking
zone_last_deterrent = {}  # zone_name -> timestamp of last deterrent
zone_last_notification = {}  # zone_name -> timestamp of last notification
person_zone_alerts = {}  # person_id -> {zone_name: last_alert_time}

def load_zones_config():
    """Load zone configuration from JSON file"""
    # Better default zones for typical webcam resolutions (640x480 or 1280x720)
    default_zones = {
        "center_area": [200, 150, 520, 350],  # Main central monitoring area
        "left_side": [50, 100, 250, 400],     # Left side of frame
        "right_side": [450, 100, 620, 400],   # Right side of frame
        "entrance": [250, 50, 450, 200]       # Upper center (like doorway)
    }
    
    try:
        if os.path.exists(ZONES_CONFIG_FILE):
            with open(ZONES_CONFIG_FILE, 'r') as f:
                zones = json.load(f)
                print(f"✅ Loaded zones configuration: {list(zones.keys())}")
                return zones
        else:
            # Create default zones file
            with open(ZONES_CONFIG_FILE, 'w') as f:
                json.dump(default_zones, f, indent=2)
            print(f"📝 Created default zones configuration file: {ZONES_CONFIG_FILE}")
            return default_zones
    except Exception as e:
        print(f"⚠️ Error loading zones config: {e}. Using default zones.")
        return default_zones

def is_in_zone(center, zones):
    """Check if a center point is within any defined zone"""
    for name, (x1, y1, x2, y2) in zones.items():
        if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
            return name
    return None

def play_deterrent_audio(zone_name, behavior_type):
    """Play audio deterrent warning"""
    if not AUDIO_ALERTS_ENABLED:
        return False
    
    current_time = time.time()
    
    # Check cooldown
    if zone_name in zone_last_deterrent:
        if current_time - zone_last_deterrent[zone_name] < DETERRENT_COOLDOWN:
            return False
    
    zone_last_deterrent[zone_name] = current_time
    
    # Create warning message
    warning_text = f"Warning! {behavior_type} detected in {zone_name} area. Please move along."
    
    try:
        # Try different audio playback methods based on platform
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            subprocess.run([
                "say", 
                "-v", "Alex",
                "-r", "180",  # Rate (words per minute)
                warning_text
            ], check=False, capture_output=True)
            
        elif system == "linux":
            # Try espeak first, then festival
            try:
                subprocess.run([
                    "espeak", 
                    "-s", "150",  # Speed
                    "-v", "en",
                    warning_text
                ], check=False, capture_output=True)
            except FileNotFoundError:
                try:
                    subprocess.run([
                        "festival", "--tts"
                    ], input=warning_text, text=True, check=False, capture_output=True)
                except FileNotFoundError:
                    print(f"🔊 AUDIO DETERRENT: {warning_text}")
                    return False
                    
        elif system == "windows":
            # Use Windows SAPI
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(warning_text)
                engine.runAndWait()
            except ImportError:
                print(f"🔊 AUDIO DETERRENT: {warning_text}")
                return False
        else:
            print(f"🔊 AUDIO DETERRENT: {warning_text}")
            return False
            
        print(f"🔊 Played audio deterrent for {zone_name}: {behavior_type}")
        return True
        
    except Exception as e:
        print(f"⚠️ Audio deterrent failed: {e}")
        print(f"🔊 AUDIO DETERRENT: {warning_text}")
        return False

def send_notification(person_id, zone_name, behavior_type, confidence_level="medium"):
    """Send notification about suspicious activity in zone"""
    current_time = time.time()
    
    # Check cooldown for this zone
    notification_key = f"{zone_name}_{person_id}"
    if notification_key in zone_last_notification:
        if current_time - zone_last_notification[notification_key] < NOTIFICATION_COOLDOWN:
            return False
    
    zone_last_notification[notification_key] = current_time
    
    # Create notification payload
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notification_data = {
        "timestamp": timestamp,
        "person_id": person_id,
        "zone_name": zone_name,
        "behavior_type": behavior_type,
        "confidence_level": confidence_level,
        "alert_id": f"{zone_name}_{person_id}_{int(current_time)}"
    }
    
    # Log notification (placeholder for future integration)
    print(f"📱 NOTIFICATION: {behavior_type} detected - Person {person_id} in {zone_name} area at {timestamp}")
    
    # TODO: Integrate with Firebase Cloud Messaging, Twilio, or Telegram Bot
    # Example integration points:
    # - send_firebase_notification(notification_data)
    # - send_twilio_sms(notification_data)
    # - send_telegram_message(notification_data)
    
    # For now, log to file for future processing
    try:
        notifications_log = "security_notifications.json"
        if os.path.exists(notifications_log):
            with open(notifications_log, 'r') as f:
                notifications = json.load(f)
        else:
            notifications = []
        
        notifications.append(notification_data)
        
        # Keep only last 100 notifications
        notifications = notifications[-100:]
        
        with open(notifications_log, 'w') as f:
            json.dump(notifications, f, indent=2)
            
    except Exception as e:
        print(f"⚠️ Failed to log notification: {e}")
    
    return True

def draw_zones(frame, zones):
    """Draw zone boundaries on the frame"""
    for zone_name, (x1, y1, x2, y2) in zones.items():
        # Draw zone rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Add zone label
        cv2.putText(frame, f"ZONE: {zone_name.upper()}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add deterrent status indicator
        current_time = time.time()
        if zone_name in zone_last_deterrent:
            time_since_deterrent = current_time - zone_last_deterrent[zone_name]
            if time_since_deterrent < DETERRENT_COOLDOWN:
                cooldown_remaining = int(DETERRENT_COOLDOWN - time_since_deterrent)
                cv2.putText(frame, f"Cooldown: {cooldown_remaining}s", 
                           (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Load zones configuration
zones = load_zones_config()
print(f"🎯 Monitoring zones: {list(zones.keys())}")

def extract_histogram(frame, bbox):
    """Extract color histogram from a region of interest"""
    x1, y1, x2, y2 = bbox
    # Ensure coordinates are within frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = frame[y1:y2, x1:x2]
    
    # Convert to HSV colorspace for better color differentiation
    try:
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except:
        return None

def compare_histograms(hist1, hist2):
    """Compare two histograms using correlation method"""
    if hist1 is None or hist2 is None:
        return 0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def find_matching_id(current_hist, x1, y1, x2, y2):
    """Find if this person matches an existing ID from appearance histograms"""
    if current_hist is None:
        return None
    
    best_match_id = None
    best_match_score = -1
    
    # First check recently seen IDs (more likely to match)
    recent_ids = [id for id in appearance_memory.keys() 
                  if id in memory and time.time() - memory[id]["last_seen"] < memory_timeout]
    
    for existing_id in recent_ids:
        # Skip if positions are too far apart (can't be the same person)
        if existing_id in memory:
            ex1, ey1, ex2, ey2 = memory[existing_id]["box"]
            ex_center = ((ex1 + ex2) // 2, (ey1 + ey2) // 2)
            curr_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Skip if distance is too large
            distance = math.sqrt((ex_center[0] - curr_center[0])**2 + (ex_center[1] - curr_center[1])**2)
            if distance > 300:  # If too far apart, it's likely not the same person
                continue
        
        # Compare histograms
        existing_hist = appearance_memory[existing_id]
        match_score = compare_histograms(current_hist, existing_hist)
        
        if match_score > best_match_score and match_score > appearance_match_threshold:
            best_match_score = match_score
            best_match_id = existing_id
    
    return best_match_id

def update_id_mapping(yolo_id, frame, bbox):
    """Update the ID mapping based on appearance"""
    hist = extract_histogram(frame, bbox)
    if hist is None:
        return yolo_id
    
    # Check if this person matches an existing ID
    matching_id = find_matching_id(hist, *bbox)
    
    if matching_id is not None:
        # Update histogram with a weighted average for better adaptation
        existing_hist = appearance_memory[matching_id]
        updated_hist = 0.7 * existing_hist + 0.3 * hist
        appearance_memory[matching_id] = updated_hist
        return matching_id
    else:
        # New person, store their histogram
        appearance_memory[yolo_id] = hist
        reliable_ids.add(yolo_id)
        return yolo_id

# Reset frame counter for FPS calculation
frame_count = 0
start_time = time.time()
fps = 0

print("🚀 Sentrya Enhanced Security System Started")
print("🎯 Zone-based monitoring enabled")
print("🔊 Audio deterrent system active")
print("📱 Notification system ready")
print("Press 'q' to quit, 'r' to reload zones config")

while True:
    current_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frame_diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_motion_area = sum(cv2.contourArea(c) for c in contours)
    human_detected = False
    current_ids = set()

    # Calculate FPS
    frame_count += 1
    if current_time - start_time >= 1:
        fps = frame_count / (current_time - start_time)
        frame_count = 0
        start_time = current_time

    # Draw zones on frame
    draw_zones(frame, zones)

    if total_motion_area > 5000:
        results = model.track(frame, persist=True, verbose=False)[0]

        # First pass - update positions and calculate centers
        for result in results.boxes:
            cls_id = int(result.cls[0])
            confidence = float(result.conf[0])  # Get detection confidence
            
            # Only process high-confidence person detections
            if model.names[cls_id] == 'person' and confidence > MIN_DETECTION_CONFIDENCE:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                yolo_obj_id = int(result.id[0]) if result.id is not None else -1
                if yolo_obj_id == -1:
                    continue
                
                # Use appearance-based tracking to get more stable ID
                obj_id = update_id_mapping(yolo_obj_id, frame, (x1, y1, x2, y2))
                
                current_ids.add(obj_id)
                human_detected = True
                
                # Track frame count for this ID
                if obj_id not in id_frame_count:
                    id_frame_count[obj_id] = 0
                id_frame_count[obj_id] += 1
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # NEW: Check if person is in any zone
                current_zone = is_in_zone(center, zones)
                
                # Update or create memory entry
                if obj_id not in memory:
                    memory[obj_id] = {
                        "entry_time": current_time,
                        "last_seen": current_time,
                        "box": (x1, y1, x2, y2),
                        "center": center,
                        "positions": deque(maxlen=POSITION_HISTORY_SIZE),
                        "speeds": deque(maxlen=POSITION_HISTORY_SIZE),
                        "loitering": False,
                        "is_running": False,
                        "erratic_movement": False,
                        "in_group": False,
                        "sus_last_detected": 0,
                        "current_zone": current_zone,  # NEW: Track current zone
                        "zone_entry_time": current_time if current_zone else None  # NEW: Zone entry time
                    }
                    memory[obj_id]["positions"].append(center)
                    
                    # NEW: Initial zone entry notification
                    if current_zone:
                        print(f"👤 Person {obj_id} entered zone: {current_zone}")
                        
                else:
                    memory[obj_id]["last_seen"] = current_time
                    memory[obj_id]["box"] = (x1, y1, x2, y2)
                    memory[obj_id]["center"] = center
                    
                    # NEW: Zone transition handling
                    prev_zone = memory[obj_id].get("current_zone")
                    if current_zone != prev_zone:
                        memory[obj_id]["current_zone"] = current_zone
                        memory[obj_id]["zone_entry_time"] = current_time if current_zone else None
                        
                        if current_zone:
                            print(f"👤 Person {obj_id} entered zone: {current_zone}")
                        elif prev_zone:
                            print(f"👤 Person {obj_id} left zone: {prev_zone}")
                    
                    # Calculate speed and movement pattern
                    if len(memory[obj_id]["positions"]) > 0:
                        prev_pos = memory[obj_id]["positions"][-1]
                        distance = math.sqrt((center[0] - prev_pos[0])**2 + (center[1] - prev_pos[1])**2)
                        
                        # Only consider significant movements to reduce noise
                        if distance > MIN_MOVEMENT_THRESHOLD:
                            memory[obj_id]["speeds"].append(distance)
                        
                        # Running detection with zone-based alerts (only for stable tracking)
                        recent_speeds = list(memory[obj_id]["speeds"])
                        if len(recent_speeds) >= 5 and id_frame_count[obj_id] > STABLE_TRACKING_FRAMES:  # Require more samples
                            avg_speed = sum(recent_speeds[-5:]) / 5  # Use last 5 speeds
                            memory[obj_id]["is_running"] = avg_speed > SPEED_THRESHOLD
                            
                            if memory[obj_id]["is_running"] and current_time - memory[obj_id]["sus_last_detected"] > SUSPICIOUS_MEMORY_TIMEOUT:
                                behavior = "Running"
                                suspicious_activities.append((current_time, f"Person {obj_id} is running"))
                                memory[obj_id]["sus_last_detected"] = current_time
                                
                                # NEW: Zone-based deterrent and notification
                                if current_zone:
                                    play_deterrent_audio(current_zone, behavior)
                                    send_notification(obj_id, current_zone, behavior, "high")
                        
                        # Erratic movement detection with zone-based alerts (more stable)
                        if len(memory[obj_id]["positions"]) >= 5 and id_frame_count[obj_id] > STABLE_TRACKING_FRAMES:
                            pos_array = list(memory[obj_id]["positions"])
                            if len(pos_array) >= 5:
                                # Use longer movement vectors for more stable detection
                                v1 = (pos_array[-3][0] - pos_array[-5][0], pos_array[-3][1] - pos_array[-5][1])
                                v2 = (pos_array[-1][0] - pos_array[-3][0], pos_array[-1][1] - pos_array[-3][1])
                                
                                # Calculate angle between vectors only for significant movements
                                v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
                                v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
                                
                                if v1_mag > MIN_MOVEMENT_THRESHOLD and v2_mag > MIN_MOVEMENT_THRESHOLD:
                                    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                                    magnitudes = v1_mag * v2_mag
                                    cos_angle = min(1, max(-1, dot_product / magnitudes))
                                    angle = math.degrees(math.acos(cos_angle))
                                    
                                    memory[obj_id]["erratic_movement"] = angle > ERRATIC_THRESHOLD
                                    
                                    if memory[obj_id]["erratic_movement"] and current_time - memory[obj_id]["sus_last_detected"] > SUSPICIOUS_MEMORY_TIMEOUT:
                                        behavior = "Erratic Movement"
                                        suspicious_activities.append((current_time, f"Person {obj_id} has erratic movement"))
                                        memory[obj_id]["sus_last_detected"] = current_time
                                        
                                        # NEW: Zone-based deterrent and notification
                                        if current_zone:
                                            play_deterrent_audio(current_zone, behavior)
                                            send_notification(obj_id, current_zone, behavior, "medium")
                    
                    memory[obj_id]["positions"].append(center)

        # Second pass - detect groups with zone-based alerts
        active_people = [id for id in current_ids if id in memory]
        if len(active_people) >= GROUP_SIZE_THRESHOLD:
            # Check for groups
            for i, id1 in enumerate(active_people):
                group_members = 0
                for j, id2 in enumerate(active_people):
                    if i != j:
                        center1 = memory[id1]["center"]
                        center2 = memory[id2]["center"]
                        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                        if distance < GROUP_DISTANCE_THRESHOLD:
                            group_members += 1
                
                prev_group_status = memory[id1].get("in_group", False)
                memory[id1]["in_group"] = group_members >= GROUP_SIZE_THRESHOLD - 1
                
                # Log group formation as suspicious only on change with zone-based alerts
                if memory[id1]["in_group"] and not prev_group_status and current_time - memory[id1]["sus_last_detected"] > SUSPICIOUS_MEMORY_TIMEOUT:
                    behavior = "Group Formation"
                    suspicious_activities.append((current_time, f"Group formed including person {id1}"))
                    memory[id1]["sus_last_detected"] = current_time
                    
                    # NEW: Zone-based deterrent and notification for groups
                    current_zone = memory[id1].get("current_zone")
                    if current_zone:
                        play_deterrent_audio(current_zone, behavior)
                        send_notification(id1, current_zone, behavior, "high")

    # Update loitering status and draw boxes with zone-based alerts
    for obj_id in list(memory.keys()):
        time_since_seen = current_time - memory[obj_id]["last_seen"]
        time_in_frame = current_time - memory[obj_id]["entry_time"]

        if time_since_seen >= memory_timeout:
            # Clean up old ID data
            if obj_id in id_frame_count:
                del id_frame_count[obj_id]
            if obj_id in appearance_memory:
                del appearance_memory[obj_id]
            reliable_ids.discard(obj_id)
            del memory[obj_id]
            continue

        # NEW: Zone-based loitering detection
        current_zone = memory[obj_id].get("current_zone")
        zone_time = 0
        if current_zone and memory[obj_id].get("zone_entry_time"):
            zone_time = current_time - memory[obj_id]["zone_entry_time"]

        if time_in_frame >= loiter_threshold:
            memory[obj_id]["loitering"] = True
            if time_in_frame - loiter_threshold < 1:  # Just started loitering
                behavior = "Loitering"
                suspicious_activities.append((current_time, f"Person {obj_id} is loitering"))
                
                # NEW: Zone-based loitering alerts
                if current_zone:
                    play_deterrent_audio(current_zone, behavior)
                    send_notification(obj_id, current_zone, behavior, "high")

        x1, y1, x2, y2 = memory[obj_id]["box"]
        
        # Determine box color based on behavior
        is_loitering = memory[obj_id]["loitering"]
        is_running = memory[obj_id].get("is_running", False)
        is_erratic = memory[obj_id].get("erratic_movement", False)
        in_group = memory[obj_id].get("in_group", False)
        
        # Suspicious behaviors get different colors
        if is_loitering:
            color = (0, 0, 255)  # Red for loitering
        elif is_running:
            color = (0, 165, 255)  # Orange for running
        elif is_erratic:
            color = (0, 255, 255)  # Yellow for erratic movement
        elif in_group:
            color = (255, 0, 255)  # Purple for group
        else:
            color = (0, 255, 0)  # Green for normal
        
        # Create the label with zone information
        label = f"ID {obj_id}"
        suspicious_behaviors = []
        
        # Only show behaviors for stably tracked persons
        is_stable = id_frame_count.get(obj_id, 0) > STABLE_TRACKING_FRAMES
        
        if is_loitering and is_stable:
            suspicious_behaviors.append(f"Loitering {int(time_in_frame)}s")
        if is_running and is_stable:
            suspicious_behaviors.append("Running")
        if is_erratic and is_stable:
            suspicious_behaviors.append("Erratic")
        if in_group and is_stable:
            suspicious_behaviors.append("Group")
        
        # Show tracking status for new detections
        if not is_stable:
            suspicious_behaviors.append(f"Tracking ({id_frame_count.get(obj_id, 0)})")
        
        # NEW: Add zone information to label
        if current_zone:
            suspicious_behaviors.append(f"Zone:{current_zone}")
            if zone_time > 0 and is_stable:
                suspicious_behaviors.append(f"{int(zone_time)}s")
            
        if suspicious_behaviors:
            label += f" ({', '.join(suspicious_behaviors)})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display recent suspicious activities
    recent_sus = [act for act in suspicious_activities if current_time - act[0] < 5]
    for i, (_, activity) in enumerate(recent_sus[-3:]):  # Show last 3 suspicious activities
        cv2.putText(frame, f"Alert: {activity}", (10, 90 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # UI overlays
    stable_people_count = len([id for id in reliable_ids if id_frame_count.get(id, 0) > STABLE_TRACKING_FRAMES])
    cv2.putText(frame, f"People Detected: {stable_people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # NEW: Zone monitoring status
    active_zones = set(memory[obj_id].get("current_zone") for obj_id in memory.keys() 
                      if memory[obj_id].get("current_zone"))
    if active_zones:
        zone_text = f"Active Zones: {', '.join(active_zones)}"
        cv2.putText(frame, zone_text, (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # NEW: Debug info
    debug_text = f"Tracking: {len(memory)} | Motion: {int(total_motion_area)}"
    cv2.putText(frame, debug_text, (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    if not human_detected:
        cv2.putText(frame, 'No Human Detected', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Sentrya Enhanced Security Monitor - Zone-Based Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # NEW: Reload zones configuration
        print("🔄 Reloading zones configuration...")
        zones = load_zones_config()
        print(f"✅ Zones reloaded: {list(zones.keys())}")

    prev_gray = gray.copy()

# Print summary of suspicious activities
print(f"\n📊 SESSION SUMMARY")
print(f"🚨 Detected {len(suspicious_activities)} suspicious activities:")
for timestamp, activity in suspicious_activities:
    print(f"  - {time.strftime('%H:%M:%S', time.localtime(timestamp))}: {activity}")

print(f"\n🎯 Zone Statistics:")
for zone_name in zones.keys():
    deterrent_count = 1 if zone_name in zone_last_deterrent else 0
    notification_count = len([k for k in zone_last_notification.keys() if k.startswith(zone_name)])
    print(f"  - {zone_name}: {deterrent_count} deterrents, {notification_count} notifications")

cap.release()
cv2.destroyAllWindows()
print("\n👋 Sentrya Enhanced Security System Stopped")
