#!/usr/bin/env python3
"""
Sentrya Zone Configuration Utility

Interactive tool to set up monitoring zones by clicking on the camera feed.
Run this script to easily configure your zones before running the main security system.
"""

import cv2
import json
import os

class ZoneSetup:
    def __init__(self):
        self.zones = {}
        self.current_zone_name = ""
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_rect = None
        
        # Load existing zones if available
        self.load_existing_zones()
        
    def load_existing_zones(self):
        """Load existing zone configuration"""
        try:
            if os.path.exists("zones_config.json"):
                with open("zones_config.json", 'r') as f:
                    self.zones = json.load(f)
                print(f"✅ Loaded existing zones: {list(self.zones.keys())}")
            else:
                print("📝 No existing zones found. Starting fresh.")
        except Exception as e:
            print(f"⚠️ Error loading zones: {e}")
            self.zones = {}
    
    def save_zones(self):
        """Save zone configuration to JSON file"""
        try:
            with open("zones_config.json", 'w') as f:
                json.dump(self.zones, f, indent=2)
            print(f"💾 Saved {len(self.zones)} zones to zones_config.json")
            return True
        except Exception as e:
            print(f"❌ Error saving zones: {e}")
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zone drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_zone_name:
                self.drawing = True
                self.start_point = (x, y)
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_zone_name:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_zone_name:
                self.drawing = False
                self.end_point = (x, y)
                
                # Calculate zone coordinates (ensure x1 < x2 and y1 < y2)
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                
                # Only save if rectangle has meaningful size
                if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                    self.zones[self.current_zone_name] = [x1, y1, x2, y2]
                    print(f"✅ Zone '{self.current_zone_name}' created: [{x1}, {y1}, {x2}, {y2}]")
                    self.current_zone_name = ""
                else:
                    print("❌ Zone too small. Please draw a larger rectangle.")
                
                self.start_point = None
                self.end_point = None
    
    def draw_zones(self, frame):
        """Draw existing zones and current zone being drawn"""
        # Draw existing zones
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, zone_name, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw zone being created
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 0), 2)
            if self.current_zone_name:
                cv2.putText(frame, f"Drawing: {self.current_zone_name}", 
                           self.start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_instructions(self, frame):
        """Draw instructions on the frame"""
        instructions = [
            "Zone Setup Mode - Instructions:",
            "1. Enter zone name in terminal",
            "2. Click and drag to draw zone rectangle",
            "3. Press 's' to save zones",
            "4. Press 'd' to delete last zone",
            "5. Press 'q' to quit",
            "",
            f"Current zones: {len(self.zones)}"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 25
            color = (255, 255, 255) if instruction else (100, 100, 100)
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def run(self):
        """Main setup loop"""
        print("🎯 Sentrya Zone Setup Utility")
        print("=" * 50)
        print("This tool helps you configure monitoring zones for your security system.")
        print("Follow the on-screen instructions to draw zones on your camera feed.")
        print()
        
        # Initialize camera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        cv2.namedWindow("Zone Setup", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Zone Setup", self.mouse_callback)
        
        print("📹 Camera feed started. Follow instructions on screen.")
        print("💡 Tip: Position your camera and lighting as they will be during operation.")
        print()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Draw zones and instructions
            self.draw_zones(frame)
            self.draw_instructions(frame)
            
            # Show frame
            cv2.imshow("Zone Setup", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.save_zones():
                    print("💾 Zones saved successfully!")
                else:
                    print("❌ Failed to save zones")
            elif key == ord('d'):
                if self.zones:
                    last_zone = list(self.zones.keys())[-1]
                    del self.zones[last_zone]
                    print(f"🗑️ Deleted zone: {last_zone}")
                else:
                    print("❌ No zones to delete")
            
            # Check for new zone input
            if not self.current_zone_name and not self.drawing:
                try:
                    # Non-blocking input check
                    import select
                    import sys
                    
                    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                        zone_name = input().strip()
                        if zone_name:
                            if zone_name in self.zones:
                                overwrite = input(f"Zone '{zone_name}' exists. Overwrite? (y/n): ").lower()
                                if overwrite == 'y':
                                    self.current_zone_name = zone_name
                                    print(f"🎯 Ready to draw zone: {zone_name}")
                            else:
                                self.current_zone_name = zone_name
                                print(f"🎯 Ready to draw zone: {zone_name}")
                except:
                    # Fallback for systems without select
                    pass
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final save prompt
        if self.zones:
            print(f"\n📊 Final Configuration:")
            for zone_name, coords in self.zones.items():
                print(f"  - {zone_name}: {coords}")
            
            save_final = input("\n💾 Save final configuration? (y/n): ").lower()
            if save_final == 'y':
                self.save_zones()
                print("✅ Zone configuration complete!")
                print("🚀 Run 'python main.py' to start the enhanced security system.")
            else:
                print("❌ Configuration not saved")
        else:
            print("❌ No zones configured")

def main():
    """Main entry point with better input handling"""
    setup = ZoneSetup()
    
    print("🎯 Sentrya Zone Setup Utility")
    print("=" * 50)
    print("This tool helps you configure monitoring zones for your security system.")
    print()
    
    # Initialize camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    cv2.namedWindow("Zone Setup", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Zone Setup", setup.mouse_callback)
    
    print("📹 Camera feed started.")
    print("📝 Enter zone names below, then click and drag on the video to create zones.")
    print("💾 Press 's' in video window to save | 'd' to delete last zone | 'q' to quit")
    print()
    
    if setup.zones:
        print(f"📋 Existing zones: {list(setup.zones.keys())}")
        print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera")
            break
        
        # Draw zones and instructions
        setup.draw_zones(frame)
        setup.draw_instructions(frame)
        
        # Show frame
        cv2.imshow("Zone Setup", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            if setup.save_zones():
                print("💾 Zones saved successfully!")
            else:
                print("❌ Failed to save zones")
        elif key == ord('d'):
            if setup.zones:
                last_zone = list(setup.zones.keys())[-1]
                del setup.zones[last_zone]
                print(f"🗑️ Deleted zone: {last_zone}")
            else:
                print("❌ No zones to delete")
        
        # Prompt for new zone if not currently drawing
        if not setup.current_zone_name and not setup.drawing:
            print(f"Current zones: {list(setup.zones.keys())}")
            zone_name = input("Enter zone name (or 'done' to finish): ").strip()
            
            if zone_name.lower() == 'done':
                break
            elif zone_name:
                if zone_name in setup.zones:
                    overwrite = input(f"Zone '{zone_name}' exists. Overwrite? (y/n): ").lower()
                    if overwrite == 'y':
                        setup.current_zone_name = zone_name
                        print(f"🎯 Ready to draw zone: {zone_name}")
                        print("Click and drag on the video to create the zone rectangle.")
                else:
                    setup.current_zone_name = zone_name
                    print(f"🎯 Ready to draw zone: {zone_name}")
                    print("Click and drag on the video to create the zone rectangle.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final save prompt
    if setup.zones:
        print(f"\n📊 Final Configuration:")
        for zone_name, coords in setup.zones.items():
            print(f"  - {zone_name}: {coords}")
        
        save_final = input("\n💾 Save final configuration? (y/n): ").lower()
        if save_final == 'y':
            setup.save_zones()
            print("✅ Zone configuration complete!")
            print("🚀 Run 'python main.py' to start the enhanced security system.")
        else:
            print("❌ Configuration not saved")
    else:
        print("❌ No zones configured")

if __name__ == "__main__":
    main() 