#!/usr/bin/env python3
"""
Sentrya Enhanced Security System - Quick Start Script

This script provides an easy way to start the Sentrya system with proper setup.
"""

import os
import sys
import subprocess
import json

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import cv2
        import torch
        from ultralytics import YOLO
        import numpy as np
        print("✅ All core dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: python3 -m pip install -r requirements.txt")
        return False

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is working")
                return True
            else:
                print("⚠️ Camera opened but no frame received")
                return False
        else:
            print("❌ Could not open camera")
            return False
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

def check_zones():
    """Check if zones are configured"""
    if os.path.exists("zones_config.json"):
        try:
            with open("zones_config.json", 'r') as f:
                zones = json.load(f)
            if zones:
                print(f"✅ Zones configured: {list(zones.keys())}")
                return True
            else:
                print("⚠️ Zone file exists but is empty")
                return False
        except Exception as e:
            print(f"⚠️ Zone file exists but has errors: {e}")
            return False
    else:
        print("⚠️ No zone configuration found")
        return False

def run_zone_setup():
    """Run the zone setup utility"""
    print("\n🎯 Starting zone setup utility...")
    try:
        subprocess.run([sys.executable, "zone_setup.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Zone setup failed")
        return False
    except FileNotFoundError:
        print("❌ zone_setup.py not found")
        return False

def run_security_system():
    """Run the main security system"""
    print("\n🚀 Starting Sentrya Enhanced Security System...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Security system exited with error")
    except KeyboardInterrupt:
        print("\n👋 Security system stopped by user")
    except FileNotFoundError:
        print("❌ main.py not found")

def main():
    print("🏠 Sentrya Enhanced Security System - Quick Start")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking system requirements...")
    if not check_dependencies():
        print("\n❌ Please install dependencies first:")
        print("   python3 -m pip install -r requirements.txt")
        return
    
    # Check camera
    if not check_camera():
        print("\n❌ Camera issues detected. Please:")
        print("   1. Make sure your webcam is connected")
        print("   2. Check if another application is using the camera")
        print("   3. Try changing the camera index in main.py (line: VideoCapture(1))")
        
        # Ask if user wants to continue anyway
        choice = input("\nContinue anyway? (y/n): ").lower().strip()
        if choice != 'y':
            return
    
    # Check zones
    zones_configured = check_zones()
    
    if not zones_configured:
        print("\n🎯 Zone configuration required!")
        print("The system works best with configured zones for monitoring specific areas.")
        print("You can:")
        print("  1. Set up zones now using the interactive tool")
        print("  2. Run with default zones (not recommended)")
        print("  3. Exit and configure zones manually")
        
        choice = input("\nChoice (1/2/3): ").strip()
        
        if choice == "1":
            if not run_zone_setup():
                print("❌ Zone setup failed. Exiting.")
                return
        elif choice == "2":
            print("⚠️ Running with default zones...")
        elif choice == "3":
            print("👋 Edit zones_config.json manually, then run this script again.")
            return
        else:
            print("❌ Invalid choice. Exiting.")
            return
    
    # Final check
    print("\n🔧 Final system check...")
    print("✅ Dependencies: OK")
    print("✅ Camera: OK" if check_camera() else "⚠️ Camera: WARNING")
    print("✅ Zones: OK" if check_zones() else "⚠️ Zones: Using defaults")
    
    print("\n🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reload zone configuration")
    print("  - Audio deterrents will play automatically")
    print("  - Notifications logged to security_notifications.json")
    
    input("\nPress Enter to start the security system...")
    
    # Run the main system
    run_security_system()
    
    print("\n📊 Session complete!")
    print("Check security_notifications.json for logged alerts.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your installation and try again.") 