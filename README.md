# 🏠 Sentrya Enhanced Residential AI Security System

An advanced computer vision security system with **zone-based threat detection** and **real-time audio deterrence** capabilities.

## 🚀 Features

### Core Security Detection
- **Person Tracking**: Advanced appearance-based tracking with histogram matching
- **Loitering Detection**: Identifies people staying in frame for extended periods
- **Running Detection**: Detects rapid movement patterns
- **Erratic Movement**: Identifies suspicious movement patterns
- **Group Formation**: Detects when multiple people gather together

### 🎯 NEW: Zone-Based Monitoring
- **Configurable Zones**: Define specific areas to monitor (porch, driveway, backyard, etc.)
- **Zone-Specific Alerts**: Trigger alerts only when suspicious activity occurs in defined zones
- **Zone Transition Tracking**: Monitor when people enter/exit zones
- **Visual Zone Indicators**: See zone boundaries and activity status on screen

### 🔊 Audio Deterrent System
- **Real-time Voice Warnings**: Plays automated voice warnings when threats are detected
- **Platform Support**: 
  - macOS: Uses built-in `say` command
  - Linux: Uses `espeak` or `festival`
  - Windows: Uses `pyttsx3` text-to-speech
- **Cooldown Management**: Prevents audio spam with configurable cooldown periods
- **Zone-Specific Messages**: Customized warnings based on detected zone and behavior

### 📱 Notification System
- **Structured Notifications**: JSON-formatted alerts with timestamps and metadata
- **Confidence Levels**: High/Medium/Low threat classification
- **Notification Logging**: Persistent storage in `security_notifications.json`
- **Integration Ready**: Prepared hooks for Firebase, Twilio, or Telegram Bot integration

## 🛠️ Installation

1. **Clone/Download** the project files
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Platform-specific audio setup**:
   
   **macOS**: No additional setup needed (uses built-in `say`)
   
   **Linux**:
   ```bash
   sudo apt-get install espeak espeak-data libespeak-dev
   # OR
   sudo apt-get install festival festvox-kallpc16k
   ```
   
   **Windows**: `pyttsx3` will be installed via requirements.txt

## 🎮 Usage

### Basic Operation
```bash
python main.py
```

### Controls
- **`q`**: Quit the application
- **`r`**: Reload zone configuration from `zones_config.json`

### Zone Configuration

Edit `zones_config.json` to define your monitoring zones:

```json
{
  "porch": [100, 100, 400, 300],
  "driveway": [450, 150, 750, 400],
  "backyard": [50, 350, 350, 500],
  "front_yard": [200, 50, 600, 200],
  "side_gate": [20, 200, 150, 350]
}
```

**Zone Format**: `[x1, y1, x2, y2]` where:
- `(x1, y1)`: Top-left corner
- `(x2, y2)`: Bottom-right corner

**Setting Up Zones**:
1. Run the system and observe your camera feed
2. Note the pixel coordinates of areas you want to monitor
3. Update `zones_config.json` with your desired zones
4. Press `r` to reload zones without restarting

## 🔧 Configuration

### Audio Settings
```python
AUDIO_ALERTS_ENABLED = True          # Enable/disable audio deterrents
DETERRENT_COOLDOWN = 30              # Seconds between audio warnings per zone
```

### Detection Thresholds
```python
SPEED_THRESHOLD = 50                 # Running detection sensitivity
ERRATIC_THRESHOLD = 30               # Erratic movement angle threshold
GROUP_SIZE_THRESHOLD = 3             # Minimum people for group detection
loiter_threshold = 10                # Seconds before loitering alert
```

### Notification Settings
```python
NOTIFICATION_COOLDOWN = 60           # Seconds between notifications per zone/person
```

## 📊 Monitoring Dashboard

The enhanced system displays:
- **Zone Boundaries**: Yellow rectangles showing monitored areas
- **Person Tracking**: Color-coded boxes based on behavior:
  - 🟢 Green: Normal behavior
  - 🔴 Red: Loitering
  - 🟠 Orange: Running
  - 🟡 Yellow: Erratic movement
  - 🟣 Purple: Group formation
- **Zone Information**: Shows current zone and time spent in zone
- **Active Zones**: Lists zones with current activity
- **Deterrent Status**: Shows cooldown timers for each zone

## 📱 Notification Integration

The system logs all notifications to `security_notifications.json`. Example notification:

```json
{
  "timestamp": "2024-01-15 14:30:45",
  "person_id": 123,
  "zone_name": "porch",
  "behavior_type": "Loitering",
  "confidence_level": "high",
  "alert_id": "porch_123_1705327845"
}
```

### Future Integration Options
- **Firebase Cloud Messaging**: Push notifications to mobile devices
- **Twilio SMS**: Send text message alerts
- **Telegram Bot**: Real-time messaging integration
- **Email Alerts**: SMTP-based notifications
- **Webhook Integration**: Custom API endpoints

## 🎯 Zone-Based Detection Logic

The system triggers deterrents and notifications when:

1. **Loitering** in any defined zone (after 10+ seconds)
2. **Running** behavior detected within a zone
3. **Erratic movement** patterns in a zone
4. **Group formation** (3+ people) within a zone

Each alert respects cooldown periods to prevent spam while maintaining security effectiveness.

## 📈 Performance Features

- **Real-time FPS display**
- **Memory-efficient tracking** with automatic cleanup
- **Appearance-based re-identification** for consistent person tracking
- **Optimized zone calculations** for minimal performance impact

## 🔍 Technical Architecture

### Core Components
- **YOLO v8**: Object detection and tracking
- **OpenCV**: Computer vision processing
- **Appearance Histograms**: Person re-identification
- **Zone Engine**: Spatial monitoring logic
- **Audio Engine**: Cross-platform voice synthesis
- **Notification Engine**: Alert management and logging

### Detection Pipeline
1. **Motion Detection**: Initial movement filtering
2. **Person Detection**: YOLO-based human identification
3. **Appearance Tracking**: Histogram-based re-identification
4. **Zone Analysis**: Spatial behavior assessment
5. **Threat Classification**: Behavior pattern analysis
6. **Response Activation**: Audio deterrent + notification triggers

## 🚨 Security Best Practices

- **Privacy**: All processing happens locally (no cloud dependencies)
- **Performance**: Optimized for real-time operation on standard hardware
- **Reliability**: Robust error handling and graceful fallbacks
- **Configurability**: Easy to adjust for different environments
- **Expandability**: Clean architecture for adding new features

## 🔮 Future Enhancements

- [ ] Mobile app integration
- [ ] Cloud backup and analytics
- [ ] Advanced behavior patterns (pacing, lurking, etc.)
- [ ] Integration with smart home systems
- [ ] Face recognition capabilities
- [ ] Night vision optimization
- [ ] Multi-camera support

---

**Sentrya Enhanced** - Professional-grade residential security with intelligent zone monitoring and real-time deterrence. 