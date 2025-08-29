import subprocess
import os

URL = "http://localhost:5000"

def launch_browser():
    os.environ.pop("DISPLAY", None)
    
    cmd = [
        "startx", 
        "/usr/bin/chromium-browser",
        "--noerrdialogs",
        "--disable-infobars",
        "--kiosk",
        URL
    ]
    subprocess.call(cmd)

def close_browser():
    # Kill Chromium
    subprocess.call(["pkill", "-f", "chromium-browser"])
    # Kill X server (optional, only if you want to return to console)
    subprocess.call(["pkill", "-f", "Xorg"])
