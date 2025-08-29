To set up kiosk on pi:

sudo apt update
sudo apt install --no-install-recommends \
  xserver-xorg \
  x11-xserver-utils \
  xinit \
  openbox \
  chromium-browser

sudo usermod -aG tty,video,input,dialout natta

sudo apt install xserver-xorg-legacy
sudo nano /etc/X11/Xwrapper.config

set: 
allowed_users=anybody
needs_root_rights=yes
