## Goal

Headless access to a Raspberry Pi 4 over the network, **guaranteed SSH login**, even if Raspberry Pi Imager customisations don’t apply cleanly.

---

## 1) Flash Raspberry Pi OS to the microSD

1. Use **Raspberry Pi Imager**.
2. Choose your OS (Lite or Full GUI both work).
3. Click **Write** and let it **verify**.

*(You can still set Wi-Fi/SSH in Imager, but the key to reliability is the next step.)*

---

## 2) On Ubuntu: locate the SD card mount paths

Insert the SD card and run:

```bash
lsblk -o NAME,FSTYPE,SIZE,LABEL,MOUNTPOINTS
```

Find the **boot partition** (vfat) mountpoint, e.g.:

* `/media/nathan/bootfs`  ✅ this is the one we used

---

## 3) Force-enable SSH + create the user (the “make it work no matter what” step)

On your Ubuntu machine (replace the path if yours differs):

```bash
# Enable SSH on next boot
sudo touch /media/nathan/bootfs/ssh

# Create a user "nathan" with password "0000"
HASH=$(echo '0000' | openssl passwd -6 -stdin)
echo "nathan:$HASH" | sudo tee /media/nathan/bootfs/userconf.txt

sync
```

What this does:

* `ssh` file → tells Raspberry Pi OS to enable SSH at boot
* `userconf.txt` → creates the user on first boot (so you aren’t blocked by GUI first-boot setup or missing user creation)

---

## 4) Eject SD card, boot the Pi on Ethernet

* Put SD card in the Pi
* Plug **Ethernet to your router/box**
* Power on
* Wait ~1–2 minutes

---

## 5) Find the Pi’s IP from Ubuntu

Scan your LAN (adjust subnet if yours isn’t `192.168.1.0/24`):

```bash
sudo nmap -sn 192.168.1.0/24
```

You saw something like:

* `raspberrypi.lan (192.168.1.149)` with MAC vendor “Raspberry Pi Trading”

---

## 6) Verify SSH is open, then connect

```bash
nc -vz 192.168.1.149 22
ssh nathan@192.168.1.149
```

* First connect prompts you to accept the host key (`yes`).
* Enter password `0000`.

---

## Notes / why this worked

* It avoids relying on `.local` name resolution (mDNS can be flaky).
* It avoids relying on Imager’s “first boot customisation” alone.
* It guarantees a user exists and SSH is enabled on first boot.

If you want, I can also give you the “clean-up” steps to (a) change the password, (b) set a proper hostname (no spaces), and (c) switch to SSH keys after you’re in.

## Afterward (dowload GitHub repository)

```bash
sudo apt update
sudo apt install -y git
```

```bash
git clone https://github.com/NathanAmoussou/3d_edge_object_detection.git
```

Only to create a virtual environment the first time (to download the dependencies):

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

## Connect from public WiFi

...
