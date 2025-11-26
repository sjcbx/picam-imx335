#!/usr/bin/env python3
"""
imx335_cam_v3.py

Camera app for Arducam IMX335 on Raspberry Pi.

Features:
- Live preview (640x480 by default)
- Space / S = capture still
- m = toggle AE (auto/manual)
- [ / ] = exposure (manual only)
- - / = = analogue gain (manual only)
- r = toggle RAW <-> JPEG capture mode
  - JPEG: captures a processed RGB still and saves as .jpg (via capture_file)
  - RAW: captures raw Bayer array and metadata, saves as compressed .npz for later processing
- q = quit

Dependencies:
- python3-picamera2
- python3-opencv
- python3-numpy

Install extra lib:
sudo apt update
sudo apt install -y python3-numpy
"""

import os
import time
import threading
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import math

# ---- Config ----
DISPLAY_SIZE = (640, 480)              # preview size (DSI target)
STILL_SIZE = (2592, 1944)              # full sensor still resolution
SAVE_DIR = "/home/user/Pictures"         # where captures are stored
FLASH_MS = 0.6                         # flash/feedback duration

EXP_STEP_US = 2000                     # 2 ms step for manual exposure
GAIN_STEP = 0.1                        # analogue gain step

MIN_EXPOSURE_US = 100                  # 0.1 ms
MAX_EXPOSURE_US = 85_000           # 5 s
MIN_GAIN = 1.0
MAX_GAIN = 9.6

os.makedirs(SAVE_DIR, exist_ok=True)


class SimpleCameraApp:
    def __init__(self):
        self.picam2 = Picamera2()

        # Preview config (processed RGB for on-screen preview)
        self.preview_config = self.picam2.create_preview_configuration(
            main={"size": DISPLAY_SIZE, "format": "RGB888"}
        )

        # Still configs:
        # - processed RGB still (fast to JPEG encode)
        self.still_config_jpeg = self.picam2.create_still_configuration(
            main={"size": STILL_SIZE, "format": "RGB888"}
        )
        
        # This guarantees we use the exact format string (e.g. 'SRGGB12_CSI2P') the driver expects.
        try:
            raw_mode = next(m for m in self.picam2.sensor_modes if m['size'] == STILL_SIZE)
            self.still_config_raw = self.picam2.create_still_configuration(raw=raw_mode)
        except StopIteration:
            print(f"Warning: Resolution {STILL_SIZE} not found in sensor modes. Using defaults.")
            # Fallback if specific resolution isn't found
            self.still_config_raw = self.picam2.create_still_configuration(raw={"size": STILL_SIZE, "format": "SRGGB12_CSI2P"})

        # Start preview config by default
        self.picam2.configure(self.preview_config)


        # Start preview config by default
        self.picam2.configure(self.preview_config)

        # State
        self.running = True
        self.show_help = False
        self.ae_enabled = True
        self.manual_exposure_us = None
        self.manual_gain = None
        self.flash_until = 0.0
        self.flash_text = ""
        self.lock = threading.Lock()
        self.raw_mode = False  # False = JPEG, True = RAW

        # Start preview
        self.picam2.start()
        time.sleep(0.15)
        self.picam2.set_controls({"AeEnable": True})

        # Optional GPIO placeholder
        self.gpio = None
        try:
            import gpiozero  # noqa: F401
            self.gpio = True
        except Exception:
            self.gpio = False

    def run(self):
        window_name = "IMX335 Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, *DISPLAY_SIZE)

        try:
            while self.running:
                frame = self.picam2.capture_array()
                metadata = self.picam2.capture_metadata()

                exposure_us = metadata.get("ExposureTime", 0)
                # For reasons unknown to anybody but god, the picamera2 API has 2 different gains:
                # AnalogueGain & DigitalGain.
                # The former is readable and controllable. The latter is only readable.
                analogue_gain = round(metadata.get("AnalogueGain", 1.0),2)
                digital_gain = round(metadata.get("DigitalGain", 1.0),2)
                iso_est = int(analogue_gain * 100)
                total_gain = round((analogue_gain * digital_gain),2)
                
                display_gain = (self.manual_gain if (not self.ae_enabled and self.manual_gain is not None) else total_gain)
                display_gain = float(display_gain)
                display_gain = max(1.0, min(9.6, display_gain))
                iso_est_raw = float(display_gain) * 100.0
                
                # --- map sensor gain -> ISO (power law) ---
                GAIN_MIN = 1.0
                GAIN_MAX = 9.6   # now use 9.6 as top end
                ISO_MIN = 100
                ISO_MAX = 6400
                
                expo = 1.0
                if GAIN_MAX > GAIN_MIN and ISO_MAX > ISO_MIN:
                    expo = math.log(ISO_MAX / ISO_MIN) / math.log(GAIN_MAX / GAIN_MIN)

                iso_float = ISO_MIN * ((display_gain / GAIN_MIN) ** expo)

                # --- rounding to camera-style ISO (nearest standard ISO) ---
                STANDARD_ISOS = [100,110,125,140,160,180,200,220,250,280,320,400,500,640,800,1000,1250,1600,2000,2200,2350,2500,2800,3000,3200,3600,4000,4500,5000,5600,6400]
                iso_est = min(STANDARD_ISOS, key=lambda s: abs(s - iso_float))
                # iso_est now is e.g. 100, 200, 400, 800, 1600, 3200, 6400 etc.


                shutter_str = self._format_shutter(exposure_us)

                # Draw overlays
                self._draw_overlay(frame, shutter_str, analogue_gain, total_gain, iso_est)

                # Flash overlay if active
                now = time.time()
                with self.lock:
                    if now < self.flash_until:
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
                        alpha = 0.25
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        if self.flash_text:
                            cv2.putText(frame, self.flash_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3,
                                        cv2.LINE_AA)

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key != 0xFF:
                    self._handle_keypress(key, exposure_us, analogue_gain)

        finally:
            self.picam2.stop()
            cv2.destroyAllWindows()

    # ---------- Overlay / UI helpers ----------
    def _format_shutter(self, exposure_us: int) -> str:
        if not exposure_us or exposure_us <= 0:
            return "â"
        exposure_s = exposure_us / 1_000_000.0
        if exposure_s < 1.0:
            denom = round(1.0 / exposure_s)
            if denom <= 0:
                return f"{exposure_s:.3f}s"
            return f"1/{denom}"
        else:
            if exposure_s < 10:
                return f"{exposure_s:.1f}s"
            else:
                return f"{int(exposure_s)}s"

    def _draw_overlay(self, frame, shutter_str: str, a_gain, d_gain, iso_est):
        h, w, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.7
        th = 2
        margin = 12
        
        #r_gain = round((self.manual_gain),2)

        mode_text = "RAW" if self.raw_mode else "JPEG"
        ae_text = "AE: ON" if self.ae_enabled else f"AE: OFF"
        info_text = f"{ae_text}  {shutter_str}   ISO {iso_est}   MODE: {mode_text}"
        cv2.putText(frame, info_text, (margin + 2, h - margin + 2), font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(frame, info_text, (margin, h - margin), font, fs, (255, 255, 255), th, cv2.LINE_AA)

        ae_text = f"AE: ON  E={shutter_str or 'â'} AG={a_gain or 'â'} TG={d_gain or 'â'}" if self.ae_enabled else f"AE: OFF  E={self.manual_exposure_us or 'â'} AG={self.manual_gain or 'â'} TG={d_gain or 'â'}"
        cv2.putText(frame, ae_text, (margin, 26), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if self.show_help:
            lines = [
                "SPACE / s = CAPTURE",
                "m = AE (auto/manual)",
                "[ / ] = EXPOSURE (manual only)",
                "- / + = GAIN / ISO (manual only)",
                "r = RAW <-> JPEG",
                "q = QUIT"
            ]
            hy = 52
            for line in lines:
                cv2.putText(frame, line, (margin, hy), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                hy += 18

    # ---------- Controls / Capture ----------
    def _handle_keypress(self, key, cur_exposure_us, cur_gain):
        # Quit
        if key == ord("q"):
            self.running = False
            return

        # Toggle help
        if key == ord("h"):
            self.show_help = not self.show_help
            return

        # Toggle RAW / JPEG mode
        if key == ord("r"):
            self.raw_mode = not self.raw_mode
            with self.lock:
                self.flash_text = f"MODE: {'RAW' if self.raw_mode else 'JPEG'}"
                self.flash_until = time.time() + 0.7
            return

        # Capture (shutter)
        if key == ord(" ") or key == ord("s"):
            threading.Thread(target=self._capture_still).start()
            return

        # Toggle AE/manual
        if key == ord("m") or key == ord("a"):
            self._toggle_ae(cur_exposure_us, cur_gain)
            return

        # Manual exposure adjustments (only when AE disabled)
        if not self.ae_enabled:
            if key == ord("["):
                self.manual_exposure_us = max(MIN_EXPOSURE_US, (self.manual_exposure_us or cur_exposure_us) - EXP_STEP_US)
                self._apply_manual_controls()
            elif key == ord("]"):
                self.manual_exposure_us = min(MAX_EXPOSURE_US, (self.manual_exposure_us or cur_exposure_us) + EXP_STEP_US)
                self._apply_manual_controls()
            elif key == ord("-"):
                self.manual_gain = max(MIN_GAIN, (self.manual_gain or cur_gain) - GAIN_STEP)
                self._apply_manual_controls()
            elif key == ord("=") or key == ord("+"):
                self.manual_gain = min(MAX_GAIN, (self.manual_gain or cur_gain) + GAIN_STEP)
                self._apply_manual_controls()

    def _toggle_ae(self, cur_exposure_us, cur_gain):
        self.ae_enabled = not self.ae_enabled
        if self.ae_enabled:
            self.picam2.set_controls({"AeEnable": True})
            self.manual_exposure_us = None
            self.manual_gain = None
        else:
            self.manual_exposure_us = int(cur_exposure_us or 1000)
            self.manual_gain = float(cur_gain or 1.0)
            self._apply_manual_controls()

    def _apply_manual_controls(self):
        ctrl = {"AeEnable": False}
        if self.manual_exposure_us:
            ctrl["ExposureTime"] = int(self.manual_exposure_us)
        if self.manual_gain:
            ctrl["AnalogueGain"] = float(self.manual_gain)
        self.picam2.set_controls(ctrl)

    def _capture_still(self):
        """
        Capture a still in either JPEG or RAW mode.
        JPEG: reconfigure to processed still config and use capture_file(filename.jpg)
        RAW: reconfigure to RAW still config, capture_array() to retrieve Bayer data,
             save compressed .npz with raw + metadata for later processing
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.raw_mode:
            filename = os.path.join(SAVE_DIR, f"imx335_{timestamp}.npz")
        else:
            filename = os.path.join(SAVE_DIR, f"imx335_{timestamp}.jpg")

        # flash UI
        with self.lock:
            self.flash_text = "CAPTURING..."
            self.flash_until = time.time() + FLASH_MS

        try:
            # Stop preview and switch to selected still config
            self.picam2.stop()
            if self.raw_mode:
                cfg = self.still_config_raw
            else:
                cfg = self.still_config_jpeg

            self.picam2.configure(cfg)
            self.picam2.start()
            time.sleep(0.20)  # settle

            # If AE disabled, apply manual controls to still capture
            if not self.ae_enabled:
                ctrl = {}
                if self.manual_exposure_us:
                    ctrl["ExposureTime"] = int(self.manual_exposure_us)
                if self.manual_gain:
                    ctrl["AnalogueGain"] = float(self.manual_gain)
                ctrl["AeEnable"] = False
                self.picam2.set_controls(ctrl)

            # Capture depending on mode
            if self.raw_mode:
                raw_arr = self.picam2.capture_array("raw")
                meta = self.picam2.capture_metadata()
                # Compute small diagnostics to help later debugging
                try:
                    raw_stats = {
                        "dtype": str(raw_arr.dtype),
                        "shape": raw_arr.shape,
                        "min": int(raw_arr.min()),
                        "max": int(raw_arr.max())
                    }
                except Exception:
                    raw_stats = {"info": "could not compute stats"}

                # Save raw + metadata + stats (compressed)
                np.savez_compressed(filename, raw=raw_arr, metadata=str(meta), raw_stats=str(raw_stats))


            else:
                # JPEG mode: let libcamera encode to JPEG
                self.picam2.capture_file(filename)

            # saved flash
            with self.lock:
                self.flash_text = f"SAVED: {os.path.basename(filename)}"
                self.flash_until = time.time() + 1.2

        except Exception as e:
            with self.lock:
                self.flash_text = f"ERROR: {e}"
                self.flash_until = time.time() + 1.5

        finally:
            # Restore preview config
            try:
                self.picam2.stop()
                self.picam2.configure(self.preview_config)
                self.picam2.start()
            except Exception:
                pass

            time.sleep(1.2)
            with self.lock:
                self.flash_text = ""

def main():
    app = SimpleCameraApp()
    app.run()

if __name__ == "__main__":
    main()



