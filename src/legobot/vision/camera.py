"""Raspberry Pi camera access (Picamera2), imported lazily so the rest of the
package works on machines without the camera stack."""

from __future__ import annotations

from legobot.config import CameraConfig


class PiCamera:
    def __init__(self, cfg: CameraConfig):
        from picamera2 import Picamera2

        self._cam = Picamera2()
        self._cam.configure(
            self._cam.create_preview_configuration(main={"size": (cfg.width, cfg.height)})
        )
        self._cam.start()

    def capture_bgr(self):
        import cv2

        return cv2.cvtColor(self._cam.capture_array(), cv2.COLOR_RGB2BGR)

    def close(self):
        self._cam.stop()
