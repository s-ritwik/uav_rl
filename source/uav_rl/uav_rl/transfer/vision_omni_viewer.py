from __future__ import annotations

import omni.ui as ui


class OmniVisionOverlayWindow:
    """Small Isaac UI window that displays RGBA frames through a ByteImageProvider."""

    def __init__(self, title: str, width: int, height: int):
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(title, width=max(int(width), 160), height=max(int(height), 120))
        with self._window.frame:
            with ui.VStack():
                ui.ImageWithProvider(self._provider)

    def update_rgba(self, rgba_bytes: bytes, width: int, height: int) -> None:
        self._provider.set_bytes_data(list(rgba_bytes), [int(width), int(height)])

    def destroy(self) -> None:
        if self._window is not None:
            self._window.visible = False
            self._window = None

