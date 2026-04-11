"""Tkinter desktop control panel for Blink Mouse Control."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from .config import DetectionConfig
from .detector import DetectionControl, run_detection


PRESET_THRESHOLDS = {
    "Low": 0.18,
    "Medium": 0.22,
    "High": 0.26,
}


class BlinkControlPanel:
    """Desktop UI to control the blink detection runtime."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.control: DetectionControl | None = None
        self.worker_thread: threading.Thread | None = None

        self.root = tk.Tk()
        self.root.title("Blink Mouse Control")
        self.root.geometry("420x490")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_var = tk.StringVar(value="Stopped")
        self.start_button_var = tk.StringVar(value="Start")
        self.sensitivity_var = tk.DoubleVar(value=self.config.fallback_ear_threshold)
        self.sensitivity_display_var = tk.StringVar(value=f"{self.sensitivity_var.get():.3f}")
        self.preset_var = tk.StringVar(value="Medium")
        self.cursor_enabled_var = tk.BooleanVar(value=True)
        self.fps_var = tk.StringVar(value="0.0")
        self.ear_var = tk.StringVar(value="-")
        self.blink_count_var = tk.StringVar(value="0")
        self.current_threshold_var = tk.StringVar(value=f"{self.sensitivity_var.get():.3f}")
        self.recalibrate_button: ttk.Button | None = None

        self._setup_style()

        self._build_layout()
        self._schedule_status_poll()

    def _setup_style(self) -> None:
        """Configure a compact, professional visual style for widgets."""
        style = ttk.Style(self.root)
        style.configure("Section.TLabelframe", padding=10)
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("StatusValue.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Body.TLabel", font=("Segoe UI", 9))
        style.configure("Primary.TButton", padding=(8, 6))
        style.configure("Secondary.TButton", padding=(8, 6))

    def _build_layout(self) -> None:
        """Build and place all UI widgets in the main window."""
        container = ttk.Frame(self.root, padding=14)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)

        status_frame = ttk.LabelFrame(container, text="Status", style="Section.TLabelframe")
        status_frame.grid(row=0, column=0, sticky=tk.EW, pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)

        ttk.Label(status_frame, text="System state", style="Body.TLabel").grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, 10),
        )
        ttk.Label(status_frame, textvariable=self.status_var, style="StatusValue.TLabel").grid(
            row=0,
            column=1,
            sticky=tk.W,
        )

        controls_frame = ttk.LabelFrame(container, text="Controls", style="Section.TLabelframe")
        controls_frame.grid(row=1, column=0, sticky=tk.EW, pady=(0, 10))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)

        start_button = ttk.Button(
            controls_frame,
            textvariable=self.start_button_var,
            command=self._toggle_start_stop,
            style="Primary.TButton",
        )
        start_button.grid(row=0, column=0, padx=(0, 6), sticky=tk.EW)

        self.recalibrate_button = ttk.Button(
            controls_frame,
            text="Recalibrate",
            command=self._request_recalibration,
            style="Secondary.TButton",
        )
        self.recalibrate_button.grid(row=0, column=1, padx=(6, 0), sticky=tk.EW)

        settings_frame = ttk.LabelFrame(container, text="Settings", style="Section.TLabelframe")
        settings_frame.grid(row=2, column=0, sticky=tk.EW)
        settings_frame.columnconfigure(0, weight=1)

        ttk.Label(settings_frame, text="Sensitivity preset", style="Body.TLabel").grid(
            row=0,
            column=0,
            sticky=tk.W,
            pady=(0, 4),
        )

        preset_dropdown = ttk.Combobox(
            settings_frame,
            textvariable=self.preset_var,
            values=list(PRESET_THRESHOLDS.keys()),
            state="readonly",
        )
        preset_dropdown.grid(row=1, column=0, sticky=tk.EW)
        preset_dropdown.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Label(settings_frame, text="Manual sensitivity (EAR threshold)", style="Body.TLabel").grid(
            row=2,
            column=0,
            sticky=tk.W,
            pady=(10, 4),
        )

        sensitivity_scale = ttk.Scale(
            settings_frame,
            from_=0.12,
            to=0.35,
            variable=self.sensitivity_var,
            command=self._on_sensitivity_changed,
        )
        sensitivity_scale.grid(row=3, column=0, sticky=tk.EW)

        sensitivity_value = ttk.Label(
            settings_frame,
            textvariable=self.sensitivity_display_var,
            style="Body.TLabel",
        )
        sensitivity_value.grid(row=4, column=0, sticky=tk.E, pady=(4, 0))

        cursor_toggle = ttk.Checkbutton(
            settings_frame,
            text="Enable cursor control",
            variable=self.cursor_enabled_var,
            command=self._on_cursor_toggle,
        )
        cursor_toggle.grid(row=5, column=0, sticky=tk.W, pady=(12, 0))

        stats_frame = ttk.LabelFrame(container, text="Live Statistics", style="Section.TLabelframe")
        stats_frame.grid(row=3, column=0, sticky=tk.EW, pady=(10, 0))
        stats_frame.columnconfigure(1, weight=1)

        ttk.Label(stats_frame, text="FPS", style="Body.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(stats_frame, textvariable=self.fps_var, style="Body.TLabel").grid(row=0, column=1, sticky=tk.E)

        ttk.Label(stats_frame, text="Current EAR", style="Body.TLabel").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(stats_frame, textvariable=self.ear_var, style="Body.TLabel").grid(row=1, column=1, sticky=tk.E)

        ttk.Label(stats_frame, text="Blink count", style="Body.TLabel").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(stats_frame, textvariable=self.blink_count_var, style="Body.TLabel").grid(
            row=2,
            column=1,
            sticky=tk.E,
        )

        ttk.Label(stats_frame, text="Current threshold", style="Body.TLabel").grid(
            row=3,
            column=0,
            sticky=tk.W,
        )
        ttk.Label(stats_frame, textvariable=self.current_threshold_var, style="Body.TLabel").grid(
            row=3,
            column=1,
            sticky=tk.E,
        )

        # Initialize slider from the default preset for predictable startup behavior.
        self._apply_preset("Medium")

    def _toggle_start_stop(self) -> None:
        """Start or stop the background blink detection thread."""
        if self._is_running():
            self._stop_detection()
            return

        self.control = DetectionControl()
        self.control.set_threshold_override(self.sensitivity_var.get())
        self.control.set_cursor_control_enabled(self.cursor_enabled_var.get())

        self.worker_thread = threading.Thread(
            target=run_detection,
            kwargs={"config": self.config, "control": self.control},
            daemon=True,
        )
        self.worker_thread.start()
        self._set_status("Running")
        self.start_button_var.set("Stop")

    def _stop_detection(self) -> None:
        """Signal detection to stop and update UI status immediately."""
        if self.control is not None:
            self.control.request_stop()
        self._set_status("Stopping...")
        self.start_button_var.set("Start")

    def _on_sensitivity_changed(self, _value: str) -> None:
        """Apply sensitivity changes to the running detector when available."""
        self.sensitivity_display_var.set(f"{self.sensitivity_var.get():.3f}")
        if self.control is not None:
            self.control.set_threshold_override(self.sensitivity_var.get())

    def _apply_preset(self, preset_name: str) -> None:
        """Apply a named sensitivity preset to the manual threshold value."""
        threshold = PRESET_THRESHOLDS.get(preset_name)
        if threshold is None:
            return
        self.sensitivity_var.set(threshold)
        self._on_sensitivity_changed(str(threshold))

    def _on_preset_selected(self, _event: tk.Event) -> None:
        """Handle preset dropdown changes and apply threshold in real time."""
        self._apply_preset(self.preset_var.get())

    def _on_cursor_toggle(self) -> None:
        """Enable or disable action dispatching in the detector."""
        if self.control is not None:
            self.control.set_cursor_control_enabled(self.cursor_enabled_var.get())

    def _request_recalibration(self) -> None:
        """Ask the running detector to perform recalibration on the next iteration."""
        if self.control is not None:
            self.control.request_recalibration()

    def _is_running(self) -> bool:
        """Return True if the detector background thread is active."""
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _schedule_status_poll(self) -> None:
        """Refresh UI state periodically to reflect worker thread status."""
        self._refresh_live_stats()
        if not self._is_running() and self.status_var.get() != "Stopped":
            self._set_status("Stopped")
            self.start_button_var.set("Start")
        self.root.after(200, self._schedule_status_poll)

    def _refresh_live_stats(self) -> None:
        """Pull a thread-safe live stats snapshot from the detector and update labels."""
        if self.control is None:
            self.fps_var.set("0.0")
            self.ear_var.set("-")
            self.blink_count_var.set("0")
            self.current_threshold_var.set(f"{self.sensitivity_var.get():.3f}")
            return

        stats = self.control.get_live_stats()
        fps_value = stats.get("fps")
        self.fps_var.set(f"{float(fps_value) if fps_value is not None else 0.0:.1f}")

        ear = stats["ear"]
        self.ear_var.set("-" if ear is None else f"{float(ear):.3f}")

        blink_count_value = stats.get("blink_count")
        self.blink_count_var.set(str(int(blink_count_value) if blink_count_value is not None else 0))

        threshold_value = stats.get("threshold")
        self.current_threshold_var.set(
            f"{float(threshold_value) if threshold_value is not None else self.sensitivity_var.get():.3f}"
        )

    def _set_status(self, status: str) -> None:
        """Set and normalize user-visible status text."""
        self.status_var.set(status)

    def _on_close(self) -> None:
        """Gracefully stop detection before closing the UI window."""
        self._stop_detection()
        self.root.after(250, self.root.destroy)

    def run(self) -> None:
        """Run the Tkinter event loop."""
        self.root.mainloop()


def launch_control_panel(config: DetectionConfig | None = None) -> None:
    """Start the desktop control panel."""
    BlinkControlPanel(config=config).run()
