"""Tkinter desktop control panel for Blink Mouse Control."""

from __future__ import annotations

import threading
import tkinter as tk

import customtkinter as ctk

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
        self._is_closing = False
        self._stop_in_progress = False

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Blink Mouse Control")
        self.root.geometry("460x560")
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
        self.recalibrate_button: ctk.CTkButton | None = None
        self.start_button: ctk.CTkButton | None = None
        self.preset_dropdown: ctk.CTkComboBox | None = None
        self.sensitivity_slider: ctk.CTkSlider | None = None
        self.cursor_switch: ctk.CTkSwitch | None = None

        self._build_layout()
        self._schedule_status_poll()

    def _build_layout(self) -> None:
        """Build and place all UI widgets in the main window."""
        container = ctk.CTkFrame(self.root, corner_radius=12)
        container.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        container.columnconfigure(0, weight=1)

        status_frame = ctk.CTkFrame(container, corner_radius=10)
        status_frame.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=(10, 8))
        status_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            status_frame,
            text="Status",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(10, 2))

        ctk.CTkLabel(status_frame, text="System state", font=ctk.CTkFont(size=12)).grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(12, 10),
            pady=(28, 10),
        )
        ctk.CTkLabel(
            status_frame,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(
            row=0,
            column=1,
            sticky=tk.W,
            pady=(28, 10),
        )

        controls_frame = ctk.CTkFrame(container, corner_radius=10)
        controls_frame.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=(0, 8))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            controls_frame,
            text="Controls",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(10, 2))

        start_button = ctk.CTkButton(
            controls_frame,
            textvariable=self.start_button_var,
            command=self._toggle_start_stop,
            corner_radius=10,
            height=36,
        )
        start_button.grid(row=1, column=0, padx=(12, 6), pady=(8, 12), sticky=tk.EW)
        self.start_button = start_button

        recalibrate_button = ctk.CTkButton(
            controls_frame,
            text="Recalibrate",
            command=self._request_recalibration,
            corner_radius=10,
            height=36,
        )
        recalibrate_button.grid(row=1, column=1, padx=(6, 12), pady=(8, 12), sticky=tk.EW)
        self.recalibrate_button = recalibrate_button

        settings_frame = ctk.CTkFrame(container, corner_radius=10)
        settings_frame.grid(row=2, column=0, sticky=tk.EW, padx=10, pady=(0, 8))
        settings_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            settings_frame,
            text="Settings",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky=tk.W, padx=12, pady=(10, 2))

        ctk.CTkLabel(settings_frame, text="Sensitivity preset", font=ctk.CTkFont(size=12)).grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(34, 4),
        )

        preset_dropdown = ctk.CTkComboBox(
            settings_frame,
            textvariable=self.preset_var,
            values=list(PRESET_THRESHOLDS.keys()),
            command=self._on_preset_selected,
            corner_radius=8,
            height=32,
        )
        preset_dropdown.grid(row=1, column=0, sticky=tk.EW, padx=12)
        self.preset_dropdown = preset_dropdown

        ctk.CTkLabel(settings_frame, text="Manual sensitivity (EAR threshold)", font=ctk.CTkFont(size=12)).grid(
            row=2,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(10, 4),
        )

        sensitivity_scale = ctk.CTkSlider(
            settings_frame,
            from_=0.12,  # type: ignore[arg-type]
            to=0.35,  # type: ignore[arg-type]
            variable=self.sensitivity_var,
            command=self._on_sensitivity_changed,
            number_of_steps=230,
            height=16,
        )
        sensitivity_scale.grid(row=3, column=0, sticky=tk.EW, padx=12)
        self.sensitivity_slider = sensitivity_scale

        sensitivity_value = ctk.CTkLabel(
            settings_frame,
            textvariable=self.sensitivity_display_var,
            font=ctk.CTkFont(size=12),
        )
        sensitivity_value.grid(row=4, column=0, sticky=tk.E, padx=12, pady=(4, 0))

        cursor_toggle = ctk.CTkSwitch(
            settings_frame,
            text="Enable cursor control",
            variable=self.cursor_enabled_var,
            onvalue=True,
            offvalue=False,
            command=self._on_cursor_toggle,
        )
        cursor_toggle.grid(row=5, column=0, sticky=tk.W, padx=12, pady=(12, 10))
        self.cursor_switch = cursor_toggle

        stats_frame = ctk.CTkFrame(container, corner_radius=10)
        stats_frame.grid(row=3, column=0, sticky=tk.EW, padx=10, pady=(0, 10))
        stats_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            stats_frame,
            text="Live Statistics",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(10, 6))

        ctk.CTkLabel(stats_frame, text="FPS", font=ctk.CTkFont(size=12)).grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=12,
        )
        ctk.CTkLabel(stats_frame, textvariable=self.fps_var, font=ctk.CTkFont(size=12)).grid(
            row=1,
            column=1,
            sticky=tk.E,
            padx=12,
        )

        ctk.CTkLabel(stats_frame, text="Current EAR", font=ctk.CTkFont(size=12)).grid(
            row=2,
            column=0,
            sticky=tk.W,
            padx=12,
        )
        ctk.CTkLabel(stats_frame, textvariable=self.ear_var, font=ctk.CTkFont(size=12)).grid(
            row=2,
            column=1,
            sticky=tk.E,
            padx=12,
        )

        ctk.CTkLabel(stats_frame, text="Blink count", font=ctk.CTkFont(size=12)).grid(
            row=3,
            column=0,
            sticky=tk.W,
            padx=12,
        )
        ctk.CTkLabel(stats_frame, textvariable=self.blink_count_var, font=ctk.CTkFont(size=12)).grid(
            row=3,
            column=1,
            sticky=tk.E,
            padx=12,
        )

        ctk.CTkLabel(stats_frame, text="Current threshold", font=ctk.CTkFont(size=12)).grid(
            row=4,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(0, 10),
        )
        ctk.CTkLabel(stats_frame, textvariable=self.current_threshold_var, font=ctk.CTkFont(size=12)).grid(
            row=4,
            column=1,
            sticky=tk.E,
            padx=12,
            pady=(0, 10),
        )

        # Initialize slider from the default preset for predictable startup behavior.
        self._apply_preset("Medium")

    def _toggle_start_stop(self) -> None:
        """Start or stop the background blink detection thread."""
        if self._stop_in_progress:
            return

        if self._is_running():
            self._stop_detection()
            return

        if self.status_var.get() == "Stopping...":
            return

        self.control = DetectionControl()
        self.control.set_threshold_override(self.sensitivity_var.get())
        self.control.set_cursor_control_enabled(self.cursor_enabled_var.get())

        self.worker_thread = threading.Thread(
            target=run_detection,
            kwargs={"config": self.config, "control": self.control},
            daemon=False,
        )
        self.worker_thread.start()
        self._set_status("Running")
        self.start_button_var.set("Stop")
        self._set_interactions_enabled(True)

    def _stop_detection(self) -> None:
        """Signal detection to stop and update UI status immediately."""
        if self._stop_in_progress:
            return

        if self.control is None:
            self._set_status("Stopped")
            return

        self._stop_in_progress = True
        self.control.request_stop()
        self._set_status("Stopping...")
        self.start_button_var.set("Stopping...")
        self._set_interactions_enabled(False)
        self._wait_for_worker_stop()

    def _wait_for_worker_stop(self) -> None:
        """Poll worker thread completion without blocking the Tk main loop."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.root.after(100, self._wait_for_worker_stop)
            return

        if self.control is not None:
            self.control.wait_until_stopped(timeout=0.0)

        self.worker_thread = None
        self.control = None
        self._stop_in_progress = False
        self.start_button_var.set("Start")

        if self._is_closing:
            self.root.destroy()
            return

        self._set_status("Stopped")
        self._set_interactions_enabled(True)

    def _on_sensitivity_changed(self, _value: float) -> None:
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
        self._on_sensitivity_changed(threshold)

    def _on_preset_selected(self, selected_value: str) -> None:
        """Handle preset dropdown changes and apply threshold in real time."""
        self._apply_preset(selected_value)

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
        if not self._is_running() and not self._stop_in_progress and self.status_var.get() not in (
            "Stopped",
            "Starting...",
        ):
            self._set_status("Stopped")
            self.start_button_var.set("Start")
        self.root.after(200, self._schedule_status_poll)

    def _set_interactions_enabled(self, enabled: bool) -> None:
        """Enable or disable interactive controls during lifecycle transitions."""
        state = "normal" if enabled else "disabled"
        if self.start_button is not None:
            self.start_button.configure(state=state)
        if self.recalibrate_button is not None:
            self.recalibrate_button.configure(state=state)
        if self.preset_dropdown is not None:
            self.preset_dropdown.configure(state=state)
        if self.sensitivity_slider is not None:
            self.sensitivity_slider.configure(state=state)
        if self.cursor_switch is not None:
            self.cursor_switch.configure(state=state)

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
        self._is_closing = True

        if self._is_running() or self._stop_in_progress:
            self._stop_detection()
            return

        self.root.destroy()

    def run(self) -> None:
        """Run the Tkinter event loop."""
        self.root.mainloop()


def launch_control_panel(config: DetectionConfig | None = None) -> None:
    """Start the desktop control panel."""
    BlinkControlPanel(config=config).run()
