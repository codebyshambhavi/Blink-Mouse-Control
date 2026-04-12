"""Tkinter desktop control panel for Blink Mouse Control."""

from __future__ import annotations

from collections import deque
import threading
import tkinter as tk

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import BEAUTY_FILTER_LEVELS, DetectionConfig
from .detector import DetectionControl, run_detection
from .settings import load_theme_mode, save_theme_mode


PRESET_THRESHOLDS = {
    "Low": 0.18,
    "Medium": 0.22,
    "High": 0.26,
}


THEME_PALETTES = {
    "Dark": {
        "background": "#141414",
        "surface": "#1c1c1c",
        "card": "#242424",
        "text": "#f5f5f5",
        "muted": "#c7c7c7",
        "border": "#3a3a3a",
        "accent": "#2563eb",
        "accent_hover": "#1d4ed8",
        "slider_fg": "#2a2a2a",
    },
    "Light": {
        "background": "#f3f4f6",
        "surface": "#e5e7eb",
        "card": "#ffffff",
        "text": "#111827",
        "muted": "#4b5563",
        "border": "#d1d5db",
        "accent": "#2563eb",
        "accent_hover": "#1d4ed8",
        "slider_fg": "#e5e7eb",
    },
}


class BlinkControlPanel:
    """Desktop UI to control the blink detection runtime."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.control: DetectionControl | None = None
        self.worker_thread: threading.Thread | None = None
        self._is_closing = False
        self._stop_in_progress = False
        self.theme_mode = load_theme_mode()

        ctk.set_appearance_mode(self.theme_mode)
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Blink Mouse Control")
        self.root.geometry("860x560")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_var = tk.StringVar(value="Stopped")
        self.start_button_var = tk.StringVar(value="Start")
        self.sensitivity_var = tk.DoubleVar(value=self.config.fallback_ear_threshold)
        self.sensitivity_display_var = tk.StringVar(value=f"{self.sensitivity_var.get():.3f}")
        self.preset_var = tk.StringVar(value="Medium")
        self.cursor_enabled_var = tk.BooleanVar(value=True)
        self.beauty_filter_level_var = tk.StringVar(value=self.config.beauty_filter_level)
        self.dark_mode_var = tk.BooleanVar(value=self.theme_mode == "Dark")
        self.fps_var = tk.StringVar(value="0.0")
        self.ear_var = tk.StringVar(value="-")
        self.blink_count_var = tk.StringVar(value="0")
        self.current_threshold_var = tk.StringVar(value=f"{self.sensitivity_var.get():.3f}")
        self.recalibrate_button: ctk.CTkButton | None = None
        self.start_button: ctk.CTkButton | None = None
        self.preset_dropdown: ctk.CTkComboBox | None = None
        self.sensitivity_slider: ctk.CTkSlider | None = None
        self.cursor_switch: ctk.CTkSwitch | None = None
        self.beauty_filter_dropdown: ctk.CTkComboBox | None = None
        self.theme_switch: ctk.CTkSwitch | None = None
        self.container: ctk.CTkFrame | None = None
        self.top_bar: ctk.CTkFrame | None = None
        self.status_card: ctk.CTkFrame | None = None
        self.metrics_card: ctk.CTkFrame | None = None
        self.content_frame: ctk.CTkFrame | None = None
        self.left_panel: ctk.CTkFrame | None = None
        self.controls_frame: ctk.CTkFrame | None = None
        self.settings_frame: ctk.CTkFrame | None = None
        self.stats_frame: ctk.CTkFrame | None = None
        self.status_value_label: ctk.CTkLabel | None = None
        self.metrics_value_label: ctk.CTkLabel | None = None
        self.stats_ear_value_label: ctk.CTkLabel | None = None
        self.blink_count_value_label: ctk.CTkLabel | None = None
        self.threshold_value_label: ctk.CTkLabel | None = None
        self.sensitivity_value_label: ctk.CTkLabel | None = None
        self.ear_history: deque[float] = deque([self.sensitivity_var.get()] * 80, maxlen=80)
        self.ear_figure: Figure | None = None
        self.ear_axes = None
        self.ear_line = None
        self.ear_canvas: FigureCanvasTkAgg | None = None

        self._build_layout()
        self._apply_theme(self.theme_mode, persist=False)
        self._schedule_status_poll()

    def _build_layout(self) -> None:
        """Build and place all UI widgets in the main window."""
        container = ctk.CTkFrame(self.root, corner_radius=12)
        container.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        top_bar = ctk.CTkFrame(container, corner_radius=10)
        top_bar.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=(10, 8))
        top_bar.columnconfigure(0, weight=2)
        top_bar.columnconfigure(1, weight=1)

        status_card = ctk.CTkFrame(top_bar, corner_radius=8)
        status_card.grid(row=0, column=0, padx=(8, 5), pady=8, sticky=tk.EW)
        ctk.CTkLabel(status_card, text="System", font=ctk.CTkFont(size=12)).grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(10, 0),
        )
        self.status_value_label = ctk.CTkLabel(
            status_card,
            text=self.status_var.get(),
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#ef4444",
        )
        self.status_value_label.grid(row=1, column=0, sticky=tk.W, padx=12, pady=(0, 12))

        metrics_card = ctk.CTkFrame(top_bar, corner_radius=8)
        metrics_card.grid(row=0, column=1, padx=(5, 8), pady=8, sticky=tk.EW)
        ctk.CTkLabel(metrics_card, text="Live Metrics", font=ctk.CTkFont(size=12)).grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(10, 2),
        )
        self.metrics_value_label = ctk.CTkLabel(
            metrics_card,
            text=f"FPS {self.fps_var.get()}   |   EAR {self.ear_var.get()}",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.metrics_value_label.grid(row=1, column=0, sticky=tk.W, padx=12, pady=(0, 12))

        content_frame = ctk.CTkFrame(container, corner_radius=10)
        content_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=10, pady=(0, 10))
        content_frame.columnconfigure(0, weight=3)
        content_frame.columnconfigure(1, weight=2)
        content_frame.rowconfigure(0, weight=1)

        left_panel = ctk.CTkFrame(content_frame, corner_radius=8)
        left_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=(8, 5), pady=8)
        left_panel.columnconfigure(0, weight=1)

        controls_frame = ctk.CTkFrame(left_panel, corner_radius=8)
        controls_frame.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=(10, 8))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            controls_frame,
            text="Controls",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(12, 8))

        start_button = ctk.CTkButton(
            controls_frame,
            textvariable=self.start_button_var,
            command=self._toggle_start_stop,
            corner_radius=10,
            height=52,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
        )
        start_button.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 10), sticky=tk.EW)
        self.start_button = start_button

        recalibrate_button = ctk.CTkButton(
            controls_frame,
            text="Recalibrate",
            command=self._request_recalibration,
            corner_radius=10,
            height=34,
            fg_color="transparent",
            border_width=1,
            border_color="#4b5563",
            text_color="#d1d5db",
            hover_color="#374151",
        )
        recalibrate_button.grid(row=2, column=0, columnspan=2, padx=12, pady=(0, 12), sticky=tk.EW)
        self.recalibrate_button = recalibrate_button

        settings_frame = ctk.CTkFrame(left_panel, corner_radius=8)
        settings_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=10, pady=(0, 10))
        settings_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            settings_frame,
            text="Settings",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky=tk.W, padx=12, pady=(12, 8))

        ctk.CTkLabel(settings_frame, text="Sensitivity preset", font=ctk.CTkFont(size=12)).grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(4, 6),
        )

        preset_dropdown = ctk.CTkComboBox(
            settings_frame,
            variable=self.preset_var,
            values=list(PRESET_THRESHOLDS.keys()),
            command=self._on_preset_selected,
            corner_radius=8,
            height=32,
        )
        preset_dropdown.grid(row=2, column=0, sticky=tk.EW, padx=12, pady=(0, 8))
        self.preset_dropdown = preset_dropdown

        ctk.CTkLabel(settings_frame, text="Manual sensitivity (EAR threshold)", font=ctk.CTkFont(size=12)).grid(
            row=3,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(8, 6),
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
        sensitivity_scale.grid(row=4, column=0, sticky=tk.EW, padx=12, pady=(0, 6))
        self.sensitivity_slider = sensitivity_scale

        sensitivity_value = ctk.CTkLabel(
            settings_frame,
            text=self.sensitivity_display_var.get(),
            font=ctk.CTkFont(size=12),
        )
        sensitivity_value.grid(row=5, column=0, sticky=tk.E, padx=12, pady=(0, 8))
        self.sensitivity_value_label = sensitivity_value

        cursor_toggle = ctk.CTkSwitch(
            settings_frame,
            text="Enable cursor control",
            variable=self.cursor_enabled_var,
            onvalue=True,
            offvalue=False,
            command=self._on_cursor_toggle,
        )
        cursor_toggle.grid(row=6, column=0, sticky=tk.W, padx=12, pady=(8, 14))
        self.cursor_switch = cursor_toggle

        ctk.CTkLabel(settings_frame, text="Beauty filter level", font=ctk.CTkFont(size=12)).grid(
            row=7,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(2, 6),
        )

        beauty_dropdown = ctk.CTkComboBox(
            settings_frame,
            variable=self.beauty_filter_level_var,
            values=list(BEAUTY_FILTER_LEVELS),
            command=self._on_beauty_filter_level_selected,
            corner_radius=8,
            height=32,
        )
        beauty_dropdown.grid(row=8, column=0, sticky=tk.EW, padx=12, pady=(0, 14))
        self.beauty_filter_dropdown = beauty_dropdown

        self.theme_switch = ctk.CTkSwitch(
            settings_frame,
            text="Dark Mode",
            variable=self.dark_mode_var,
            onvalue=True,
            offvalue=False,
            command=self._on_theme_toggle,
        )
        self.theme_switch.grid(row=9, column=0, sticky=tk.W, padx=12, pady=(0, 14))

        stats_frame = ctk.CTkFrame(content_frame, corner_radius=8)
        stats_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(5, 8), pady=8)
        stats_frame.columnconfigure(1, weight=1)
        stats_frame.rowconfigure(6, weight=1)

        ctk.CTkLabel(
            stats_frame,
            text="Live Monitoring",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(12, 6))

        ctk.CTkLabel(stats_frame, text="Blink Count", font=ctk.CTkFont(size=12)).grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(2, 2),
        )

        self.blink_count_value_label = ctk.CTkLabel(
            stats_frame,
            text=self.blink_count_var.get(),
            font=ctk.CTkFont(size=36, weight="bold"),
        )
        self.blink_count_value_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(0, 16))

        ctk.CTkLabel(stats_frame, text="Current EAR", font=ctk.CTkFont(size=12)).grid(
            row=3,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(0, 8),
        )
        self.stats_ear_value_label = ctk.CTkLabel(
            stats_frame,
            text=self.ear_var.get(),
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.stats_ear_value_label.grid(row=3, column=1, sticky=tk.E, padx=12, pady=(0, 8))

        ctk.CTkLabel(stats_frame, text="Current threshold", font=ctk.CTkFont(size=12)).grid(
            row=4,
            column=0,
            sticky=tk.W,
            padx=12,
            pady=(0, 10),
        )
        self.threshold_value_label = ctk.CTkLabel(
            stats_frame,
            text=self.current_threshold_var.get(),
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.threshold_value_label.grid(row=4, column=1, sticky=tk.E, padx=12, pady=(0, 10))

        ctk.CTkLabel(stats_frame, text="EAR trend", font=ctk.CTkFont(size=12)).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky=tk.W,
            padx=12,
            pady=(2, 6),
        )
        self._init_ear_graph(stats_frame)

        self.container = container
        self.top_bar = top_bar
        self.status_card = status_card
        self.metrics_card = metrics_card
        self.content_frame = content_frame
        self.left_panel = left_panel
        self.controls_frame = controls_frame
        self.settings_frame = settings_frame
        self.stats_frame = stats_frame

        # Initialize slider from the default preset for predictable startup behavior.
        self._apply_preset("Medium")

    def _init_ear_graph(self, parent: ctk.CTkFrame) -> None:
        """Create and place the embedded EAR graph canvas."""
        figure = Figure(figsize=(4.0, 2.0), dpi=100)
        figure.patch.set_facecolor("#1f2937")
        axes = figure.add_subplot(111)
        axes.set_facecolor("#111827")

        line, = axes.plot([], [], color="#3b82f6", linewidth=2.0)
        axes.set_xlim(0, max(len(self.ear_history) - 1, 1))
        axes.set_ylim(0.10, 0.35)
        axes.set_xticks([])
        axes.tick_params(axis="y", colors="#9ca3af", labelsize=8, length=0)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.spines["bottom"].set_visible(False)
        axes.spines["left"].set_color("#4b5563")
        axes.margins(x=0)

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=6, column=0, columnspan=2, sticky=tk.NSEW, padx=12, pady=(0, 12))
        canvas_widget.configure(highlightthickness=0, bg="#1f2937")

        self.ear_figure = figure
        self.ear_axes = axes
        self.ear_line = line
        self.ear_canvas = canvas
        self._update_ear_graph(self.sensitivity_var.get())

    def _apply_theme(self, mode: str, *, persist: bool = True) -> None:
        """Apply the selected UI theme to all visible widgets."""
        normalized_mode = mode if mode in THEME_PALETTES else "Dark"
        theme = THEME_PALETTES[normalized_mode]

        self.theme_mode = normalized_mode
        self.dark_mode_var.set(normalized_mode == "Dark")
        ctk.set_appearance_mode(normalized_mode)
        self.root.configure(fg_color=theme["background"])

        for frame in (self.container, self.top_bar, self.content_frame):
            if frame is not None:
                frame.configure(fg_color=theme["surface"])

        for card in (self.status_card, self.metrics_card, self.left_panel, self.controls_frame, self.settings_frame, self.stats_frame):
            if card is not None:
                card.configure(fg_color=theme["card"])

        for label in (
            self.status_value_label,
            self.metrics_value_label,
            self.stats_ear_value_label,
            self.blink_count_value_label,
            self.threshold_value_label,
            self.sensitivity_value_label,
        ):
            if label is not None:
                label.configure(text_color=theme["text"])

        if self.metrics_value_label is not None:
            self.metrics_value_label.configure(text_color=theme["accent"])

        if self.blink_count_value_label is not None:
            self.blink_count_value_label.configure(text_color=theme["accent"])

        if self.threshold_value_label is not None:
            self.threshold_value_label.configure(text_color=theme["accent"])

        if self.start_button is not None:
            self.start_button.configure(
                fg_color=theme["accent"],
                hover_color=theme["accent_hover"],
                text_color="#ffffff",
            )

        if self.recalibrate_button is not None:
            self.recalibrate_button.configure(
                fg_color="transparent",
                border_color=theme["border"],
                text_color=theme["text"],
                hover_color=theme["surface"],
            )

        for combo_box in (self.preset_dropdown, self.beauty_filter_dropdown):
            if combo_box is not None:
                combo_box.configure(
                    fg_color=theme["card"],
                    border_color=theme["border"],
                    button_color=theme["accent"],
                    button_hover_color=theme["accent_hover"],
                    text_color=theme["text"],
                )

        if self.sensitivity_slider is not None:
            self.sensitivity_slider.configure(
                fg_color=theme["slider_fg"],
                progress_color=theme["accent"],
                button_color=theme["accent"],
                button_hover_color=theme["accent_hover"],
            )

        for switch in (self.cursor_switch, self.theme_switch):
            if switch is not None:
                switch.configure(
                    text_color=theme["text"],
                    fg_color=theme["card"],
                    progress_color=theme["accent"],
                    button_color=theme["accent"],
                    button_hover_color=theme["accent_hover"],
                )

        if self.ear_figure is not None and self.ear_axes is not None:
            self.ear_figure.patch.set_facecolor(theme["card"])
            self.ear_axes.set_facecolor(theme["surface"])
            self.ear_axes.tick_params(axis="y", colors=theme["muted"], labelsize=8, length=0)
            self.ear_axes.spines["left"].set_color(theme["border"])
            if self.ear_canvas is not None:
                self.ear_canvas.get_tk_widget().configure(bg=theme["card"])
                self.ear_canvas.draw_idle()

        self._set_status(self.status_var.get())

        if persist:
            save_theme_mode(normalized_mode)

    def _on_theme_toggle(self) -> None:
        """Switch the UI between dark and light themes."""
        self._apply_theme("Dark" if self.dark_mode_var.get() else "Light")

    def _update_ear_graph(self, ear_value: float | None) -> None:
        """Append a new EAR sample and redraw the graph efficiently."""
        if self.ear_axes is None or self.ear_line is None or self.ear_canvas is None:
            return

        if ear_value is None:
            fallback = self.ear_history[-1] if self.ear_history else self.sensitivity_var.get()
            self.ear_history.append(fallback)
        else:
            self.ear_history.append(float(ear_value))

        values = list(self.ear_history)
        self.ear_line.set_data(range(len(values)), values)
        self.ear_axes.set_xlim(0, max(len(values) - 1, 1))

        min_value = min(values)
        max_value = max(values)
        padding = 0.02
        lower = max(0.05, min_value - padding)
        upper = min(0.50, max_value + padding)
        if upper - lower < 0.08:
            midpoint = (upper + lower) / 2
            lower = max(0.05, midpoint - 0.04)
            upper = min(0.50, midpoint + 0.04)
        self.ear_axes.set_ylim(lower, upper)

        self.ear_canvas.draw_idle()

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
        self.control.set_beauty_filter_level(self.beauty_filter_level_var.get())

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
        if self.sensitivity_value_label is not None:
            self.sensitivity_value_label.configure(text=self.sensitivity_display_var.get())
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

    def _on_beauty_filter_level_selected(self, selected_value: str) -> None:
        """Apply the selected beauty filter intensity to the detector."""
        if self.control is not None:
            self.control.set_beauty_filter_level(selected_value)

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
        if self.beauty_filter_dropdown is not None:
            self.beauty_filter_dropdown.configure(state=state)

    def _refresh_live_stats(self) -> None:
        """Pull a thread-safe live stats snapshot from the detector and update labels."""
        if self.control is None:
            self.fps_var.set("0.0")
            self.ear_var.set("-")
            self.blink_count_var.set("0")
            self.current_threshold_var.set(f"{self.sensitivity_var.get():.3f}")
            self._update_ear_graph(None)
            self._sync_stats_labels()
            return

        stats = self.control.get_live_stats()
        fps_value = stats.get("fps")
        self.fps_var.set(f"{float(fps_value) if fps_value is not None else 0.0:.1f}")

        ear = stats["ear"]
        self.ear_var.set("-" if ear is None else f"{float(ear):.3f}")
        self._update_ear_graph(None if ear is None else float(ear))

        blink_count_value = stats.get("blink_count")
        self.blink_count_var.set(str(int(blink_count_value) if blink_count_value is not None else 0))

        threshold_value = stats.get("threshold")
        self.current_threshold_var.set(
            f"{float(threshold_value) if threshold_value is not None else self.sensitivity_var.get():.3f}"
        )

        beauty_level_value = stats.get("beauty_filter_level")
        if isinstance(beauty_level_value, str) and beauty_level_value != self.beauty_filter_level_var.get():
            self.beauty_filter_level_var.set(beauty_level_value)

        self._sync_stats_labels()

    def _sync_stats_labels(self) -> None:
        """Push cached stat strings into the visible labels."""
        if self.status_value_label is not None:
            self.status_value_label.configure(text=self.status_var.get())
        if self.metrics_value_label is not None:
            self.metrics_value_label.configure(text=f"FPS {self.fps_var.get()}   |   EAR {self.ear_var.get()}")
        if self.stats_ear_value_label is not None:
            self.stats_ear_value_label.configure(text=self.ear_var.get())
        if self.blink_count_value_label is not None:
            self.blink_count_value_label.configure(text=self.blink_count_var.get())
        if self.threshold_value_label is not None:
            self.threshold_value_label.configure(text=self.current_threshold_var.get())

    def _set_status(self, status: str) -> None:
        """Set and normalize user-visible status text."""
        self.status_var.set(status)
        if self.status_value_label is not None:
            color = "#ef4444"
            if status == "Running":
                color = "#10b981"
            elif status == "Stopping...":
                color = "#f59e0b"
            self.status_value_label.configure(text=status, text_color=color)

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
