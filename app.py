"""
PawVision AI — Cat vs Dog Image Classifier
Production Web Application powered by NiceGUI
"""

import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from nicegui import ui, app
import plotly.graph_objects as go
from PIL import Image

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import (
    BEST_MODEL_PATH,
    METRICS_JSON_PATH,
    HISTORY_JSON_PATH,
    SERVER_HOST,
    SERVER_PORT,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    MAX_IMAGE_FILE_SIZE_MB
)
from src.predictor import get_predictor
from src.evaluation import load_saved_metrics
from src.ui_components import THEME_CSS, get_theme_toggle_js
from download_data import is_dataset_ready

# Add custom CSS and scripts shared across all pages
ui.add_head_html(f"<style>{THEME_CSS}</style>", shared=True)
ui.add_head_html(f"<script>{get_theme_toggle_js()}</script>", shared=True)


# ============================================================================
# Application State Management for Upload and Prediction (Zero Stale State)
# ============================================================================
class ClassifierState:
    def __init__(self):
        self.image_bytes: Optional[bytes] = None
        self.filename: Optional[str] = None
        self.b64_preview: Optional[str] = None
        self.image_hash: Optional[str] = None
        self.dimensions: Optional[tuple] = None
        self.file_size_kb: float = 0.0
        self.prediction_result: Optional[Dict[str, Any]] = None
        self.is_classifying: bool = False

    def set_uploaded_image(self, name: str, data: bytes):
        """Safely stores raw image bytes, calculates identity hash and preview data."""
        self.clear()
        self.image_bytes = data
        self.filename = name
        self.file_size_kb = len(data) / 1024.0
        self.image_hash = hashlib.sha256(data).hexdigest()[:16]

        # Extract dimensions and create base64 preview
        try:
            with Image.open(io.BytesIO(data)) as img:
                self.dimensions = img.size
                fmt = (img.format or "JPEG").lower()
                if fmt == "jpg":
                    fmt = "jpeg"
            b64_str = base64.b64encode(data).decode("utf-8")
            self.b64_preview = f"data:image/{fmt};base64,{b64_str}"
        except Exception:
            self.b64_preview = None
            self.dimensions = None

    def clear(self):
        """Atomic reset of preview, uploaded bytes, and prediction results."""
        self.image_bytes = None
        self.filename = None
        self.b64_preview = None
        self.image_hash = None
        self.dimensions = None
        self.file_size_kb = 0.0
        self.prediction_result = None
        self.is_classifying = False


# State instance for classifier session
state = ClassifierState()


# ============================================================================
# Header and Layout Components
# ============================================================================
def render_header(current_page: str):
    """Renders the top branding navigation bar and centralized global theme toggle.
    Uses NiceGUI's native ui.dark_mode() with app.storage.user for bulletproof persistence.
    """
    # Read saved theme from server-side user storage (persists across pages & refreshes)
    is_dark = app.storage.user.get("dark_mode", False)

    # NiceGUI's native dark mode control — syncs Quasar body--dark class automatically
    dark = ui.dark_mode(value=is_dark)

    def toggle_theme():
        """Toggle and persist theme using server-side storage."""
        new_val = not app.storage.user.get("dark_mode", False)
        app.storage.user["dark_mode"] = new_val
        dark.set_value(new_val)
        # Also sync our CSS variable system on the client side
        ui.run_javascript(f"window.pawvisionApplyTheme && window.pawvisionApplyTheme({str(new_val).lower()})")

    with ui.header().classes("pv-header px-6 py-3 flex items-center justify-between shadow-sm"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("pets", size="2rem").classes("text-blue-600 dark:text-blue-400")
            with ui.column().classes("gap-0"):
                ui.label("PawVision AI").classes("text-xl font-bold tracking-tight text-blue-600 dark:text-blue-400")
                ui.label("Cat vs Dog Image Classifier").classes("text-xs text-slate-500 dark:text-slate-400 font-medium")

        with ui.row().classes("items-center gap-2"):
            pages = [
                ("Overview", "/"),
                ("Classifier", "/classifier"),
                ("Model Insights", "/insights"),
                ("Dataset & About", "/about")
            ]
            for title, route in pages:
                active = "bg-blue-600 text-white dark:bg-blue-600 dark:text-white" if current_page == title else "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                ui.button(title, on_click=lambda r=route: ui.navigate.to(r)).props("flat no-caps").classes(f"font-semibold text-sm rounded-lg px-3 py-1.5 {active}")

            ui.button(
                icon="light_mode" if is_dark else "dark_mode",
                on_click=toggle_theme
            ).props("flat round").classes(
                f"pv-theme-toggle ml-3 {'text-amber-400' if is_dark else 'text-slate-600'}"
            ).tooltip("Toggle Light / Dark Mode")

    # Sync our custom CSS variable system with the server-side value on page load
    ui.run_javascript(f"window.pawvisionApplyTheme && window.pawvisionApplyTheme({str(is_dark).lower()})")


# ============================================================================
# Page 1: Overview Dashboard
# ============================================================================
@ui.page("/")
def page_overview():
    render_header("Overview")

    metrics = load_saved_metrics()
    ready, counts = is_dataset_ready()
    model_exists = BEST_MODEL_PATH.exists()

    with ui.column().classes("w-full max-w-6xl mx-auto px-4 py-8 gap-8"):
        # Hero Banner
        with ui.card().classes("w-full pv-card p-8 bg-gradient-to-r from-blue-900/20 via-slate-900/10 to-emerald-900/20 border border-blue-500/30 rounded-2xl"):
            with ui.row().classes("w-full justify-between items-center"):
                with ui.column().classes("gap-2 max-w-2xl"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label("PROJECT 05 — COMPUTER VISION").classes("text-xs font-bold uppercase tracking-wider text-blue-500 dark:text-blue-400 bg-blue-500/10 px-2.5 py-1 rounded-full")
                        if model_exists:
                            ui.label("● Model Trained & Ready").classes("text-xs font-semibold text-emerald-500 bg-emerald-500/10 px-2.5 py-1 rounded-full")
                        else:
                            ui.label("○ Model Needs Training").classes("text-xs font-semibold text-amber-500 bg-amber-500/10 px-2.5 py-1 rounded-full")

                    ui.label("PawVision AI — Precision Cat & Dog Classification").classes("text-3xl font-extrabold tracking-tight")
                    ui.label("State-of-the-art transfer learning model built with MobileNetV2 and ImageNet weights. Provides instant, high-confidence inference on cat and dog images with detailed probability calibration.").classes("text-base text-slate-600 dark:text-slate-300 leading-relaxed")

                    with ui.row().classes("gap-3 mt-4"):
                        ui.button("Launch Classifier", icon="center_focus_strong", on_click=lambda: ui.navigate.to("/classifier")).props("unelevated no-caps").classes("bg-blue-600 hover:bg-blue-700 text-white font-bold px-6 py-2.5 rounded-xl shadow-lg")
                        ui.button("View Model Insights", icon="analytics", on_click=lambda: ui.navigate.to("/insights")).props("outline no-caps").classes("border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-200 font-semibold px-5 py-2.5 rounded-xl")

                ui.icon("pets", size="7rem").classes("text-blue-500/40 hidden md:block")

        # Metric Stats Row
        with ui.grid(columns=4).classes("w-full gap-4"):
            acc_val = f"{metrics.get('accuracy', 0.0) * 100:.1f}%" if metrics else "N/A"
            prec_val = f"{metrics.get('precision', 0.0) * 100:.1f}%" if metrics else "N/A"
            rec_val = f"{metrics.get('recall', 0.0) * 100:.1f}%" if metrics else "N/A"
            f1_val = f"{metrics.get('f1_score', 0.0) * 100:.1f}%" if metrics else "N/A"

            stat_items = [
                ("Test Accuracy", acc_val, "check_circle", "text-blue-500"),
                ("Precision", prec_val, "target", "text-emerald-500"),
                ("Recall", rec_val, "psychology", "text-amber-500"),
                ("F1-Score", f1_val, "auto_graph", "text-indigo-500")
            ]
            for label, val, icon_name, color_class in stat_items:
                with ui.card().classes("pv-metric-card"):
                    with ui.row().classes("w-full justify-between items-center"):
                        ui.label(label).classes("pv-metric-label")
                        ui.icon(icon_name, size="1.25rem").classes(color_class)
                    ui.label(val).classes(f"pv-metric-value {color_class}")
                    ui.label("Held-out test set").classes("text-xs text-slate-400")

        # Dataset & Model Cards
        with ui.grid(columns=2).classes("w-full gap-6"):
            with ui.card().classes("pv-card gap-4"):
                ui.label("Dataset Distribution").classes("text-lg font-bold flex items-center gap-2")
                ui.label("Official TensorFlow Cats vs Dogs filtered dataset, partitioned into a strict 80% train, 20% internal validation, and 1,000-image held-out test set.").classes("text-sm text-slate-600 dark:text-slate-300")

                with ui.column().classes("w-full gap-2 mt-2"):
                    train_total = counts.get('train_total', 2000) if counts else 2000
                    test_total = counts.get('val_total', 1000) if counts else 1000
                    with ui.row().classes("w-full justify-between text-sm py-1 border-b border-slate-200 dark:border-slate-800"):
                        ui.label("Training Set (Train + Val)")
                        ui.label(f"{train_total} images (50% Cats / 50% Dogs)").classes("font-semibold")

                    with ui.row().classes("w-full justify-between text-sm py-1 border-b border-slate-200 dark:border-slate-800"):
                        ui.label("Held-Out Test Set")
                        ui.label(f"{test_total} images (500 Cats / 500 Dogs)").classes("font-semibold")

                    with ui.row().classes("w-full justify-between text-sm py-1"):
                        ui.label("Class Balance")
                        ui.label("1:1 Perfect Balance").classes("font-semibold text-emerald-600 dark:text-emerald-400")

            with ui.card().classes("pv-card gap-4"):
                ui.label("Model Architecture").classes("text-lg font-bold")
                ui.label("MobileNetV2 feature extractor initialized with ImageNet weights, coupled with a custom binary classification head.").classes("text-sm text-slate-600 dark:text-slate-300")

                with ui.column().classes("w-full gap-2 mt-2"):
                    specs = [
                        ("Base Model", "MobileNetV2 (Pretrained on ImageNet)"),
                        ("Input Resolution", "160 × 160 × 3 (RGB)"),
                        ("Pooling & Regularization", "GlobalAveragePooling2D + 20% Dropout"),
                        ("Output Layer", "Dense(1, Sigmoid activation)")
                    ]
                    for title, val in specs:
                        with ui.row().classes("w-full justify-between text-sm py-1 border-b border-slate-200 dark:border-slate-800 last:border-0"):
                            ui.label(title).classes("text-slate-500 dark:text-slate-400")
                            ui.label(val).classes("font-semibold text-right")


# ============================================================================
# Page 2: Image Classifier (The Main Page with Zero-Stale State)
# ============================================================================
@ui.page("/classifier")
def page_classifier():
    render_header("Classifier")

    predictor = get_predictor()

    with ui.column().classes("w-full max-w-5xl mx-auto px-4 py-8 gap-6"):
        with ui.column().classes("gap-1"):
            ui.label("AI Image Classifier").classes("text-2xl font-black tracking-tight")
            ui.label("Upload an image to identify whether it contains a Cat or a Dog with calibrated probability scores.").classes("text-sm text-slate-500 dark:text-slate-400")

        # Main Workspace Container: 2 columns (Left: Upload & Preview, Right: Inference & Prediction)
        with ui.grid(columns=2).classes("w-full gap-6 items-start"):

            # ---------------- LEFT COLUMN: UPLOAD & PREVIEW ----------------
            with ui.card().classes("pv-card gap-4 w-full"):
                ui.label("1. Upload Image").classes("text-base font-bold text-slate-700 dark:text-slate-200")

                # State Preview Containers
                preview_container = ui.column().classes("w-full items-center justify-center")
                metadata_container = ui.column().classes("w-full gap-1")
                action_buttons_container = ui.row().classes("w-full gap-3 mt-2")

                async def handle_upload(e):
                    """Event handler when file is dropped or selected."""
                    try:
                        if hasattr(e, "file"):
                            name = e.file.name
                            file_bytes = await e.file.read()
                        elif hasattr(e, "content"):
                            name = getattr(e, "name", "uploaded_image.jpg")
                            file_bytes = e.content.read()
                        else:
                            raise ValueError("Unrecognized upload event structure.")

                        state.set_uploaded_image(name, file_bytes)
                        ui.notify(f"Loaded: {name} ({state.file_size_kb:.1f} KB)", type="positive", position="top")
                        update_ui_state()
                    except Exception as err:
                        ui.notify(f"Upload error: {err}", type="negative", position="top")

                # NiceGUI Uploader
                uploader = ui.upload(
                    on_upload=handle_upload,
                    auto_upload=True,
                    max_file_size=MAX_IMAGE_FILE_SIZE_MB * 1024 * 1024
                ).props('accept=".jpg,.jpeg,.png,.webp" flat bordered').classes("w-full")

                with ui.row().classes("w-full justify-between items-center text-xs text-slate-400"):
                    ui.label("Supported: JPG, JPEG, PNG, WEBP")
                    ui.label(f"Max size: {MAX_IMAGE_FILE_SIZE_MB} MB")

            # ---------------- RIGHT COLUMN: PREDICTION RESULTS ----------------
            with ui.card().classes("pv-card gap-4 w-full"):
                ui.label("2. Classification Results").classes("text-base font-bold text-slate-700 dark:text-slate-200")
                result_container = ui.column().classes("w-full")

        # ---------------- BOTTOM LIMITATION WARNING ----------------
        with ui.card().classes("w-full pv-card-elevated border-amber-500/30 bg-amber-500/5 p-4"):
            with ui.row().classes("items-start gap-3"):
                ui.icon("info", size="1.5rem").classes("text-amber-500 mt-0.5")
                with ui.column().classes("gap-1"):
                    ui.label("Important Model Scope & Limitation:").classes("text-sm font-bold text-amber-600 dark:text-amber-400")
                    ui.label("This neural network is trained specifically as a binary classifier for Cats vs Dogs. It is not an open-domain object detector; non-cat/dog images (e.g. cars, humans, landscapes) will still be mapped to Cat or Dog probabilities. Please inspect confidence scores and uncertainty warnings.").classes("text-xs text-slate-600 dark:text-slate-300")

    # ---------------- UI REFRESH AND ACTION LOGIC ----------------
    def execute_classification():
        """Runs model inference on the uploaded bytes."""
        if not state.image_bytes:
            ui.notify("Please upload an image first.", type="warning", position="top")
            return

        state.is_classifying = True
        update_ui_state()

        # Run inference through predictor
        result = predictor.predict(state.image_bytes)
        state.prediction_result = result
        state.is_classifying = False

        if result["success"]:
            ui.notify(f"Inference complete: {result['predicted_class']}", type="positive", position="top")
        else:
            ui.notify(f"Inference error: {result['error']}", type="negative", position="top")

        update_ui_state()

    def execute_clear():
        """Atomic reset of uploaded image and prediction cards."""
        state.clear()
        uploader.reset()
        update_ui_state()
        ui.notify("Classifier reset.", type="info", position="top")

    def update_ui_state():
        """Refreshes the dynamic preview and results DOM without full-page reloads."""
        # 1. Update Preview Column
        preview_container.clear()
        metadata_container.clear()
        action_buttons_container.clear()

        if state.b64_preview:
            with preview_container:
                ui.image(state.b64_preview).classes("w-full max-h-72 object-contain rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm")

            with metadata_container:
                with ui.row().classes("w-full justify-between text-xs text-slate-500 dark:text-slate-400 py-1"):
                    ui.label(f"File: {state.filename}")
                    if state.dimensions:
                        ui.label(f"Dimensions: {state.dimensions[0]} × {state.dimensions[1]} px")
                with ui.row().classes("w-full justify-between text-xs text-slate-400"):
                    ui.label(f"Size: {state.file_size_kb:.1f} KB")
                    ui.label(f"Hash: {state.image_hash}")

            with action_buttons_container:
                ui.button("Classify Image", icon="play_arrow", on_click=execute_classification).props("unelevated").classes("bg-blue-600 hover:bg-blue-700 text-white font-bold flex-1 py-2 rounded-xl")
                ui.button("Clear", icon="delete_outline", on_click=execute_clear).props("outline").classes("border-rose-400 text-rose-500 font-semibold px-4 rounded-xl")
        else:
            with preview_container:
                with ui.column().classes("h-48 w-full border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-xl items-center justify-center text-slate-400 gap-2"):
                    ui.icon("image", size="3rem")
                    ui.label("No image loaded yet").classes("text-sm")

        # 2. Update Result Column
        result_container.clear()
        with result_container:
            if state.is_classifying:
                with ui.column().classes("w-full h-64 items-center justify-center gap-3"):
                    ui.spinner("dots", size="3rem").classes("text-blue-500")
                    ui.label("Running neural network inference...").classes("text-sm text-slate-500 font-medium")
            elif state.prediction_result and state.prediction_result.get("success"):
                res = state.prediction_result
                pred_class = res["predicted_class"]
                is_cat = pred_class == "Cat"

                badge_class = "pv-badge-cat" if is_cat else "pv-badge-dog"
                badge_icon = "pets" if is_cat else "cruelty_free"

                # Prediction Header Card
                with ui.card().classes(f"w-full pv-card-elevated border-2 {'border-sky-500/50' if is_cat else 'border-amber-500/50'} p-5 gap-3"):
                    with ui.row().classes("w-full justify-between items-center"):
                        ui.label("PREDICTED CLASS").classes("text-xs font-bold tracking-wider text-slate-400")
                        with ui.row().classes(f"pv-badge {badge_class}"):
                            ui.icon(badge_icon, size="1rem")
                            ui.label(pred_class.upper()).classes("font-black text-sm")

                    ui.label(f"{pred_class}").classes(f"text-4xl font-black {'text-sky-600 dark:text-sky-400' if is_cat else 'text-amber-600 dark:text-amber-400'}")
                    ui.label(f"Overall Model Confidence: {res['confidence_percentage']}%").classes("text-sm font-semibold text-slate-600 dark:text-slate-300")

                # Calibrated Probabilities Card
                with ui.card().classes("w-full pv-card p-4 gap-4 mt-2"):
                    ui.label("Estimated Class Probabilities").classes("text-xs font-bold text-slate-400 uppercase tracking-wider")

                    # Cat Probability Bar
                    with ui.column().classes("w-full gap-1"):
                        with ui.row().classes("w-full justify-between text-sm font-semibold"):
                            ui.label("Cat Probability").classes("text-sky-600 dark:text-sky-400 flex items-center gap-1.5")
                            ui.label(f"{res['cat_percentage']}%")
                        ui.linear_progress(value=res['cat_probability'], size="10px").props("rounded").classes("text-sky-500")

                    # Dog Probability Bar
                    with ui.column().classes("w-full gap-1 mt-2"):
                        with ui.row().classes("w-full justify-between text-sm font-semibold"):
                            ui.label("Dog Probability").classes("text-amber-600 dark:text-amber-400 flex items-center gap-1.5")
                            ui.label(f"{res['dog_percentage']}%")
                        ui.linear_progress(value=res['dog_probability'], size="10px").props("rounded").classes("text-amber-500")

                # Uncertainty Warning Card (if top score < 60%)
                if res.get("is_uncertain"):
                    with ui.card().classes("w-full bg-rose-500/10 border border-rose-500/40 p-3 mt-2 rounded-xl"):
                        with ui.row().classes("items-center gap-2 text-rose-500"):
                            ui.icon("warning", size="1.25rem")
                            ui.label("Uncertainty Warning").classes("font-bold text-xs uppercase")
                        ui.label(res["uncertainty_warning"]).classes("text-xs text-rose-400 font-medium")

            elif state.prediction_result and not state.prediction_result.get("success"):
                with ui.card().classes("w-full bg-rose-500/10 border border-rose-500/30 p-4"):
                    ui.label("Classification Failed").classes("font-bold text-rose-500")
                    ui.label(state.prediction_result.get("error", "Unknown error")).classes("text-sm text-slate-600 dark:text-slate-300")
            else:
                with ui.column().classes("h-64 w-full border border-slate-200 dark:border-slate-800 rounded-xl items-center justify-center text-slate-400 gap-2 p-6 text-center"):
                    ui.icon("psychology", size="3rem").classes("text-slate-300 dark:text-slate-600")
                    ui.label("Awaiting Image Classification").classes("font-semibold text-slate-600 dark:text-slate-300 text-sm")
                    ui.label("Upload an image on the left and click 'Classify Image' to run inference.").classes("text-xs max-w-xs text-slate-400")

    # Initial state paint
    update_ui_state()


# ============================================================================
# Page 3: Model Insights
# ============================================================================
@ui.page("/insights")
def page_insights():
    render_header("Model Insights")

    metrics = load_saved_metrics()

    # Load training history if available
    history = {}
    if HISTORY_JSON_PATH.exists():
        try:
            with open(HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}

    with ui.column().classes("w-full max-w-6xl mx-auto px-4 py-8 gap-8"):
        with ui.column().classes("gap-1"):
            ui.label("Model Architecture & Performance Insights").classes("text-2xl font-black tracking-tight")
            ui.label("Genuine evaluation metrics computed on the untouched 1,000-image held-out test set.").classes("text-sm text-slate-500 dark:text-slate-400")

        if not metrics:
            with ui.card().classes("w-full pv-card p-12 text-center items-center justify-center gap-4"):
                ui.icon("analytics", size="4rem").classes("text-slate-400")
                ui.label("No Model Evaluation Metrics Found").classes("text-xl font-bold")
                ui.label("The model has not been trained or evaluated yet. Run train.py or train_model.bat to train MobileNetV2 and generate genuine performance curves and metrics.").classes("text-sm text-slate-500 max-w-md")
                ui.button("Train Model Now", icon="play_arrow", on_click=lambda: ui.notify("Please run train_model.bat from terminal", type="info")).props("unelevated").classes("bg-blue-600 text-white font-bold")
            return

        # Metrics Overview Cards
        with ui.grid(columns=5).classes("w-full gap-4"):
            cards_data = [
                ("Accuracy", f"{metrics['accuracy']*100:.2f}%", "check_circle", "text-blue-500"),
                ("Precision", f"{metrics['precision']*100:.2f}%", "gps_fixed", "text-emerald-500"),
                ("Recall", f"{metrics['recall']*100:.2f}%", "history", "text-amber-500"),
                ("F1-Score", f"{metrics['f1_score']*100:.2f}%", "auto_graph", "text-indigo-500"),
                ("ROC-AUC", f"{metrics['roc_auc']:.4f}", "trending_up", "text-cyan-500"),
            ]
            for title, val, icon_name, color in cards_data:
                with ui.card().classes("pv-metric-card"):
                    with ui.row().classes("w-full justify-between items-center"):
                        ui.label(title).classes("pv-metric-label")
                        ui.icon(icon_name, size="1.25rem").classes(color)
                    ui.label(val).classes(f"pv-metric-value {color}")
                    ui.label("Test Set").classes("text-xs text-slate-400")

        # Interactive Charts Row: Training Curves & Confusion Matrix
        with ui.grid(columns=2).classes("w-full gap-6"):
            # Chart 1: Training & Validation Curves
            with ui.card().classes("pv-card gap-2"):
                ui.label("Training & Validation Accuracy Curves").classes("text-base font-bold")

                if history and "accuracy" in history and "val_accuracy" in history:
                    epochs = list(range(1, len(history["accuracy"]) + 1))
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(x=epochs, y=history["accuracy"], mode="lines+markers", name="Train Accuracy", line=dict(color="#2563EB", width=2)))
                    fig_acc.add_trace(go.Scatter(x=epochs, y=history["val_accuracy"], mode="lines+markers", name="Val Accuracy", line=dict(color="#10B981", width=2)))
                    fig_acc.update_layout(
                        margin=dict(l=40, r=20, t=20, b=40),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Epoch", gridcolor="rgba(128,128,128,0.2)"),
                        yaxis=dict(title="Accuracy", gridcolor="rgba(128,128,128,0.2)"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    ui.plotly(fig_acc).classes("w-full h-72")
                else:
                    ui.label("History curve data not available.").classes("text-sm text-slate-400 py-12 text-center")

            # Chart 2: Confusion Matrix Heatmap
            with ui.card().classes("pv-card gap-2"):
                ui.label("Test Set Confusion Matrix").classes("text-base font-bold")

                cm = metrics.get("confusion_matrix", {}).get("matrix")
                if cm:
                    z = cm
                    x_labels = ["Pred: Cat", "Pred: Dog"]
                    y_labels = ["True: Cat", "True: Dog"]
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=z,
                        x=x_labels,
                        y=y_labels,
                        colorscale=[[0, "#EEF2FF"], [1, "#1D4ED8"]],
                        text=z,
                        texttemplate="%{text}",
                        textfont={"size": 16, "color": "white"}
                    ))
                    fig_cm.update_layout(
                        margin=dict(l=40, r=20, t=20, b=40),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed")
                    )
                    ui.plotly(fig_cm).classes("w-full h-72")
                else:
                    ui.label("Confusion matrix data not available.").classes("text-sm text-slate-400 py-12 text-center")

        # Per-Class Metrics Table
        with ui.card().classes("w-full pv-card gap-4"):
            ui.label("Per-Class Classification Report").classes("text-base font-bold")
            per_class = metrics.get("per_class", {})

            columns = [
                {"name": "class", "label": "Class", "field": "class", "align": "left"},
                {"name": "precision", "label": "Precision", "field": "precision", "align": "center"},
                {"name": "recall", "label": "Recall", "field": "recall", "align": "center"},
                {"name": "f1", "label": "F1-Score", "field": "f1", "align": "center"},
                {"name": "support", "label": "Test Support", "field": "support", "align": "center"},
            ]
            rows = []
            for cls_name in ["Cat", "Dog"]:
                c_data = per_class.get(cls_name, {})
                rows.append({
                    "class": f"{'🐱' if cls_name=='Cat' else '🐶'} {cls_name}",
                    "precision": f"{c_data.get('precision', 0.0)*100:.2f}%",
                    "recall": f"{c_data.get('recall', 0.0)*100:.2f}%",
                    "f1": f"{c_data.get('f1_score', 0.0)*100:.2f}%",
                    "support": f"{c_data.get('support', 0)} samples",
                })
            ui.table(columns=columns, rows=rows, row_key="class").classes("w-full")


# ============================================================================
# Page 4: Dataset & About
# ============================================================================
@ui.page("/about")
def page_about():
    render_header("Dataset & About")

    ready, counts = is_dataset_ready()

    with ui.column().classes("w-full max-w-5xl mx-auto px-4 py-8 gap-8"):
        with ui.column().classes("gap-1"):
            ui.label("Dataset & System Architecture").classes("text-2xl font-black tracking-tight")
            ui.label("Complete methodology, training pipeline, and architectural specifications.").classes("text-sm text-slate-500 dark:text-slate-400")

        # Section 1: Dataset Information
        with ui.card().classes("w-full pv-card gap-4"):
            ui.label("1. Dataset & Source").classes("text-lg font-bold")
            ui.label(
                "This application utilizes the official TensorFlow Cats vs Dogs filtered dataset "
                "demonstrated in the official computer vision transfer learning tutorial. "
                "The archive contains authentic, high-diversity photographs of domestic cats and dogs."
            ).classes("text-sm text-slate-600 dark:text-slate-300 leading-relaxed")

            with ui.grid(columns=3).classes("w-full gap-4 mt-2"):
                with ui.card().classes("pv-card-elevated p-4"):
                    ui.label("Training Subset (80%)").classes("text-xs font-bold text-slate-400 uppercase")
                    ui.label(f"{counts.get('train_cats', 800) + counts.get('train_dogs', 800)} Images").classes("text-xl font-extrabold text-blue-600 dark:text-blue-400")
                    ui.label("800 Cats / 800 Dogs").classes("text-xs text-slate-500")

                with ui.card().classes("pv-card-elevated p-4"):
                    ui.label("Validation Subset (20%)").classes("text-xs font-bold text-slate-400 uppercase")
                    ui.label(f"{counts.get('val_cats', 200) + counts.get('val_dogs', 200)} Images").classes("text-xl font-extrabold text-emerald-600 dark:text-emerald-400")
                    ui.label("200 Cats / 200 Dogs").classes("text-xs text-slate-500")

                with ui.card().classes("pv-card-elevated p-4"):
                    ui.label("Held-Out Test Set").classes("text-xs font-bold text-slate-400 uppercase")
                    ui.label(f"{counts.get('val_total', 1000)} Images").classes("text-xl font-extrabold text-amber-600 dark:text-amber-400")
                    ui.label("500 Cats / 500 Dogs (Untouched)").classes("text-xs text-slate-500")

        # Section 2: Image Preprocessing & Data Augmentation
        with ui.card().classes("w-full pv-card gap-4"):
            ui.label("2. Image Preprocessing & Pipeline").classes("text-lg font-bold")
            ui.markdown("""
- **Aspect-Ratio Preserving Resize:** Images are dynamically scaled and padded with a neutral border to 160 × 160 pixels, preventing physical distortion of pet body proportions.
- **Color Mode Normalization:** All inputs (including RGBA transparent PNGs, CMYK, and single-channel Grayscale) are safely converted to 3-channel RGB.
- **EXIF Orientation Correction:** Image metadata is inspected and automatically rotated according to camera orientation tags (`exif_transpose`).
- **MobileNetV2 Normalization:** Pixel values in $[0, 255]$ are scaled to $[-1.0, 1.0]$ matching ImageNet pretraining.
- **Training Data Augmentation:** Realistic, training-only augmentation is applied: random horizontal flip, small rotation (±10%), zoom (±10%), and contrast adjustment.
            """).classes("text-sm text-slate-600 dark:text-slate-300 leading-relaxed")

        # Section 3: Model Architecture & Transfer Learning
        with ui.card().classes("w-full pv-card gap-4"):
            ui.label("3. Transfer Learning Architecture").classes("text-lg font-bold")
            ui.markdown("""
- **Pretrained Backbone:** MobileNetV2 pretrained on ImageNet (1.4 million images across 1,000 classes), providing rich low-level visual features (edges, textures, shapes).
- **Classification Head:** Global Average Pooling 2D layer compresses 2D spatial feature maps into a 1280-element feature vector, followed by a 20% Dropout regularization layer, and a single Dense unit with Sigmoid activation.
- **Two-Phase Training:**
  - *Phase 1:* Base MobileNetV2 is frozen; only the top classification head is trained with Adam optimizer ($\text{lr} = 5 \times 10^{-4}$).
  - *Phase 2 (Fine-Tuning):* Upper layers (layer 100+) are unfrozen and trained at a fine learning rate ($\text{lr} = 5 \times 10^{-5}$) with EarlyStopping.
            """).classes("text-sm text-slate-600 dark:text-slate-300 leading-relaxed")

        # Section 4: Model Limitations & Safety
        with ui.card().classes("w-full pv-card gap-4 border-amber-500/30"):
            ui.label("4. Limitations & Scope of Model").classes("text-lg font-bold text-amber-600 dark:text-amber-400")
            ui.markdown("""
- **Binary Closed-Set Assumption:** The model is strictly trained to differentiate **Cats vs Dogs**. It is not a universal object detector. When presented with images of cars, humans, or birds, the model will output a probability split between Cat and Dog.
- **Uncertainty Calibration:** Predictions where the maximum class probability is below **60%** are flagged with an explicit *Low Confidence Uncertainty Warning*.
- **CPU Optimization:** Designed for responsive CPU execution on standard hardware (Intel i5, 8GB RAM).
            """).classes("text-sm text-slate-600 dark:text-slate-300 leading-relaxed")


# Entrypoint
if __name__ in {"__main__", "__mp_main__"}:
    print(f"[*] Starting PawVision AI Web Application on http://{SERVER_HOST}:{SERVER_PORT}")
    ui.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        title="PawVision AI — Cat vs Dog Image Classifier",
        favicon="🐾",
        reload=False,
        storage_secret="pawvision-ai-theme-storage-key-2024"
    )
