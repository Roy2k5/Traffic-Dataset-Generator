from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QSpinBox, QGroupBox, QComboBox,
    QCheckBox, QTextEdit, QColorDialog, QSizePolicy, QLineEdit, QFileDialog
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QFont, QPen
from libs.camera import Camera
from render.renderer import Renderer
from scene.scene_manager import SceneManager
from exporter.exporter import Exporter
import time, os, glob

# ── Render view modes ─────────────────────────────────────────────────────────
VIEW_NORMAL  = "Normal"
VIEW_DEPTH   = "Depth"
VIEW_SEG     = "Segmentation"
VIEW_DETECT  = "Object Detection"
VIEW_MODEL   = "Model"
VIEW_MODES   = [VIEW_NORMAL, VIEW_DEPTH, VIEW_SEG, VIEW_DETECT, VIEW_MODEL]

# ── Generation modes ──────────────────────────────────────────────────────────
MODE_TIMED  = "Timed (2s interval)"
MODE_RANDOM = "Random Position"

# ── Label formats ─────────────────────────────────────────────────────────────
FMT_COCO  = "COCO"
FMT_YOLO  = "YOLO"
FMT_BOTH  = "Both"


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.camera       = Camera(distance=35.0, pitch=35.0, yaw=45.0)
        self.renderer     = None
        self.scene_manager = SceneManager()
        self.last_pos     = None
        self._last_time   = None

        # animation timer (~60 fps)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_timer.start(16)

        # timed export timer
        self._export_timer = QTimer(self)
        self._export_timer.timeout.connect(self._on_timed_export)

        # export settings (set by MainWindow)
        self.gen_mode    = MODE_TIMED
        self.label_fmt   = FMT_BOTH
        self.output_dir  = "output"
        self.exporter    = None
        self._frame_idx  = 0
        self._exporting  = False
        self.log_fn      = None
        # render view mode
        self.render_mode    = VIEW_NORMAL
        self.post_processor = None   # models.PostProcessor instance
        self._annotated_img = None   # latest annotated numpy image
        self._infer_running = False  # guard: only one thread at a time

    # ── GL lifecycle ──────────────────────────────────────────────────────────
    def initializeGL(self):
        self.renderer = Renderer(self.width(), self.height())
        self.renderer.init_gl()
        self.scene_manager.load_models()
        self.scene_manager.generate_scene(seed=42)
        self._last_time = time.perf_counter()

    def resizeGL(self, w, h):
        if self.renderer:
            r = self.devicePixelRatio()
            self.renderer.resize(int(w * r), int(h * r))

    def paintGL(self):
        if not self.renderer:
            return
        r  = self.devicePixelRatio()
        pw = int(self.width()  * r)
        ph = int(self.height() * r)

        pass_map = {
            VIEW_NORMAL: 'rgb',
            VIEW_DEPTH:  'depth',
            VIEW_SEG:    'mask',
            VIEW_DETECT: 'rgb',
            VIEW_MODEL:  'rgb',
        }
        self.renderer.render(
            self.scene_manager.objects, self.camera,
            pass_type=pass_map.get(self.render_mode, 'rgb'),
            offscreen=False,
            default_fbo=self.defaultFramebufferObject(),
            vp_width=pw, vp_height=ph
        )
        # For MODEL/DETECT modes, overlay the post-processed annotated image via QPainter
        if self.render_mode in (VIEW_DETECT, VIEW_MODEL) and self._annotated_img is not None:
            from PyQt6.QtGui import QImage, QPixmap
            img = self._annotated_img          # grab reference (atomic under GIL)
            h, w = img.shape[:2]
            # tobytes() copies the buffer → safe even if inference thread replaces _annotated_img
            raw = img.astype('uint8').tobytes()
            qi = QImage(raw, w, h, 3*w, QImage.Format.Format_RGB888)
            p = QPainter(self)
            p.drawPixmap(0, 0, self.width(), self.height(), QPixmap.fromImage(qi))
            p.end()

    # ── Animation tick ────────────────────────────────────────────────────────
    def _on_anim_tick(self):
        now = time.perf_counter()
        dt  = now - self._last_time if self._last_time else 0.016
        dt  = min(dt, 0.05)
        self._last_time = now
        self.scene_manager.update(dt)
        self.update()
        # Real-time inference: kick off in background if model is active
        if (self.render_mode in (VIEW_DETECT, VIEW_MODEL)
                and self.post_processor
                and len(self.post_processor) > 0
                and not self._infer_running):
            self._start_infer_thread()

    def _start_infer_thread(self):
        """Render current frame offscreen and run PostProcessor in a thread."""
        import threading, numpy as np
        self._infer_running = True
        # Capture RGB synchronously (must be on GL thread)
        self.makeCurrent()
        rgb = self.renderer.render(
            self.scene_manager.objects, self.camera,
            'rgb', offscreen=True)
        self.doneCurrent()
        if rgb is None:
            self._infer_running = False
            return
        # Copy so the thread has its own buffer
        frame = rgb.copy()
        pp    = self.post_processor

        def _worker():
            try:
                annotated = pp.run(frame)
                self._annotated_img = annotated
            except Exception as e:
                print(f"[Infer] error: {e}")
            finally:
                self._infer_running = False

        threading.Thread(target=_worker, daemon=True).start()

    # ── Timed export ──────────────────────────────────────────────────────────
    def start_export(self, num_frames, output_dir, gen_mode, label_fmt, bg_color):
        self._exporting  = True
        self._frame_idx  = 0
        self._num_frames = num_frames
        self.output_dir  = output_dir
        self.gen_mode    = gen_mode
        self.label_fmt   = label_fmt
        self.renderer.bg_color = (bg_color.redF(), bg_color.greenF(), bg_color.blueF())
        os.makedirs(output_dir, exist_ok=True)
        self.exporter = Exporter(output_dir)
        if gen_mode == MODE_TIMED:
            self._export_timer.start(2000)   # every 2 s
        else:
            self._capture_frame()            # random mode: capture immediately then loop

    def stop_export(self):
        self._export_timer.stop()
        self._exporting = False
        if self.exporter:
            self.exporter.save_coco()
        self._log("Export finished.")

    def _on_timed_export(self):
        if self._frame_idx >= self._num_frames:
            self.stop_export()
            return
        self._capture_frame()

    def _capture_frame(self):
        if not self._exporting:
            return
        if self.gen_mode == MODE_RANDOM:
            self.scene_manager.generate_scene()   # re-randomise positions each frame

        self.makeCurrent()
        rgb   = self.renderer.render(self.scene_manager.objects, self.camera,
                                     'rgb',   offscreen=True)
        depth = self.renderer.render(self.scene_manager.objects, self.camera,
                                     'depth', offscreen=True)
        mask  = self.renderer.render(self.scene_manager.objects, self.camera,
                                     'mask',  offscreen=True)
        self.doneCurrent()

        self.exporter.export(self._frame_idx, rgb, depth, mask,
                             self.scene_manager.objects,
                             label_fmt=self.label_fmt)
        self._log(f"Frame {self._frame_idx+1}/{self._num_frames} captured.")
        self._frame_idx += 1

        if self._frame_idx >= self._num_frames:
            self.stop_export()
        elif self.gen_mode == MODE_RANDOM:
            # Schedule the next frame capture (singleShot lets Qt process events)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self._capture_frame)

    def _log(self, msg):
        if self.log_fn:
            self.log_fn(msg)

    # ── Mouse controls (Trackball) ────────────────────────────────────────────
    def mousePressEvent(self, event):
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        winsize = [self.width(), self.height()]
        old = [self.last_pos.x(), self.height() - self.last_pos.y()]
        new = [event.position().x(), self.height() - event.position().y()]
        buttons = event.buttons()
        if buttons & Qt.MouseButton.LeftButton:
            self.camera.drag(old, new, winsize)          # rotate (trackball)
        if buttons & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            self.camera.pan(old, new)                    # pan (translate)
        self.last_pos = event.position()
        self.update()

    def wheelEvent(self, event):
        self.camera.zoom(event.angleDelta().y() / 120.0,
                         max(self.width(), self.height()))
        self.update()

    def set_bg_color(self, qcolor):
        if self.renderer:
            self.renderer.bg_color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthetic Scene Generation")
        self.resize(1280, 768)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        # ── GL viewport ───────────────────────────────────────────────────
        self.gl = GLWidget()
        self.gl.log_fn = self._append_log

        # ── Config sidebar ────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background:#1e1e2e; color:#cdd6f4;")
        sv = QVBoxLayout(sidebar)
        sv.setContentsMargins(8,8,8,8)
        sv.setSpacing(6)

        def section(title):
            gb = QGroupBox(title)
            gb.setStyleSheet(
                "QGroupBox{color:#89b4fa;border:1px solid #45475a;"
                "border-radius:4px;margin-top:6px;padding-top:4px;}"
                "QGroupBox::title{subcontrol-origin:margin;left:6px;}"
            )
            return gb

        # Frames
        grp_frames = section("Frames")
        gl_f = QVBoxLayout(grp_frames)
        self.spin_frames = QSpinBox(); self.spin_frames.setRange(1,9999); self.spin_frames.setValue(20)
        gl_f.addWidget(QLabel("Number of frames:")); gl_f.addWidget(self.spin_frames)
        sv.addWidget(grp_frames)

        # Output dir
        grp_out = section("Output")
        gl_o = QVBoxLayout(grp_out)
        from PyQt6.QtWidgets import QLineEdit
        self.edit_out = QLineEdit("output"); gl_o.addWidget(QLabel("Output folder:")); gl_o.addWidget(self.edit_out)
        sv.addWidget(grp_out)

        # Generation mode
        grp_mode = section("Generation Mode")
        gl_m = QVBoxLayout(grp_mode)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([MODE_TIMED, MODE_RANDOM])
        gl_m.addWidget(self.combo_mode)
        sv.addWidget(grp_mode)

        # Label format
        grp_lbl = section("Label Format")
        gl_l = QVBoxLayout(grp_lbl)
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems([FMT_BOTH, FMT_COCO, FMT_YOLO])
        gl_l.addWidget(self.combo_fmt)
        sv.addWidget(grp_lbl)

        # Background colour
        grp_bg = section("Background Color")
        gl_bg = QVBoxLayout(grp_bg)
        self._bg_color = QColor(128, 178, 255)
        self.btn_bg = QPushButton("Pick Color")
        self.btn_bg.clicked.connect(self._pick_bg)
        self._update_bg_btn()
        gl_bg.addWidget(self.btn_bg)
        sv.addWidget(grp_bg)

        # Generate button
        self.btn_gen = QPushButton("▶  Generate Batch")
        self.btn_gen.setStyleSheet(
            "QPushButton{background:#89b4fa;color:#1e1e2e;border-radius:6px;"
            "padding:8px;font-weight:bold;}"
            "QPushButton:hover{background:#b4befe;}"
        )
        self.btn_gen.clicked.connect(self._on_generate)
        sv.addWidget(self.btn_gen)

        # Log area
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet("background:#181825;color:#a6e3a1;font-family:monospace;font-size:11px;")
        self.log.setFixedHeight(160)
        sv.addWidget(QLabel("Log:"))
        sv.addWidget(self.log)
        # ── View mode ─────────────────────────────────────────────────────
        grp_view = section("View Mode")
        gl_v = QVBoxLayout(grp_view)
        self.combo_view = QComboBox()
        self.combo_view.addItems(VIEW_MODES)
        self.combo_view.currentTextChanged.connect(self._on_view_mode_changed)
        gl_v.addWidget(self.combo_view)
        sv.addWidget(grp_view)

        # ── Model selector (shown only in Model mode) ─────────────────────
        self.grp_model = section("Detection Model")
        gl_mod = QVBoxLayout(self.grp_model)
        self.combo_model = QComboBox()
        self._refresh_model_list()
        gl_mod.addWidget(QLabel("Model file:"))
        gl_mod.addWidget(self.combo_model)
        btn_load_model = QPushButton("Load Model")
        btn_load_model.clicked.connect(self._on_load_model)
        gl_mod.addWidget(btn_load_model)
        self.grp_model.setVisible(False)
        sv.addWidget(self.grp_model)

        sv.addStretch()

        root.addWidget(sidebar)
        root.addWidget(self.gl, 1)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _pick_bg(self):
        c = QColorDialog.getColor(self._bg_color, self, "Background Color")
        if c.isValid():
            self._bg_color = c
            self._update_bg_btn()
            self.gl.set_bg_color(c)

    def _update_bg_btn(self):
        r,g,b = self._bg_color.red(), self._bg_color.green(), self._bg_color.blue()
        fg = "#000000" if (r+g+b) > 382 else "#ffffff"
        self.btn_bg.setStyleSheet(
            f"QPushButton{{background:rgb({r},{g},{b});color:{fg};"
            "border-radius:4px;padding:5px;}}"
        )

    def _on_generate(self):
        self.btn_gen.setEnabled(False)
        self.btn_gen.setText("Generating…")
        self.gl.start_export(
            num_frames  = self.spin_frames.value(),
            output_dir  = self.edit_out.text() or "output",
            gen_mode    = self.combo_mode.currentText(),
            label_fmt   = self.combo_fmt.currentText(),
            bg_color    = self._bg_color,
        )
        QTimer.singleShot(500, self._poll_export_done)

    def _poll_export_done(self):
        if not self.gl._exporting:
            self.btn_gen.setEnabled(True)
            self.btn_gen.setText("▶  Generate Batch")
        else:
            QTimer.singleShot(500, self._poll_export_done)

    def _append_log(self, msg):
        self.log.append(msg)

    # ── View mode ──────────────────────────────────────────────────────────────
    def _on_view_mode_changed(self, mode):
        self.gl.render_mode = mode
        self.grp_model.setVisible(mode == VIEW_MODEL)
        if mode in (VIEW_DETECT, VIEW_MODEL):
            self._run_postprocess()
        else:
            self.gl._annotated_img = None
        self.gl.update()

    def _run_postprocess(self):
        """
        Capture current RGB frame, pass through PostProcessor pipeline,
        store annotated result in gl._annotated_img for display.
        """
        if self.gl.post_processor is None or len(self.gl.post_processor) == 0:
            self.gl._annotated_img = None
            return
        self.gl.makeCurrent()
        rgb = self.gl.renderer.render(
            self.gl.scene_manager.objects, self.gl.camera,
            'rgb', offscreen=True)
        self.gl.doneCurrent()
        if rgb is not None:
            self.gl._annotated_img = self.gl.post_processor.run(rgb)

    # ── Model management ───────────────────────────────────────────────────────
    def _refresh_model_list(self):
        self.combo_model.clear()
        os.makedirs('models/weights', exist_ok=True)
        files = glob.glob('models/weights/*.pt') + glob.glob('models/weights/*.pth')
        if files:
            self.combo_model.addItems([os.path.basename(f) for f in files])
        else:
            self.combo_model.addItem('(no weights – place .pt files in models/weights/)')

    def _on_load_model(self):
        from models import PostProcessor, YOLODetectionPlugin
        sel = self.combo_model.currentText()
        weight_path = os.path.join('models', 'weights', sel)
        if not os.path.exists(weight_path):
            self._append_log(f"[Model] Not found: {weight_path}")
            return
        # Build a fresh PostProcessor with one YOLOv8 detection plugin
        pp = PostProcessor()
        plugin = YOLODetectionPlugin(weight_path, conf=0.25)
        pp.add(plugin)                         # loads weights here
        self.gl.post_processor = pp
        self._append_log(f"[Model] Loaded YOLOv8 detection: {sel}")
        self._run_postprocess()
        self.gl.update()


