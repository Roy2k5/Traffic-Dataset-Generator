import sys
import argparse
import os

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QSurfaceFormat, QOffscreenSurface, QOpenGLContext
from PyQt6.QtCore import QCoreApplication

def run_gui():
    from ui.main_window import MainWindow
    
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

def run_headless(args):
    print(f"Running headless mode: {args.num_frames} frames, {args.width}x{args.height}")
    
    app = QApplication(sys.argv)
    
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    
    context = QOpenGLContext()
    context.setFormat(fmt)
    if not context.create():
        print("Failed to create OpenGL context")
        sys.exit(1)
        
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    
    if not context.makeCurrent(surface):
        print("Failed to make context current")
        sys.exit(1)
        
    from render.renderer import Renderer
    from scene.scene_manager import SceneManager
    from exporter.exporter import Exporter
    from libs.camera import Camera
    import random
    
    renderer = Renderer(args.width, args.height)
    renderer.init_gl()
    
    scene_manager = SceneManager()
    scene_manager.load_models()
    
    exporter = Exporter(args.output_dir)
    
    camera = Camera(distance=20.0, pitch=45.0, yaw=45.0)
    
    for i in range(args.num_frames):
        print(f"Generating frame {i+1}/{args.num_frames}...")
        scene_objects = scene_manager.generate_scene(seed=args.seed + i if args.seed else None)
        
        # Randomize camera slightly
        camera = Camera(
            distance=random.uniform(15.0, 25.0),
            pitch=random.uniform(20.0, 60.0),
            yaw=random.uniform(0.0, 360.0)
        )
        
        rgb = renderer.render(scene_objects, camera, pass_type='rgb', offscreen=True)
        depth = renderer.render(scene_objects, camera, pass_type='depth', offscreen=True)
        mask = renderer.render(scene_objects, camera, pass_type='mask', offscreen=True)
        
        # In depth map, we only rendered one channel (grayscale) but output format is RGB in FBO.
        exporter.export(i, rgb, depth, mask, scene_objects)
        
    exporter.save_coco()
    print(f"Batch generated successfully in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Scene Generation")
    parser.add_argument('--headless', action='store_true', help="Run without UI")
    parser.add_argument('--num-frames', type=int, default=10, help="Number of frames to generate in headless mode")
    parser.add_argument('--width', type=int, default=1024, help="Image width")
    parser.add_argument('--height', type=int, default=768, help="Image height")
    parser.add_argument('--output-dir', type=str, default='output', help="Output directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.headless:
        run_headless(args)
    else:
        run_gui()
