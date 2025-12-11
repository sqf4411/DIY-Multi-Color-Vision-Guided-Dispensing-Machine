from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSlot
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np

class Visualizer3D(QWidget):
    """
    3D Visualizer widget for displaying machine needle position including offset.
    Uses pyqtgraph's OpenGL module for lightweight rendering.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create OpenGL View
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 200  # Camera distance
        self.view.opts['elevation'] = 30  # Camera elevation angle
        self.view.opts['azimuth'] = 45    # Camera azimuth angle
        self.layout.addWidget(self.view)
        
        # Initialize Scene Items
        self._init_scene()
        
    def _init_scene(self):
        """Initialize grid, axes, and needle cursor."""
        # 1. Add Grid (Floor)
        # Assuming typical work area ~300x300mm
        gx = gl.GLGridItem()
        gx.setSize(x=300, y=300, z=0)
        gx.setSpacing(x=10, y=10, z=10) # 10mm grid
        gx.translate(150, 150, 0) # Move to positive quadrant mostly if origin is 0,0
        # Actually usually CNC origin is 0,0. Let's center grid at 150,150 to cover 0-300 range approx
        self.view.addItem(gx)
        
        # 2. Add Axes
        axis = gl.GLAxisItem()
        axis.setSize(x=50, y=50, z=50)
        self.view.addItem(axis)
        
        # 3. Add Needle Cursor (Represented as a cone or sphere)
        # Using a simple GLScatterPlotItem for the tip (point)
        # and a GLMeshItem (Cylinder/Cone) for the body if needed.
        # Let's keep it simple: A red sphere for the tip.
        
        md = gl.MeshData.sphere(rows=10, cols=20, radius=2)
        self.needle_tip = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(1, 0, 0, 1), # Red
            shader='shaded',
            glOptions='opaque'
        )
        self.view.addItem(self.needle_tip)
        
        # Add a line indicating the tool length/vertical drop (optional)
        # For now just the tip is enough.
        
        # Store current position
        self.current_pos = [0.0, 0.0, 0.0]

    @pyqtSlot(float, float, float)
    def update_position(self, x, y, z):
        """
        Update the position of the needle cursor.
        
        Args:
            x (float): Machine X coordinate
            y (float): Machine Y coordinate
            z (float): Machine Z coordinate
        """
        self.current_pos = [x, y, z]
        
        # Translate the mesh
        # GLMeshItem does not have setPos directly like GraphicsItem, 
        # it uses transform matrix or translate method which accumulates.
        # So we need to reset transform and translate to new pos.
        
        self.needle_tip.resetTransform()
        self.needle_tip.translate(x, y, z)
        
        # Optional: Camera follow?
        # self.view.setCameraPosition(pos=pg.Vector(x,y,z))

    def set_work_envelope(self, width, height, z_height):
        """Update grid size to match machine dimensions."""
        pass 
