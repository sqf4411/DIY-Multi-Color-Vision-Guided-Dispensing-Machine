from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSlot
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np

class Visualizer3D(QWidget):
    """
    3D Visualizer widget for displaying machine needle position including offset.
    Uses pyqtgraph's OpenGL module.
    """
    def __init__(self, parent=None, limits=(300, 300, 100)):
        super().__init__(parent)
        
        self.limits = limits # (x_max, y_max, z_max)
        
        # Setup layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create OpenGL View
        self.view = gl.GLViewWidget()
        # Better initial camera position
        self.view.opts['distance'] = 400
        self.view.opts['elevation'] = 40
        self.view.opts['azimuth'] = 225  # Look from corner
        self.view.setCameraPosition(pos=pg.Vector(limits[0]/2, limits[1]/2, 0))
        
        self.layout.addWidget(self.view)
        
        # Initialize Scene Items
        self._init_scene()
        
    def _init_scene(self):
        """Initialize grid, axes, and needle cursor."""
        x_max, y_max, z_max = self.limits

        # 1. Add Grid (Floor)
        # We create a grid that covers the work area
        gx = gl.GLGridItem()
        gx.setSize(x=x_max, y=y_max, z=0)
        gx.setSpacing(x=10, y=10, z=10) # 10mm grid lines
        # GLGridItem is centered at (0,0,0) by default with size. 
        # We want it from (0,0) to (x_max, y_max).
        gx.translate(x_max/2, y_max/2, 0)
        self.view.addItem(gx)
        
        # 2. Add Axes Lines & Ticks
        self._add_axes(x_max, y_max, z_max)
        
        # 3. Add Needle Cursor
        # Tip (Sphere)
        md = gl.MeshData.sphere(rows=10, cols=20, radius=3)
        self.needle_tip = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(1, 0, 0, 1), # Red
            shader='shaded',
            glOptions='opaque'
        )
        self.view.addItem(self.needle_tip)
        
        # Shadow Line (Vertical drop to floor)
        # Using a GLPlotItem (Line)
        self.shadow_line = gl.GLLinePlotItem(
            pos=np.array([[0,0,0], [0,0,0]]), 
            color=(1, 1, 0, 0.8), # Yellow semi-transparent
            width=2,
            antialias=True
        )
        self.view.addItem(self.shadow_line)
        
        # Needle Body (Cylinder representing the nozzle)
        # md_cyl = gl.MeshData.cylinder(rows=10, cols=20, radius=[1.0, 3.0], length=10.0) 
        # For simplicity, just the tip and line is clearer for now.

        # Store current position
        self.current_pos = [0.0, 0.0, 0.0]

    def _add_axes(self, x_max, y_max, z_max):
        """Draw custom axes with labels and ticks."""
        
        # --- Axis Lines ---
        pts_x = np.array([[0,0,0], [x_max,0,0]])
        pts_y = np.array([[0,0,0], [0,y_max,0]])
        pts_z = np.array([[0,0,0], [0,0,z_max]])
        
        line_x = gl.GLLinePlotItem(pos=pts_x, color=(1,0,0,1), width=3) # R
        line_y = gl.GLLinePlotItem(pos=pts_y, color=(0,1,0,1), width=3) # G
        line_z = gl.GLLinePlotItem(pos=pts_z, color=(0,0,1,1), width=3) # B
        
        self.view.addItem(line_x)
        self.view.addItem(line_y)
        self.view.addItem(line_z)
        
        # --- Axis Labels (Text) ---
        # GLTextItem pos is (x,y,z)
        lbl_x = gl.GLTextItem(pos=[x_max, 0, 10], text=f"X ({x_max}mm)", color=(1,0,0,1))
        lbl_y = gl.GLTextItem(pos=[0, y_max, 10], text=f"Y ({y_max}mm)", color=(0,1,0,1))
        lbl_z = gl.GLTextItem(pos=[0, 0, z_max+10], text=f"Z ({z_max}mm)", color=(0,0,1,1))
        
        self.view.addItem(lbl_x)
        self.view.addItem(lbl_y)
        self.view.addItem(lbl_z)
        
        # --- Ticks (Every 50mm) ---
        tick_len = 5
        
        # X Ticks
        for x in range(0, int(x_max)+1, 50):
            if x == 0: continue
            tick_pts = np.array([[x, 0, 0], [x, -tick_len, 0]]) # Tick mark
            ti = gl.GLLinePlotItem(pos=tick_pts, color=(1,1,1,0.5), width=1)
            self.view.addItem(ti)
            # Numeric Label (Optional, GLTextItem can be heavy if too many)
            # txt = gl.GLTextItem(pos=[x, -10, 0], text=str(x), color=(1,1,1,0.5))
            # self.view.addItem(txt)

        # Y Ticks
        for y in range(0, int(y_max)+1, 50):
            if y == 0: continue
            tick_pts = np.array([[0, y, 0], [-tick_len, y, 0]])
            ti = gl.GLLinePlotItem(pos=tick_pts, color=(1,1,1,0.5), width=1)
            self.view.addItem(ti)

        # Z Ticks
        for z in range(0, int(z_max)+1, 10):
            if z == 0: continue
            tick_pts = np.array([[0, 0, z], [tick_len, 0, z]])
            ti = gl.GLLinePlotItem(pos=tick_pts, color=(1,1,1,0.5), width=1)
            self.view.addItem(ti)

    @pyqtSlot(float, float, float)
    def update_position(self, x, y, z):
        """
        Update the position of the needle cursor.
        """
        self.current_pos = [x, y, z]
        
        # Move Tip
        self.needle_tip.resetTransform()
        self.needle_tip.translate(x, y, z)
        
        # Update Shadow Line (Tip to Floor)
        # Line from (x,y,z) to (x,y,0)
        pts = np.array([[x, y, z], [x, y, 0]])
        self.shadow_line.setData(pos=pts)
        
    def set_work_envelope(self, width, height, z_height):
        """Update grid size to match machine dimensions (Runtime update)."""
        # Re-init scene if dimensions change significantly
        # For now, just logging or ignoring as full re-init needs clearing items.
        pass 
