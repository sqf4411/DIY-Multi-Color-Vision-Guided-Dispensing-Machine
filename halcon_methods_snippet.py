    # ================== Halcon Integration ==================
    def on_halcon_toggled(self):
        enabled = self.chk_use_halcon.isChecked()
        self.btn_load_halcon.setEnabled(enabled)
        if enabled and not self.halcon_manager.is_initialized:
            # Try init
            if not self.halcon_manager.init_engine():
                QMessageBox.warning(self, "Error", "Failed to initialize Halcon Engine.\nEnsure mvtec-halcon is installed.")
                self.chk_use_halcon.setChecked(False)
                self.btn_load_halcon.setEnabled(False)

    def on_load_halcon_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Halcon Script", "config", "Halcon Scripts (*.hdev *.shm)")
        if path:
            if self.halcon_manager.load_script(path):
                self.log(f"✅ Halcon script loaded: {os.path.basename(path)}")
            else:
                self.log(f"❌ Failed to load Halcon script: {path}")

    def get_vision_result(self, gray_img):
        """
        Unified vision interface.
        Returns: (paths_pixels, debug_img_bgr)
        """
        if self.chk_use_halcon.isChecked():
            # Use Halcon
            res = self.halcon_manager.process_image(gray_img)
            if res and "paths" in res:
                # Assuming res['paths'] is list of numpy arrays
                return res['paths'], cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR) # We might not get debug img from Halcon yet
            else:
                 return [], cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        else:
            # Use Legacy OpenCV
            from core.mold_detector import get_paths_and_image
            return get_paths_and_image(gray_img)
