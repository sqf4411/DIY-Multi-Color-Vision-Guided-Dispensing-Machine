
import os

file_path = "main_camera_ui.py"
try:
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "on_btn_laser_measure" in line:
                print(f"LASER_MEASURE_FOUND: Line {i+1}: {line.strip()}")
            if "相机空跑" in line:
                print(f"DRY_RUN_BTN_FOUND: Line {i+1}: {line.strip()}")
            if "DryRunDialog" in line and "import" not in line:
                # Print dry run dialog usage
                print(f"DRY_RUN_DIALOG_USAGE: Line {i+1}: {line.strip()}")

except Exception as e:
    print(f"Error: {e}")
