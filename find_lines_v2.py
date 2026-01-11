
file_path = "main_camera_ui.py"
with open("lines_found.txt", "w", encoding="utf-8") as out:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if "on_btn_laser_measure" in line:
                    out.write(f"LASER_MEASURE_FOUND: Line {i+1}: {line.strip()}\n")
                if "相机空跑" in line:
                    out.write(f"DRY_RUN_BTN_FOUND: Line {i+1}: {line.strip()}\n")
                if "def on_real_run" in line:
                    out.write(f"REAL_RUN_DEF: Line {i+1}: {line.strip()}\n")
                if "def on_dry_run" in line:
                    out.write(f"DRY_RUN_DEF: Line {i+1}: {line.strip()}\n")
    except Exception as e:
        out.write(f"Error: {e}\n")
