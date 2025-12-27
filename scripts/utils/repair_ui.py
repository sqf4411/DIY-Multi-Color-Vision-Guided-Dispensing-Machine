
import os

target_file = "d:\\CodeFromGithub\\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\\main_camera_ui.py"
snippet_file = "d:\\CodeFromGithub\\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\\halcon_methods_snippet.py"

# Read snippet
with open(snippet_file, 'r', encoding='utf-8') as f:
    snippet_content = f.read()

# Read target file safely
try:
    with open(target_file, 'rb') as f:
        content_bytes = f.read()
    
    # Attempt to decode as utf-8, ignoring errors to discard corruption
    content_str = content_bytes.decode('utf-8', errors='ignore')
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Find where to cut the corruption (the appended part)
# The append was `type ... >> ...`. 
# We look for the start of the snippet content "def on_halcon_toggled" to find where it was attached.
# Or better, we look for the end of the original file.
# The original file likely ended with `sys.exit(app.exec_())` or similar.

marker = "sys.exit(app.exec_())"
idx = content_str.rfind(marker)

if idx != -1:
    # Found the end of the main block.
    # We want to insert the methods INSIDE MainWindow class.
    # So we need to find the end of MainWindow class.
    # Usually MainWindow definitions end before `if __name__ == "__main__":`.
    
    clean_content = content_str[:idx + len(marker)]
    
    # Check for `if __name__`
    name_check = "if __name__ == \"__main__\":"
    name_idx = clean_content.rfind(name_check)
    
    if name_idx != -1:
        # We instruct the replacement:
        # 1. Keep everything up to `if __name__ ...`
        # 2. Insert snippet (indented!)
        # 3. Add back the `if __name__ ...` block
        
        pre_main = clean_content[:name_idx]
        main_block = clean_content[name_idx:]
        
        # Indent snippet
        indented_snippet = ""
        for line in snippet_content.splitlines():
            indented_snippet += "    " + line + "\n"
            
        new_content = pre_main + "\n" + indented_snippet + "\n" + main_block
        
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("Successfully repaired and patched file.")
    else:
        print("Could not find __main__ block to insert before.")
else:
    print("Could not find end marker. dumping tail...")
    print(content_str[-500:])

