import argparse
from pathlib import Path
import sys


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    default_target = project_root / "main_camera_ui.py"
    default_snippet = script_dir / "halcon_methods_snippet.py"

    parser = argparse.ArgumentParser(description="Repair and reinsert Halcon UI methods.")
    parser.add_argument("--target", default=str(default_target), help="Path to target UI file.")
    parser.add_argument("--snippet", default=str(default_snippet), help="Path to snippet file.")
    args = parser.parse_args()

    target_file = Path(args.target).resolve()
    snippet_file = Path(args.snippet).resolve()

    if not target_file.exists():
        print(f"Target file not found: {target_file}")
        return 1
    if not snippet_file.exists():
        print(f"Snippet file not found: {snippet_file}")
        return 1

    snippet_content = snippet_file.read_text(encoding="utf-8")

    try:
        content_bytes = target_file.read_bytes()
        content_str = content_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    marker = "sys.exit(app.exec_())"
    idx = content_str.rfind(marker)
    if idx == -1:
        print("Could not find end marker. dumping tail...")
        print(content_str[-500:])
        return 1

    clean_content = content_str[: idx + len(marker)]
    name_check = "if __name__ == \"__main__\":"
    name_idx = clean_content.rfind(name_check)
    if name_idx == -1:
        print("Could not find __main__ block to insert before.")
        return 1

    pre_main = clean_content[:name_idx]
    main_block = clean_content[name_idx:]
    indented_snippet = "".join(f"    {line}\n" for line in snippet_content.splitlines())
    new_content = f"{pre_main}\n{indented_snippet}\n{main_block}"
    target_file.write_text(new_content, encoding="utf-8")
    print("Successfully repaired and patched file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

