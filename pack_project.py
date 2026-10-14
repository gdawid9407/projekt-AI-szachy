from pathlib import Path
from typing import Set

VALID_EXTENSIONS: Set[str] = {'.py', '.md', '.txt', '.yaml', '.yml', '.toml'}
EXCLUDE_DIRS: Set[str] = {'.git', '.venv', 'venv', '__pycache__', 'data', 'models'}


def pack() -> None:
    root = Path(__file__).resolve().parent
    output_file = root / "project_context.txt"
    collected = []

    for file_path in sorted(root.rglob('*')):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("._") or file_path.name == ".DS_Store":
            continue
        if any(part in EXCLUDE_DIRS for part in file_path.parts):
            continue
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        if file_path.resolve() in {output_file.resolve(), Path(__file__).resolve()}:
            continue

        relative_path = file_path.relative_to(root).as_posix()
        header = f"\n{'=' * 50}\nFile: {relative_path}\n{'=' * 50}\n"

        try:
            content = file_path.read_text(encoding='utf-8')
            collected.append(header + content)
        except Exception as e:
            collected.append(f"{header}Error reading file: {e}\n")

    output_file.write_text("\n".join(collected), encoding='utf-8')
    print(f"Project packed into {output_file.name} ({len(collected)} files).")


if __name__ == "__main__":
    pack()