import os
from pathlib import Path


def get_directory_structure(root_dir: str, exclude_dirs=None, exclude_files=None):
    """Возвращает структуру директории в виде строки"""
    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'env', '.idea', '.vscode'}
    if exclude_files is None:
        exclude_files = {'.gitignore', '.env', '*.pyc', '*.pyo', '__pycache__'}

    result_lines = []

    def _walk(dir_path, prefix=""):
        try:
            items = sorted([p for p in Path(dir_path).iterdir()],
                           key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return

        for i, item in enumerate(items):
            # Пропускаем исключенные элементы
            if item.name in exclude_dirs or item.name in exclude_files:
                continue
            if any(item.name.endswith(ext) for ext in ['.pyc', '.pyo']):
                continue

            is_last = (i == len(items) - 1)

            if item.is_dir():
                result_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{item.name}/")
                extension = "    " if is_last else "│   "
                _walk(item, prefix + extension)
            else:
                result_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")

    result_lines.append(f"{Path(root_dir).name}/")
    _walk(root_dir)
    return "\n".join(result_lines)


# Использование
if __name__ == "__main__":
    structure = get_directory_structure(Path(__file__).parent.parent.parent)  # Укажите ваш путь
    print(structure)