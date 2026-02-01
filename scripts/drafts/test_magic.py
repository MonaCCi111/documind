import magic
from pathlib import Path
print(magic.from_file(str(Path(__file__).parent.parent.parent / 'data/raw/draft_tests/test_pdf.pdf')))
