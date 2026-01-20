from PIL import Image
from pathlib import Path
import easyocr

file_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'test_image.png'

image = Image.open(file_path)

reader = easyocr.Reader(['ru', 'en'], gpu=True)

results = reader.readtext(str(file_path), paragraph=True)

print(results)