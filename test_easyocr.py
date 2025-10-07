import easyocr
import json
import os

# Load reader with GPU if available, optimized for English (includes digits)
reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if no GPU/CUDA

output = []

def process_frame(inframe):
    # Read text from image, restricted to digits and timestamp chars for optimization
    result = reader.readtext(in_frame, allowlist='0123456789:.')  # Focus on numbers/timestamps
    
    # Extract texts (will be mostly/only numbers if detected)
    texts = [r[1] for r in result]
    
    # Calculate approximate video timestamp
    timestamp_sec = i / fps
    
    output.append({
        'frame': i + 1,
        'file': os.path.basename(frame_path),
        'timestamp_sec': timestamp_sec,
        'extracted_texts': texts
    })

# Save to JSON
with open('ocr_timestamps.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("OCR complete—results in ocr_timestamps.json")
