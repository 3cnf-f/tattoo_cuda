from faster_whisper import WhisperModel
import json
#Claude helped me with this

filename_base = "144850"
filename_add_wav = "_4fps_sound.wav"
filename_add_json = "_4fps.json"

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(filename_base+filename_add_wav, language="sv")
results = []
for segment in segments:
    results.append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    })
with open(filename_base+filename_add_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Transcribed {len(results)} segments")
