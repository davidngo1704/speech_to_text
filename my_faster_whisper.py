
from faster_whisper import WhisperModel

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="int8"
)

segments, info = model.transcribe(
    "record_ok.wav",
    language="vi",
    task="transcribe",
    beam_size=5,
    temperature=0.0,
    vad_filter=True,
)

for seg in segments:
    print(
        f"[{seg.start:.2f}s → {seg.end:.2f}s] {seg.text}"
    )

