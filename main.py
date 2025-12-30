import whisper

model = whisper.load_model("medium").to("cuda")

#model = whisper.load_model("large").to("cuda")

result = model.transcribe(
    "record_ok.wav",
    language="vi",
    task="transcribe",
    fp16=True,
    verbose=False,
    temperature=0.0,          # giảm hallucination
    condition_on_previous_text=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    beam_size=5,
    best_of=5, 
)

segments = result["segments"]

for seg in segments:
    print(
        f"[{seg['start']:.2f}s → {seg['end']:.2f}s] {seg['text']}"
    )
