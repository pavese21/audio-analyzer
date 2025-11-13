import io, os, traceback
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

NOTES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def _rot(v, n: int):
    o = np.zeros(12)
    for i in range(12):
        o[(i + n) % 12] = v[i]
    return o

def _suggestions(tonic, scale, conf: float):
    if scale == "major":
        rel = ((tonic + 9) % 12, "minor", "relative minor")
        par = (tonic, "minor", "parallel")
    else:
        rel = ((tonic + 3) % 12, "major", "relative major")
        par = (tonic, "major", "parallel")

    V  = ((tonic + 7) % 12, "major", "dominant (V)")
    IV = ((tonic + 5) % 12, "major", "subdominant (IV)")

    order = [rel, par, V, IV] if conf >= 0.25 else [rel, IV, V, par]
    return [
        {"name": f"{NOTES[t]} {s}", "scale": s, "relation": r}
        for (t, s, r) in order[:3]
    ]

def _estimate_key(y, sr: int):
    y_h, _ = librosa.effects.hpss(y)
    tuning = librosa.estimate_tuning(y=y_h, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, tuning=tuning)
    c = np.median(chroma, axis=1)

    n = np.linalg.norm(c)
    c = c / n if n > 0 else c

    best_score, tonic, mode = -1.0, 0, "major"
    for t in range(12):
        sM = float(c @ _rot(KS_MAJOR, (12 - t) % 12))
        if sM > best_score:
            best_score, tonic, mode = sM, t, "major"

        sm = float(c @ _rot(KS_MINOR, (12 - t) % 12))
        if sm > best_score:
            best_score, tonic, mode = sm, t, "minor"

    return {
        "tonic_idx": tonic,
        "name": f"{NOTES[tonic]} {mode}",
        "scale": mode,
        "confidence": float(round(best_score, 2)),
        "tuning_semitones": float(round(tuning, 2)),
    }

def _estimate_bpm(y, sr: int):
    onset = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, trim=False
    )
    bpm = int(round(float(tempo)))
    times = librosa.frames_to_time(beats, sr=sr).tolist()

    stab = 0.0
    if len(times) > 2:
        itv = np.diff(times)
        stab = max(
            0.0,
            min(1.0, 1.0 - float(np.std(itv) / (np.mean(itv) + 1e-9))),
        )

    half = int(round(bpm / 2))
    dbl  = int(round(bpm * 2))
    alts = [dbl, half] if (bpm < 70 and dbl <= 200) else [half, dbl]

    return {
        "primary": bpm,
        "alternates": alts,
        "confidence": float(round(stab, 2)),
        "beats": [float(round(t, 3)) for t in times],
    }

app = FastAPI()

@app.get("/health")
def health():
    return PlainTextResponse("ok", status_code=200)

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    clip_seconds: float = Form(180.0),
    x_analyzer_key: str | None = Header(None),
):
    # Optional API key gate
    analyzer_key = os.getenv("ANALYZER_KEY")
    if analyzer_key and x_analyzer_key != analyzer_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")

        # Load audio (librosa uses ffmpeg/audioread under the hood for mp3)
        y, sr = librosa.load(io.BytesIO(raw), sr=22050, mono=True)

        # Clip length for speed/safety
        if len(y) > sr * clip_seconds:
            y = y[: int(sr * clip_seconds)]

        # Normalize
        peak = np.max(np.abs(y)) + 1e-9
        y = y / peak

        # Run analysis
        k = _estimate_key(y, sr)
        t = _estimate_bpm(y, sr)

        return JSONResponse(
            {
                "ok": True,
                "result": {
                    "key": {
                        "tonic": k["name"].split()[0],
                        "scale": k["scale"],
                        "confidence": k["confidence"],
                        "tuning_semitones": k["tuning_semitones"],
                    },
                    "key_suggestions": _suggestions(
                        k["tonic_idx"], k["scale"], k["confidence"]
                    ),
                    "bpm": t,
                },
            }
        )

    except HTTPException:
        # re-raise clean HTTP errors
        raise
    except Exception as e:
        # log full traceback to Railway logs
        tb = traceback.format_exc()
        print("ANALYZER ERROR:\n", tb, flush=True)
        # return readable message to caller
        raise HTTPException(
            status_code=500, detail=f"Analyzer failed: {e}"
        )
