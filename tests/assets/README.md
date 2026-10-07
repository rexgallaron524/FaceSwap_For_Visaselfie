# Reproducible fixtures

`controlled_motion.mp4` is an eight-second, 640×360, 30 FPS deterministic test clip. It is
generated from the synthetic adult pose guides in `assets/pose_guides`; it contains no
recording of a real participant. Regenerate it from the repository root with:

```powershell
.\.venv\Scripts\python.exe tools\test_clips\generate_controlled_motion.py
```

| Time | Controlled behavior |
| --- | --- |
| 0.0–1.0 s | Steady front-neutral baseline |
| 1.0–2.5 s | Horizontal translation from left to right |
| 2.5–4.0 s | Front → left-20 → front → right-20 → front |
| 4.0–5.0 s | Scale down/up/down |
| 5.0–6.0 s | Roll from −9° through +9° and back |
| 6.0–7.0 s | Neutral → smile → neutral expression |
| 7.4–7.5 s | Three blank frames for a 100 ms tracking dropout |

The remaining frames provide a steady recovery segment. The `mp4v` encoding may vary
slightly by OpenCV build, while frame count, dimensions, timing, and source transforms remain
fixed. Other binary fixtures remain ignored unless explicitly allowlisted in `.gitignore`.
