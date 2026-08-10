# TODO — Upgrade Profesional ASL Realtime

- [x] Analisis file project (realtime.py, helpers.py, predict.py, requirements.txt)
- [x] Rencana disetujui user
- [x] Tambahkan `mediapipe` di requirements.txt
- [x] Refactor realtime.py menjadi modular (draw_ui, draw_progress_bar, draw_hand_landmarks, draw_status, draw_sentence, draw_special_animation)
- [x] Integrasi MediaPipe Hands sebagai visualisasi (skeleton 21 landmark + koneksi + bbox)
- [x] Implementasi Auto ROI (bbox tangan + 15% margin, fallback ROI statis)
- [x] Upgrade UI: overlay transparan, panel info, progress bar, status hand, inference time, FPS
- [x] Upgrade Special Mode: freeze → fade to black → ✨ SECRET UNLOCKED ✨ → loading → countdown 3-2-1 → buka browser
- [x] Pertahankan semua fitur lama (EfficientNetB0, smoothing, threshold, trigger, keyboard, ROI fallback)
- [x] Uji sintaks (py_compile) - OK
