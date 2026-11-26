NEMESIS-X — ULTIMATE HACKATHON BUILD
=====================================

CONTENTS
--------
app.py                 Main Streamlit web application
nemesis.db             SQLite database (tables are also auto-created on launch)
requirements.txt       Python dependencies
README.txt             Setup, run and usage guide
uploads/               Original uploaded videos
outputs/               Annotated slow-motion videos
person_crops/          Captured best crop for each anonymous person label
evidence/              Timestamped annotated evidence frames
clips/                 Reserved for future person-only clip exports

WINDOWS SETUP (VS CODE / POWERSHELL)
------------------------------------
1. Extract the ZIP into a NEW folder. Do not overwrite the existing project.
2. Open that extracted folder in VS Code.
3. Open Terminal > New Terminal.
4. Run:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m streamlit run app.py

If PowerShell blocks activation, run:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1

UBUNTU / WSL SETUP
------------------
   sudo apt update
   sudo apt install -y python3-venv
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m streamlit run app.py

HOW TO DEMO
-----------
1. Upload an MP4/AVI/MOV/MKV file in the Analyse video tab.
2. Set Maximum unique person IDs to 8 for an eight-person demo.
3. Keep confidence around 0.35 and frame sampling at 1 or 2 for clearer tracking.
4. Select annotated playback speed 0.5x for slower playback.
5. Click Analyse footage and wait for completion.
6. Open Person investigation. Person cards show the captured crop beside each PERSON label.
7. Select a person to see first observed time, last observed time, observed span,
   detector confidence, event timeline, evidence images, original video and annotated video.
8. Download the person's event timeline as CSV.

IMPORTANT ACCURACY NOTES
------------------------
- PERSON-01, PERSON-02, etc. are anonymous labels assigned in order of first detection
  within one analysed video. They are not permanent identities across separate videos.
- ByteTrack can lose/reassign an ID when someone is hidden, leaves the frame, or re-enters.
  The system does not use facial recognition or identify real-world identities.
- "First observed" is the first detected frame. "Last observed" is the last detected frame.
  Without a defined exit line/zone, the app cannot prove the person's physical entry/exit.
- The current event timeline records observed presence and image-position movement.
  It does not claim to know a person's intention or prove an object was handled.
- Setting max person IDs to 8 limits new labels to eight. If more than eight distinct
  tracker identities occur, later identities are not added.
- The first YOLO run downloads model weights. Internet is required for that first download.
- CPU processing may be slow. Use a short sample video first.

DATABASE
--------
SQLite database file: nemesis.db
Tables: videos, persons, events.
The app creates missing tables automatically if the database is absent.
Keep the database file and its related uploads/outputs/evidence folders together.

TROUBLESHOOTING
---------------
- "No module named cv2": python -m pip install opencv-python-headless
- "No module named streamlit": python -m pip install streamlit
- Model download fails: check internet, then retry.
- Video will not open: convert it to MP4 (H.264) and retry.
- Do not run `pip` alone; use `python -m pip` in the activated environment.
