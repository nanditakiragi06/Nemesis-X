from pathlib import Path
from datetime import datetime
import sqlite3, hashlib, json, math, shutil

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).parent
DB_PATH = ROOT / "nemesis.db"
DIRS = {n: ROOT/n for n in ["uploads","outputs","evidence","person_crops","clips"]}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="NEMESIS-X | Video Intelligence", page_icon="🛰️", layout="wide")

def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS videos(
      id INTEGER PRIMARY KEY AUTOINCREMENT, source_name TEXT NOT NULL,
      source_path TEXT NOT NULL, annotated_path TEXT, created_at TEXT NOT NULL,
      duration REAL, fps REAL, width INTEGER, height INTEGER, processed_frames INTEGER,
      model_name TEXT, confidence REAL, slow_factor REAL
    );
    CREATE TABLE IF NOT EXISTS persons(
      id INTEGER PRIMARY KEY AUTOINCREMENT, video_id INTEGER NOT NULL,
      pid TEXT NOT NULL, tracker_id INTEGER, first_seen REAL, last_seen REAL,
      seen_seconds REAL, observations INTEGER, best_confidence REAL,
      image_path TEXT, FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE,
      UNIQUE(video_id,pid)
    );
    CREATE TABLE IF NOT EXISTS events(
      id INTEGER PRIMARY KEY AUTOINCREMENT, video_id INTEGER NOT NULL,
      pid TEXT, event_type TEXT NOT NULL, timestamp REAL NOT NULL,
      end_timestamp REAL, description TEXT NOT NULL, evidence_path TEXT,
      confidence TEXT DEFAULT 'Observed', FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)
    con.commit()
    return con

def stamp(s):
    s=max(0.0,float(s or 0))
    return f"{int(s//60):02d}:{s%60:05.2f}"

def safe_stem(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(name).stem)[:60] or "video"

def process_video(video_path, source_name, model_name, conf, max_people, slow_factor, sample_every):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video. Please use a valid MP4 or AVI file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if total_frames else 0
    out_path = DIRS["outputs"] / f"{safe_stem(source_name)}_annotated_slow.mp4"
    output_fps = max(1.0, fps * slow_factor / sample_every)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width,height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create annotated MP4. Check output folder permissions.")
    people = {}  # tracker id -> person record
    tracker_to_pid = {}
    events = []
    frame_no = 0
    processed = 0
    pbar = st.progress(0, text="Preparing video analysis…")
    preview = st.empty()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if (frame_no-1) % sample_every:
                continue
            t = (frame_no-1)/fps
            res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              classes=[0], conf=conf, verbose=False)[0]
            canvas = frame.copy()
            current_ids = set()
            if res.boxes is not None and len(res.boxes) and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                scores = res.boxes.conf.cpu().numpy()
                for box, tid, score in zip(boxes, ids, scores):
                    tid = int(tid)
                    if tid not in tracker_to_pid:
                        if len(tracker_to_pid) >= max_people:
                            continue
                        pid = f"PERSON-{len(tracker_to_pid)+1:02d}"
                        tracker_to_pid[tid] = pid
                        x1,y1,x2,y2 = map(int,box)
                        crop = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
                        image_path = DIRS["person_crops"] / f"{safe_stem(source_name)}_{pid}.jpg"
                        if crop.size:
                            cv2.imwrite(str(image_path), crop)
                        people[tid] = dict(pid=pid, first=t, last=t, frames=0,
                                           conf=float(score), image_path=str(image_path),
                                           centers=[], best_area=max(0,(x2-x1)*(y2-y1)))
                        events.append((pid,"First observed",t,None,
                                       f"{pid} first appeared in the footage at {stamp(t)}.",
                                       str(image_path),"Observed detection"))
                    pid = tracker_to_pid[tid]
                    x1,y1,x2,y2 = map(int,box)
                    rec = people[tid]
                    rec["last"] = t
                    rec["frames"] += 1
                    rec["conf"] = max(rec["conf"],float(score))
                    cx,cy=(x1+x2)//2,(y1+y2)//2
                    rec["centers"].append((t,cx,cy))
                    current_ids.add(tid)
                    area=max(0,(x2-x1)*(y2-y1))
                    if area > rec["best_area"]:
                        crop=frame[max(0,y1):min(height,y2),max(0,x1):min(width,x2)]
                        if crop.size:
                            cv2.imwrite(rec["image_path"],crop)
                            rec["best_area"]=area
                    color=(20,220,170)
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                    cv2.rectangle(canvas,(x1,max(0,y1-27)),(x1+150,y1),color,-1)
                    cv2.putText(canvas,pid,(x1+5,max(18,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.58,(5,15,25),2)
                    cv2.putText(canvas,f"{float(score):.0%}",(x1,max(18,y2+20)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1)
            # Annotated frame includes a timestamp and project mark.
            cv2.putText(canvas,f"NEMESIS-X  |  {stamp(t)}",(16,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            writer.write(canvas)
            processed += 1
            if processed == 1 or processed % max(1,int(fps*2/sample_every)) == 0:
                ev = DIRS["evidence"] / f"{safe_stem(source_name)}_frame_{frame_no:07d}.jpg"
                cv2.imwrite(str(ev),canvas)
                for tid in current_ids:
                    rec=people[tid]
                    events.append((rec["pid"],"Presence observed",t,None,
                                   f"{rec['pid']} was visible at {stamp(t)}.",str(ev),"Observed detection"))
            if processed % 4 == 0:
                preview.image(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB),
                              caption=f"Processing frame {frame_no} · {stamp(t)}", use_container_width=True)
            if total_frames:
                pbar.progress(min(1.0,frame_no/total_frames),
                              text=f"Analysing frame {frame_no:,} / {total_frames:,}")
    finally:
        cap.release()
        writer.release()
    pbar.progress(1.0,text="Video analysis completed")
    # Movement summary observations, not claims of intent.
    for tid,rec in people.items():
        pts=rec["centers"]
        if len(pts)>1:
            dist=sum(math.hypot(pts[i][1]-pts[i-1][1],pts[i][2]-pts[i-1][2]) for i in range(1,len(pts)))
            events.append((rec["pid"],"Movement observed",rec["first"],rec["last"],
                           f"{rec['pid']} changed image position across the clip (approx. {int(dist)} pixels of cumulative centre movement).",
                           rec["image_path"],"Position-based observation"))
        events.append((rec["pid"],"Last observed",rec["last"],None,
                       f"{rec['pid']} was last detected at {stamp(rec['last'])}. This is the last observed time, not a verified physical exit time.",
                       rec["image_path"],"Observed detection"))
    return out_path, people, events, (duration,fps,width,height,processed)

def save_results(source_name, source_path, out_path, people, events, meta, model_name, conf, slow_factor):
    con=db()
    cur=con.execute("""INSERT INTO videos(source_name,source_path,annotated_path,created_at,duration,fps,width,height,processed_frames,model_name,confidence,slow_factor)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (source_name,str(source_path),str(out_path),datetime.now().isoformat(timespec="seconds"),
                     *meta,model_name,conf,slow_factor))
    vid=cur.lastrowid
    for tid,rec in people.items():
        con.execute("""INSERT INTO persons(video_id,pid,tracker_id,first_seen,last_seen,seen_seconds,observations,best_confidence,image_path)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (vid,rec["pid"],tid,rec["first"],rec["last"],
                     max(0,rec["last"]-rec["first"]),rec["frames"],rec["conf"],rec["image_path"]))
    for pid,etype,t,end,desc,ev,confidence in events:
        con.execute("""INSERT INTO events(video_id,pid,event_type,timestamp,end_timestamp,description,evidence_path,confidence)
                       VALUES(?,?,?,?,?,?,?,?)""",(vid,pid,etype,t,end,desc,ev,confidence))
    con.commit()
    con.close()
    return vid

def fetch_df(sql,params=()):
    con=db()
    df=pd.read_sql_query(sql,con,params=params)
    con.close()
    return df

# UI
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 85% 0%,#172a4a 0,#0b1220 38%,#080d18 100%);color:#eef4ff}
.hero{padding:24px 28px;border:1px solid #29466e;border-radius:18px;background:linear-gradient(110deg,#142c4e,#24254b);margin-bottom:18px}
.hero h1{color:#fff;margin:0;font-size:2.1rem}.hero p{color:#c1d2ef;margin:7px 0 0}
.person-card{border:1px solid #29466e;border-radius:14px;padding:10px;background:#111d31}
</style>
<div class="hero"><h1>🛰️ NEMESIS-X</h1>
<p>Open-World Video Intelligence · Person-wise tracking · Temporal evidence</p></div>
""",unsafe_allow_html=True)

db()  # initialise database
with st.sidebar:
    st.header("Analysis controls")
    model_name=st.selectbox("Detection model",["yolo11n.pt","yolov8n.pt"])
    confidence=st.slider("Detection confidence",0.10,0.90,0.35,0.05)
    max_people=st.number_input("Maximum unique person IDs",min_value=1,max_value=30,value=8)
    sample_every=st.select_slider("Frame sampling",options=[1,2,3,4,5],value=2,
                                  help="1 processes every frame; larger values are faster but less detailed.")
    slow_factor=st.select_slider("Annotated video speed",options=[0.25,0.33,0.5,0.67,1.0],value=0.5,
                                 format_func=lambda x:f"{x}× speed")
    st.caption("Person IDs are anonymous and valid only within this analysed video. No facial recognition is used.")

tab1,tab2,tab3=st.tabs(["🎥 Analyse video","👤 Person investigation","🗂️ Database & history"])
with tab1:
    st.subheader("Upload CCTV footage")
    upload=st.file_uploader("Select a video",type=["mp4","avi","mov","mkv"],key="video_upload")
    if upload:
        st.video(upload)
        if st.button("🚀 Analyse footage",type="primary",use_container_width=True):
            file_hash=hashlib.sha256(upload.getvalue()).hexdigest()[:12]
            saved=DIRS["uploads"]/f"{file_hash}_{Path(upload.name).name}"
            saved.write_bytes(upload.getvalue())
            try:
                with st.spinner("Loading model and analysing frames. The first run may download YOLO weights…"):
                    out,people,events,meta=process_video(saved,upload.name,model_name,confidence,int(max_people),float(slow_factor),int(sample_every))
                    vid=save_results(upload.name,saved,out,people,events,meta,model_name,confidence,float(slow_factor))
                st.session_state["active_video_id"]=vid
                st.success(f"Analysis saved successfully. {len(people)} unique person labels created.")
            except Exception as e:
                st.error(f"Could not complete analysis: {e}")
                st.info("Check internet for first model download, use a short MP4 test clip, and confirm requirements are installed.")
    active=st.session_state.get("active_video_id")
    if active:
        row=fetch_df("SELECT * FROM videos WHERE id=?",(active,))
        if not row.empty:
            v=row.iloc[0]
            st.subheader("Processed video")
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**Original uploaded footage**")
                if Path(v.source_path).exists(): st.video(v.source_path)
                else: st.warning("Original file path is unavailable.")
            with c2:
                st.markdown(f"**Annotated footage ({v.slow_factor}× playback)**")
                if v.annotated_path and Path(v.annotated_path).exists():
                    st.video(v.annotated_path)
                    st.download_button("Download slow-motion annotated video",Path(v.annotated_path).read_bytes(),
                                       file_name="NEMESIS_X_annotated_slow.mp4",mime="video/mp4")
                else: st.warning("Annotated output is unavailable.")
with tab2:
    st.subheader("Person directory")
    vids=fetch_df("SELECT id,source_name,created_at FROM videos ORDER BY id DESC")
    if vids.empty:
        st.info("Analyse a video first. Detected people will appear here with their captured images.")
    else:
        chosen_video=st.selectbox("Select analysed video",vids.id.tolist(),
              format_func=lambda x:f"#{x} · {vids.loc[vids.id==x,'source_name'].iloc[0]}")
        persons=fetch_df("SELECT * FROM persons WHERE video_id=? ORDER BY CAST(SUBSTR(pid,8) AS INTEGER)",(int(chosen_video),))
        if persons.empty:
            st.warning("No tracked people were detected in this video.")
        else:
            st.caption(f"{len(persons)} unique person labels stored for this video.")
            cols=st.columns(4)
            for i,(_,p) in enumerate(persons.iterrows()):
                with cols[i%4]:
                    st.markdown('<div class="person-card">',unsafe_allow_html=True)
                    if p.image_path and Path(p.image_path).exists():
                        st.image(p.image_path,use_container_width=True)
                    else:
                        st.markdown("**Image unavailable**")
                    st.markdown(f"**{p.pid}**")
                    st.caption(f"First: {stamp(p.first_seen)} · Last: {stamp(p.last_seen)}")
                    st.markdown('</div>',unsafe_allow_html=True)
            selected=st.selectbox("Open person investigation",persons.pid.tolist())
            person=persons[persons.pid==selected].iloc[0]
            st.divider()
            left,right=st.columns([1,2])
            with left:
                if person.image_path and Path(person.image_path).exists():
                    st.image(person.image_path,caption=f"Captured image · {selected}",use_container_width=True)
                st.metric("First observed / entry into view",stamp(person.first_seen))
                st.metric("Last observed",stamp(person.last_seen))
                st.metric("Observed span",f"{max(0,person.last_seen-person.first_seen):.2f} sec")
                st.metric("Detection observations",int(person.observations))
                st.metric("Best detector confidence",f"{person.best_confidence:.0%}")
                st.caption("Entry/exit are camera-view estimates: last observed is not proof of a physical exit.")
            with right:
                st.markdown(f"### {selected} · Activity and event timeline")
                ev=fetch_df("""SELECT * FROM events WHERE video_id=? AND pid=? ORDER BY timestamp,id""",
                            (int(chosen_video),selected))
                if ev.empty:
                    st.info("No event observations stored.")
                else:
                    for _,e in ev.iterrows():
                        st.markdown(f"**{stamp(e.timestamp)} — {e.event_type}**")
                        st.write(e.description)
                        st.caption(f"Evidence status: {e.confidence}")
                        if e.evidence_path and Path(e.evidence_path).exists():
                            with st.expander(f"View evidence at {stamp(e.timestamp)}"):
                                st.image(e.evidence_path,use_container_width=True)
                st.markdown("**Uploaded and annotated video**")
                v=fetch_df("SELECT * FROM videos WHERE id=?",(int(chosen_video),)).iloc[0]
                vc1,vc2=st.columns(2)
                with vc1:
                    if Path(v.source_path).exists(): st.video(v.source_path)
                    st.caption("Original uploaded video")
                with vc2:
                    if v.annotated_path and Path(v.annotated_path).exists(): st.video(v.annotated_path)
                    st.caption("Annotated slow-motion video")
                csv=ev.to_csv(index=False).encode("utf-8")
                st.download_button("Download this person's event timeline (CSV)",csv,
                                   file_name=f"{selected}_timeline.csv",mime="text/csv")
with tab3:
    st.subheader("Saved analyses")
    videos=fetch_df("SELECT * FROM videos ORDER BY id DESC")
    if videos.empty: st.info("The database is ready. No analyses have been saved yet.")
    else: st.dataframe(videos.drop(columns=["source_path","annotated_path"]),use_container_width=True,hide_index=True)
    with st.expander("All tracked people"):
        st.dataframe(fetch_df("SELECT * FROM persons ORDER BY video_id,pid"),use_container_width=True,hide_index=True)
    with st.expander("All event observations"):
        st.dataframe(fetch_df("SELECT * FROM events ORDER BY video_id,timestamp"),use_container_width=True,hide_index=True)
    if st.button("Refresh database view"): st.rerun()

st.caption("NEMESIS-X prototype. ByteTrack IDs can change after occlusion or re-entry. This application does not identify real-world people. Event records are limited to observed detections and position changes; it does not infer intent or prove object interaction.")
