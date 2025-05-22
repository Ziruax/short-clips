import streamlit as st
import os
import tempfile
import shutil # For cleanup
import uuid # For unique session IDs if needed, or unique file names
import yt_dlp
from moviepy.editor import VideoFileClip # AudioFileClip not explicitly used but good to have if needed
# from moviepy.video.fx.all import crop # crop is directly available
import cv2 # opencv-python-headless
from faster_whisper import WhisperModel
import librosa
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import json # For potential config loading/saving in future

# --- Configuration & Constants ---
APP_TITLE = "🤖 Viral Short Clip Automator Pro 🎞️✨"
TEMP_DIR_BASE = "temp_files_viral_automator" # Base for persistent temp files
MAX_CLIP_SUGGESTIONS_TO_PROCESS = 5 # Max clips to process fully
TARGET_ASPECT_RATIO = 9.0 / 16.0
FACE_ANALYSIS_SAMPLE_INTERVAL_SEC = 2 # Analyze faces every N seconds in the source video

# Initial "Viral" Keywords - User can expand this list via UI
DEFAULT_VIRAL_KEYWORDS = [
    "secret", "hack", "amazing", "shocking", "unbelievable", "must-see",
    "game-changer", "life-changing", "revealed", "exposed", "top tip",
    "you won't believe", "never seen before", "mind-blowing", "insane",
    "crazy", "wow", "omg", "finally", "easy way", "quickest method", "tutorial",
    "step-by-step", "how to", "discover", "unlock"
]

# --- Helper Functions ---

@st.cache_resource
def load_whisper_model(model_size="tiny.en"):
    try:
        st.info(f"Loading Whisper model '{model_size}'... This may take a moment.")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        st.success(f"Whisper model '{model_size}' loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}. Try a smaller model or check internet.")
        return None

@st.cache_resource
def get_face_cascade():
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        st.error("Haar Cascade file not found. Face detection will be limited.")
        return None
    return cv2.CascadeClassifier(cascade_path)

@st.cache_resource
def get_vader_analyzer():
    return SentimentIntensityAnalyzer()

def clear_app_cache_and_temp():
    """Clears Streamlit caches and the application's temporary directory."""
    st.cache_resource.clear()
    st.cache_data.clear() # If you start using st.cache_data

    # Clear specific session state keys
    keys_to_clear = ['youtube_video_path', 'final_clips_info', 'downloaded_yt_info', 'source_video_details']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    if os.path.exists(TEMP_DIR_BASE):
        try:
            shutil.rmtree(TEMP_DIR_BASE)
            st.success(f"Cleared temporary directory: {TEMP_DIR_BASE}")
            os.makedirs(TEMP_DIR_BASE, exist_ok=True) # Recreate base
        except Exception as e:
            st.warning(f"Could not fully clear {TEMP_DIR_BASE}: {e}")
    else:
        st.info("Temporary directory not found, nothing to clear there.")
    st.success("Caches and session state related to processing cleared. Consider refreshing the page.")


def download_youtube_video(url, output_dir):
    """Downloads a YouTube video, attempting to get a clean title for the filename."""
    if 'downloaded_yt_info' in st.session_state and st.session_state.downloaded_yt_info.get('url') == url:
        if os.path.exists(st.session_state.downloaded_yt_info['path']):
            st.info("Using previously downloaded YouTube video for this session.")
            return st.session_state.downloaded_yt_info['path']
        else:
            st.warning("Previously downloaded video path not found, re-downloading.")

    st.info(f"Preparing to download YouTube video: {url}")
    progress_bar_yt = st.progress(0.0, text="Initializing download...") # Initialize progress bar

    # Attempt to get video info first to make a cleaner filename
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'noplaylist': True, 'extract_flat': True}) as ydl_info:
            info_dict = ydl_info.extract_info(url, download=False)
            video_title = info_dict.get('title', 'youtube_video')
            # Sanitize title for filename
            sane_title = re.sub(r'[^\w\s-]', '', video_title).strip().replace(' ', '_')
            sane_title = re.sub(r'[-_]+', '_', sane_title) # Consolidate underscores/hyphens
            base_filename = f"{sane_title[:50]}.mp4" # Limit length
    except Exception:
        base_filename = f"youtube_video_{uuid.uuid4().hex[:8]}.mp4" # Fallback unique name

    output_path = os.path.join(output_dir, base_filename)

    counter = 0
    original_output_path = output_path
    while os.path.exists(output_path):
        counter += 1
        name, ext = os.path.splitext(original_output_path)
        output_path = f"{name}_{counter}{ext}"

    def yt_progress_hook(d):
        if d['status'] == 'downloading':
            # _percent_str is like ' 50.6%' or '100.0%'
            # _total_bytes_str might be present for total size
            # _downloaded_bytes_str for current downloaded
            percent_str = d.get('_percent_str', '0%')
            # Clean up the string: remove percentage, strip whitespace
            try:
                # Convert to float and divide by 100 for st.progress
                progress_value = float(percent_str.replace('%', '').strip()) / 100.0
                # Ensure it's within [0.0, 1.0]
                progress_value = max(0.0, min(1.0, progress_value))
                
                # Get eta string if available
                eta_str = d.get('_eta_str', '')
                speed_str = d.get('_speed_str', '')
                
                progress_text = f"Downloading: {percent_str.strip()} complete"
                if speed_str:
                    progress_text += f" at {speed_str.strip()}"
                if eta_str:
                    progress_text += f" (ETA: {eta_str.strip()})"
                
                progress_bar_yt.progress(progress_value, text=progress_text)
            except ValueError:
                # Handle cases where conversion might fail, though unlikely with _percent_str
                progress_bar_yt.progress(0.0, text=f"Processing download: {percent_str.strip()}")
        elif d['status'] == 'finished':
            progress_bar_yt.progress(1.0, text="Download finished. Processing...")
        elif d['status'] == 'error':
            progress_bar_yt.progress(0.0, text="Error during download.")


    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/mp4[height<=1080]/best[height<=1080]',
        'outtmpl': output_path,
        'noplaylist': True,
        'quiet': True, # Set to True if you fully rely on the hook for visual feedback
        'progress_hooks': [yt_progress_hook],
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'aac',
        }],
        # 'verbose': True, # Uncomment for debugging yt-dlp itself
    }
    try:
        # Using a spinner here might be redundant if the progress bar updates quickly
        # with st.spinner(f"Downloading with yt-dlp to {os.path.basename(output_path)}..."):
        st.write(f"Starting download to: {os.path.basename(output_path)}") # Initial message
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # The 'finished' status in the hook should update the bar to 1.0
        st.success(f"YouTube video downloaded: {os.path.basename(output_path)}")
        st.session_state.downloaded_yt_info = {'url': url, 'path': output_path, 'title': video_title if 'video_title' in locals() else 'YouTube Video'}
        if progress_bar_yt: progress_bar_yt.empty() # Clear the progress bar after success
        return output_path
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}") # This is where your original error was caught
        if progress_bar_yt: progress_bar_yt.empty() # Clear progress bar on error
        return None
        
def transcribe_audio_with_whisper(audio_path, whisper_model):
    if not whisper_model:
        st.error("Whisper model not loaded. Cannot transcribe.")
        return None, None, None
    try:
        st.info(f"Transcribing audio: {os.path.basename(audio_path)} (this may take a while)...")
        # word_timestamps=True is crucial. beam_size default is 5.
        raw_segments, info = whisper_model.transcribe(audio_path, word_timestamps=True, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
        
        full_transcript_text = ""
        all_word_segments_detail = []
        processed_segments = [] # Store {text, start, end} for easier clip generation

        st.info(f"Detected language: {info.language} (Confidence: {info.language_probability:.2f})")

        for seg_idx, segment in enumerate(raw_segments):
            full_transcript_text += segment.text + " "
            processed_segments.append({
                "id": seg_idx,
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end
            })
            if segment.words:
                for word_info in segment.words:
                    all_word_segments_detail.append({
                        "word": word_info.word,
                        "start": word_info.start,
                        "end": word_info.end,
                        "probability": word_info.probability
                    })
        st.success("Transcription complete.")
        return full_transcript_text.strip(), all_word_segments_detail, processed_segments
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None, None, None

def pre_analyze_video_for_faces(video_path, face_cascade, sample_interval_sec, video_duration):
    """Pre-analyzes the entire video for face detections at specified intervals."""
    if not face_cascade:
        st.warning("Face cascade not loaded, skipping face pre-analysis.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.warning("Could not open video for face pre-analysis.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.warning("Video FPS is 0, cannot perform timed face analysis.")
        cap.release()
        return []
        
    face_detections_timeline = []
    total_frames_to_sample = int(video_duration / sample_interval_sec)
    
    st.write("Pre-analyzing video for faces...")
    progress_bar_faces = st.progress(0)

    for i in range(total_frames_to_sample):
        timestamp_sec = i * sample_interval_sec
        frame_number = int(timestamp_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Adjust detection parameters for potentially better recall on diverse footage
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
            if len(faces) > 0:
                # Store all detected faces at this timestamp, could refine later
                for face_roi in faces:
                    face_detections_timeline.append({"timestamp": timestamp_sec, "roi": tuple(face_roi)}) # (x, y, w, h)
            progress_bar_faces.progress((i + 1) / total_frames_to_sample)
        else: # Reached end of video or error reading frame
            if timestamp_sec < video_duration - sample_interval_sec : # if not near the end
                 st.debug(f"Could not read frame at {timestamp_sec:.2f}s for face analysis.")

    progress_bar_faces.empty()
    cap.release()
    st.info(f"Face pre-analysis complete. Found {len(face_detections_timeline)} face instances across samples.")
    return face_detections_timeline


def create_potential_clips_from_whisper_segments(whisper_segments, min_len_sec, max_len_sec, video_duration):
    """
    Creates potential clip segments by combining consecutive Whisper segments.
    Whisper segments are often good semantic chunks.
    """
    potential_clips = []
    if not whisper_segments:
        return []

    for i in range(len(whisper_segments)):
        current_segments_text = []
        current_start_time = whisper_segments[i]['start']
        current_end_time = whisper_segments[i]['end']
        current_segments_text.append(whisper_segments[i]['text'])

        # Try forming a clip starting with just this segment
        duration = current_end_time - current_start_time
        if min_len_sec <= duration <= max_len_sec:
            potential_clips.append({
                "text": " ".join(current_segments_text),
                "start_time": current_start_time,
                "end_time": min(current_end_time, video_duration), # Ensure not exceeding video
                "duration": min(current_end_time, video_duration) - current_start_time
            })

        # Try combining with subsequent segments
        for j in range(i + 1, len(whisper_segments)):
            # Check for excessive gap between segments, indicating a new logical part
            if whisper_segments[j]['start'] - current_end_time > 3.0: # If gap > 3s, probably don't combine
                break

            new_end_time = whisper_segments[j]['end']
            new_duration = new_end_time - current_start_time
            
            if new_duration > max_len_sec:
                break # Stop adding segments if it exceeds max length

            current_segments_text.append(whisper_segments[j]['text'])
            current_end_time = new_end_time
            
            if min_len_sec <= new_duration <= max_len_sec:
                potential_clips.append({
                    "text": " ".join(current_segments_text),
                    "start_time": current_start_time,
                    "end_time": min(current_end_time, video_duration),
                    "duration": min(current_end_time, video_duration) - current_start_time
                })
    
    # Deduplicate (exact time overlaps are unlikely but text might be, sort by start time and text)
    # A simpler deduplication: if start and end times are very close, keep one.
    if not potential_clips: return []
    
    unique_clips = []
    seen_clips_by_time = set()
    for clip in sorted(potential_clips, key=lambda x: (x['start_time'], x['end_time'])):
        time_key = (round(clip['start_time'], 1), round(clip['end_time'], 1))
        if time_key not in seen_clips_by_time:
            unique_clips.append(clip)
            seen_clips_by_time.add(time_key)
    
    return unique_clips


def score_clip_segment(segment_text, segment_start, segment_end, segment_duration,
                       vader_analyzer, viral_keywords_list, pre_analyzed_faces,
                       weights):
    """Scores a text segment for 'virality' cues and face presence."""
    score = 0
    reasons = []

    # 1. Sentiment Analysis
    vs = vader_analyzer.polarity_scores(segment_text)
    sentiment_score_val = vs['compound']
    score += abs(sentiment_score_val) * weights['sentiment']
    if sentiment_score_val > 0.6: reasons.append(f"Strong Positive Sentiment ({sentiment_score_val:.2f})")
    elif sentiment_score_val > 0.2: reasons.append(f"Positive Sentiment ({sentiment_score_val:.2f})")
    if sentiment_score_val < -0.4: reasons.append(f"Negative/Controversial Sentiment ({sentiment_score_val:.2f})")
    elif sentiment_score_val < -0.1: reasons.append(f"Slightly Negative Sentiment ({sentiment_score_val:.2f})")

    # 2. Keyword Spotting
    text_lower = segment_text.lower()
    keyword_hits = 0
    hit_keywords = []
    for keyword in viral_keywords_list:
        if keyword.lower() in text_lower:
            keyword_hits +=1
            if keyword.lower() not in hit_keywords: hit_keywords.append(keyword.lower())
    if keyword_hits > 0:
        score += keyword_hits * weights['keyword']
        reasons.append(f"Keywords: {', '.join(hit_keywords)} ({keyword_hits})")

    # 3. Question Detection
    if '?' in segment_text:
        score += weights['question']
        reasons.append("Contains Question(s)")

    # 4. Exclamations
    num_exclamations = segment_text.count('!')
    if num_exclamations > 0:
        score += min(num_exclamations, 3) * weights['exclamation'] # Cap bonus from too many exclamations
        reasons.append(f"Contains Exclamation(s) ({num_exclamations})")
    
    # 5. Hook Quality (first few words / sentence)
    first_sentence_match = re.match(r"(.*?[\.?!])", segment_text) # Crude first sentence
    hook_text = first_sentence_match.group(1) if first_sentence_match else segment_text[:max(50, len(segment_text)//3)] # Approx first 50 chars or 1/3rd
    hook_text_lower = hook_text.lower()
    hook_keywords = 0
    for keyword in viral_keywords_list: # Check viral keywords in hook
        if keyword.lower() in hook_text_lower:
            hook_keywords +=1
    if hook_keywords > 0:
        score += hook_keywords * weights['hook']
        reasons.append(f"Strong Hook Element ({hook_keywords} kw)")
    if '?' in hook_text:
        score += weights['hook'] * 0.5 # Extra for question in hook
        if "Strong Hook Element" not in str(reasons) and "Contains Question(s)" not in str(reasons) : reasons.append("Question in Hook")


    # 6. Face Presence and Quality (from pre-analyzed data)
    clip_face_rois = []
    num_face_samples_in_clip_range = 0
    num_face_samples_with_face = 0

    # Estimate number of samples that *should* fall in this clip's duration
    # (this is an approximation as sample interval is fixed)
    estimated_samples_in_duration = max(1, int(segment_duration / FACE_ANALYSIS_SAMPLE_INTERVAL_SEC))

    if pre_analyzed_faces:
        for face_detection in pre_analyzed_faces:
            if segment_start <= face_detection['timestamp'] < segment_end:
                num_face_samples_in_clip_range +=1 # Counts detection events, not samples
                clip_face_rois.append(face_detection['roi'])
        
        # Count unique timestamps with faces in range
        unique_face_timestamps_in_clip = set(fd['timestamp'] for fd in pre_analyzed_faces if segment_start <= fd['timestamp'] < segment_end)
        num_face_samples_with_face = len(unique_face_timestamps_in_clip)


    best_face_roi_for_clip = None
    if num_face_samples_with_face > 0:
        face_presence_ratio = num_face_samples_with_face / estimated_samples_in_duration
        face_presence_percent = min(face_presence_ratio * 100, 100) # Cap at 100%

        score += face_presence_percent * weights['face_presence_bonus_factor'] # e.g. 100% presence * 0.2
        reasons.append(f"Face(s) Detected ({face_presence_percent:.0f}% of clip samples)")

        # Determine a representative ROI for smart crop: e.g., average of largest faces
        # Simple approach: use ROI from the middle-most detection, or average if many
        sorted_clip_rois = sorted(clip_face_rois, key=lambda r: r[2]*r[3], reverse=True) # Sort by area
        if sorted_clip_rois:
            best_face_roi_for_clip = sorted_clip_rois[0] # Take largest face in samples
    else:
        # Penalize if no face and text isn't super strong otherwise (optional)
        if sentiment_score_val < 0.3 and keyword_hits == 0:
             score *= weights['no_face_penalty_factor'] # e.g. 0.8
             reasons.append("Low engagement cues & no face")


    # 7. Clip Duration Sweet Spot (slight preference for mid-range of allowed, e.g. 20-45s)
    if 20 <= segment_duration <= 45:
        score += weights['duration_sweet_spot']
        reasons.append("Good Duration (20-45s)")
    elif segment_duration < 15:
        score *= 0.9 # Slight penalty for very short
    
    return score, reasons, best_face_roi_for_clip


def reframe_and_export_clip(original_video_path, start_time, end_time, output_path,
                            crop_mode="center", face_roi_abs=None, original_dims=(1920,1080),
                            video_codec="libx264", audio_codec="aac", preset="ultrafast", threads=4):
    try:
        with VideoFileClip(original_video_path) as video:
            # Ensure subclip times are within video duration
            start_time = max(0, start_time)
            end_time = min(video.duration, end_time)
            if start_time >= end_time:
                st.warning(f"Invalid time range for clip: {start_time}-{end_time}. Skipping.")
                return None

            subclip = video.subclip(start_time, end_time)
            
            actual_original_w, actual_original_h = subclip.size if subclip.size[0] > 0 else original_dims

            # Target dimensions for 9:16
            # Assume we want to maintain original height when cropping from landscape
            # or maintain original width if source is already portrait-ish
            if actual_original_w > actual_original_h: # Landscape or square
                target_h_crop = actual_original_h
                target_w_crop = int(target_h_crop * TARGET_ASPECT_RATIO)
                if target_w_crop > actual_original_w : # If calculated width is too wide (e.g. source is almost 9:16)
                    target_w_crop = actual_original_w
                    target_h_crop = int(target_w_crop / TARGET_ASPECT_RATIO)
            else: # Portrait or square
                target_w_crop = actual_original_w
                target_h_crop = int(target_w_crop / TARGET_ASPECT_RATIO)
                if target_h_crop > actual_original_h:
                    target_h_crop = actual_original_h
                    target_w_crop = int(target_h_crop * TARGET_ASPECT_RATIO)


            x1, y1, x2, y2 = 0, 0, 0, 0 # Crop box coordinates

            if crop_mode == "smart_static" and face_roi_abs:
                fx, fy, fw, fh = face_roi_abs # Face ROI in original full frame coordinates
                
                # Calculate center of the face
                face_center_x = fx + fw / 2
                face_center_y = fy + fh / 2

                # Attempt to center the crop box (target_w_crop x target_h_crop) around the face center
                x1 = face_center_x - target_w_crop / 2
                y1 = face_center_y - target_h_crop / 2 # Vertical centering too
                
                # Ensure crop box stays within original video boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                
                # Adjust if crop box exceeds right/bottom boundary
                if x1 + target_w_crop > actual_original_w:
                    x1 = actual_original_w - target_w_crop
                if y1 + target_h_crop > actual_original_h:
                    y1 = actual_original_h - target_h_crop
                
                # Recalculate x1,y1 to ensure they are not negative after adjustment for other boundary
                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = x1 + target_w_crop
                y2 = y1 + target_h_crop
                st.caption(f"Smart crop for {os.path.basename(output_path)} using face at ({fx},{fy}) -> crop box ({x1},{y1}) to ({x2},{y2})")

            else: # Default to center crop
                x_center = actual_original_w / 2
                y_center = actual_original_h / 2
                x1 = x_center - target_w_crop / 2
                x2 = x_center + target_w_crop / 2
                y1 = y_center - target_h_crop / 2
                y2 = y_center + target_h_crop / 2
                st.caption(f"Center crop for {os.path.basename(output_path)}")
            
            # Ensure coordinates are integers and valid
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            final_crop_width = x2 - x1
            final_crop_height = y2 - y1

            if not (x1 >=0 and y1 >=0 and x2 <= actual_original_w and y2 <= actual_original_h and final_crop_width > 0 and final_crop_height > 0):
                st.warning(f"Invalid crop coordinates for {os.path.basename(output_path)}. Defaulting to full frame or safer crop.")
                # Fallback: try to use target aspect ratio from center without going out of bounds
                if actual_original_w / actual_original_h > TARGET_ASPECT_RATIO: # wider than target
                    final_crop_height = actual_original_h
                    final_crop_width = int(final_crop_height * TARGET_ASPECT_RATIO)
                else: # taller than target
                    final_crop_width = actual_original_w
                    final_crop_height = int(final_crop_width / TARGET_ASPECT_RATIO)
                
                x1 = (actual_original_w - final_crop_width) // 2
                y1 = (actual_original_h - final_crop_height) // 2


            # MoviePy's crop is (x1, y1, x2, y2) but its fx.crop wants width and height from x1,y1
            cropped_clip = subclip.fx(VideoFileClip.crop, x1=x1, y1=y1, width=final_crop_width, height=final_crop_height)
            
            # Resize to a standard short-form dimension if needed, e.g., 1080 width for 9:16
            # Example: if target is 1080x1920
            # final_w_render, final_h_render = 1080, 1920
            # if cropped_clip.w > cropped_clip.h: # if it's landscape after crop (shouldn't be for 9:16)
            #    final_w_render, final_h_render = 1920, 1080 
            #
            # if cropped_clip.w != final_w_render:
            #    cropped_clip = cropped_clip.resize(width=final_w_render)


            # For simplicity, we'll assume the crop itself achieved the aspect ratio.
            # Further resizing can be added if strict output dimensions like 1080x1920 are required.

            cropped_clip.write_videofile(output_path, 
                                         codec=video_codec, 
                                         audio_codec=audio_codec, 
                                         preset=preset, 
                                         threads=threads,
                                         logger=None) # Use logger='bar' for progress bar in console
            st.success(f"Successfully created clip: {os.path.basename(output_path)}")
        return output_path
    except Exception as e:
        st.error(f"Error creating clip {os.path.basename(output_path)}: {e}")
        # import traceback
        # st.error(traceback.format_exc()) # For more detailed debugging if needed
        return None
    finally:
        # Ensure clips are closed to free resources
        if 'subclip' in locals() and subclip: subclip.close()
        if 'cropped_clip' in locals() and cropped_clip: cropped_clip.close()


# --- Main Application Logic ---
def run_processing_pipeline(source_video_path, whisper_model, face_cascade, vader_analyzer,
                            settings, session_temp_dir):
    """Orchestrates the main processing steps."""
    
    try:
        with VideoFileClip(source_video_path) as video_clip_info:
            original_dims = video_clip_info.size
            video_duration = video_clip_info.duration
        st.session_state['source_video_details'] = {
            "path": source_video_path,
            "dimensions": original_dims,
            "duration": video_duration,
            "filename": os.path.basename(source_video_path)
        }
    except Exception as e:
        st.error(f"Could not read video properties from {source_video_path}: {e}")
        return None

    if video_duration > 1800: # 30 minutes
         st.warning(f"⚠️ Long video detected ({video_duration/60:.1f} mins)! Processing will be very slow and may hit resource limits.")
    elif video_duration > 600: # 10 minutes
         st.warning(f"Long video ({video_duration/60:.1f} mins). Processing may take some time.")


    # 1. Extract Audio
    audio_filename = f"{os.path.splitext(os.path.basename(source_video_path))[0]}_audio.wav"
    temp_audio_path = os.path.join(session_temp_dir, audio_filename)
    extracted_audio_path = extract_audio_from_video(source_video_path, temp_audio_path)
    if not extracted_audio_path: return None

    # 2. Transcribe Audio
    full_transcript, word_segments_detail, whisper_processed_segments = transcribe_audio_with_whisper(extracted_audio_path, whisper_model)
    if not whisper_processed_segments:
        st.error("Transcription failed or produced no segments.")
        return None
    
    with st.expander("Full Transcript Preview (First 1000 chars)", expanded=False):
        st.markdown(f"> {full_transcript[:1000]}...")

    # 3. Pre-analyze video for faces (once for the whole video)
    all_face_detections_timeline = pre_analyze_video_for_faces(source_video_path, face_cascade, FACE_ANALYSIS_SAMPLE_INTERVAL_SEC, video_duration)

    # 4. Create Potential Clip Segments
    st.info("Identifying potential clip segments from transcript...")
    # Using whisper_processed_segments for clip generation is more robust
    potential_clips_from_segments = create_potential_clips_from_whisper_segments(
        whisper_processed_segments, settings['min_clip_len'], settings['max_clip_len'], video_duration
    )
    
    if not potential_clips_from_segments:
        st.warning("No potential clip segments found based on transcript and duration settings.")
        return None
    st.info(f"Found {len(potential_clips_from_segments)} potential segments. Now scoring them...")

    # 5. Score Segments and Select Top N
    scored_clips_data = []
    scoring_weights = {
        'sentiment': settings['score_weight_sentiment'],
        'keyword': settings['score_weight_keyword'],
        'question': settings['score_weight_question'],
        'exclamation': settings['score_weight_exclamation'],
        'hook': settings['score_weight_hook'],
        'face_presence_bonus_factor': settings['score_weight_face_factor'],
        'no_face_penalty_factor': 0.8, # Can be made configurable
        'duration_sweet_spot': settings['score_weight_duration_bonus']
    }

    with st.spinner(f"Analyzing and scoring {len(potential_clips_from_segments)} segments..."):
        for i, p_clip in enumerate(potential_clips_from_segments):
            score, reasons, best_face_roi = score_clip_segment(
                p_clip['text'], p_clip['start_time'], p_clip['end_time'], p_clip['duration'],
                vader_analyzer, settings['viral_keywords'], all_face_detections_timeline,
                scoring_weights
            )
            p_clip['score'] = score
            p_clip['reasons'] = reasons
            p_clip['face_roi_for_smart_crop'] = best_face_roi
            scored_clips_data.append(p_clip)
            if (i + 1) % 20 == 0 or i == len(potential_clips_from_segments) -1 :
                st.text(f"Scored {i+1}/{len(potential_clips_from_segments)} segments...") # Temporary feedback

    # Sort by score (descending)
    sorted_clips = sorted(scored_clips_data, key=lambda x: x['score'], reverse=True)
    
    if not sorted_clips:
        st.warning("No clips scored high enough or found after filtering.")
        return None
        
    return sorted_clips # Return all sorted, selection happens in main UI loop

def main_app_ui():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.markdown("Automatically discover and create engaging short vertical clips from your long-form videos. Enhanced for better analysis and customization.")

    # Ensure base temp directory exists
    os.makedirs(TEMP_DIR_BASE, exist_ok=True)
    
    # --- Sidebar Setup ---
    st.sidebar.header("⚙️ Configuration")

    # Model Loading
    whisper_model_size = st.sidebar.selectbox("Whisper Model:", ["tiny.en", "base.en", "small.en"], index=0,
                                             help="Smaller models are faster but less accurate. `.en` models are English-only.")
    
    # Load essential resources
    # These are cached, so they only load once or when parameters change.
    whisper_model = load_whisper_model(whisper_model_size)
    face_cascade = get_face_cascade()
    vader_analyzer = get_vader_analyzer()


    st.sidebar.subheader("Video Input")
    video_source_option = st.sidebar.radio("Video Source:", ("Upload Video", "YouTube Link"))
    
    input_video_path = None
    source_video_name_for_files = "uploaded_video"

    if video_source_option == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload 16:9 video (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_file:
            # Save to a persistent temp location for this session
            ext = os.path.splitext(uploaded_file.name)[1]
            # Sanitize uploaded file name
            sane_uploaded_name = re.sub(r'[^\w\s.-]', '', os.path.splitext(uploaded_file.name)[0]).strip().replace(' ', '_')
            sane_uploaded_name = re.sub(r'[-_]+', '_', sane_uploaded_name)
            temp_upload_filename = f"upload_{sane_uploaded_name[:30]}_{uuid.uuid4().hex[:6]}{ext}"

            input_video_path = os.path.join(TEMP_DIR_BASE, temp_upload_filename)
            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.video(input_video_path)
            source_video_name_for_files = os.path.splitext(temp_upload_filename)[0]
            st.session_state['source_video_details'] = {"path": input_video_path, "filename": uploaded_file.name}


    elif video_source_option == "YouTube Link":
        youtube_url = st.sidebar.text_input("YouTube Video URL:", st.session_state.get('last_yt_url', ''))
        if youtube_url:
            st.session_state['last_yt_url'] = youtube_url # Remember for next time
            if st.sidebar.button("Fetch YouTube Video"):
                with st.spinner("Downloading video... This may take a few moments."):
                    downloaded_path = download_youtube_video(youtube_url, TEMP_DIR_BASE)
                if downloaded_path:
                    input_video_path = downloaded_path
                    st.session_state['source_video_details'] = {"path": input_video_path, "filename": os.path.basename(input_video_path)}
                    # yt_dlp gives progress, sidebar.video can be after download
                    st.sidebar.success("Video ready.")
                    st.sidebar.video(input_video_path)
                    source_video_name_for_files = os.path.splitext(os.path.basename(input_video_path))[0]

            # If already downloaded and URL matches, use it
            elif 'downloaded_yt_info' in st.session_state and st.session_state.downloaded_yt_info.get('url') == youtube_url:
                 if os.path.exists(st.session_state.downloaded_yt_info['path']):
                    input_video_path = st.session_state.downloaded_yt_info['path']
                    st.sidebar.info("Using previously fetched video for this URL.")
                    st.sidebar.video(input_video_path)
                    source_video_name_for_files = os.path.splitext(os.path.basename(input_video_path))[0]
                    if 'source_video_details' not in st.session_state or st.session_state.source_video_details.get('path') != input_video_path:
                         st.session_state.source_video_details = {"path": input_video_path, "filename": os.path.basename(input_video_path)}


    st.sidebar.subheader("Clip Generation Settings")
    min_clip_len = st.sidebar.slider("Min Clip Length (s):", 5, 60, 10)
    max_clip_len = st.sidebar.slider("Max Clip Length (s):", 15, 180, 59, help="Max for TikTok is often 60-90s, YT Shorts 60s.")
    preferred_crop_mode = st.sidebar.selectbox("Crop Mode:", ["Smart Static Crop (Face Focus)", "Center Crop"], index=0)
    
    st.sidebar.subheader("Advanced: Scoring Weights")
    with st.sidebar.expander("Customize Virality Scoring"):
        score_weight_sentiment = st.number_input("Sentiment Impact", 0.0, 50.0, 15.0, 0.5)
        score_weight_keyword = st.number_input("Keyword Impact", 0.0, 50.0, 10.0, 0.5)
        score_weight_question = st.number_input("Question Bonus", 0.0, 50.0, 15.0, 0.5)
        score_weight_exclamation = st.number_input("Exclamation Bonus", 0.0, 30.0, 3.0, 0.5)
        score_weight_hook = st.number_input("Hook Quality Bonus", 0.0, 50.0, 20.0, 0.5)
        score_weight_face_factor = st.number_input("Face Presence Factor (x % presence)", 0.0, 1.0, 0.2, 0.05)
        score_weight_duration_bonus = st.number_input("Mid-Duration Clip Bonus", 0.0, 30.0, 10.0, 0.5)

    st.sidebar.subheader("Keywords for Virality")
    viral_keywords_text = st.sidebar.text_area("Viral Keywords (comma-separated):", ", ".join(DEFAULT_VIRAL_KEYWORDS), height=100)
    custom_viral_keywords = [k.strip().lower() for k in viral_keywords_text.split(',') if k.strip()]

    processing_settings = {
        'min_clip_len': min_clip_len,
        'max_clip_len': max_clip_len,
        'crop_mode': preferred_crop_mode,
        'viral_keywords': custom_viral_keywords,
        'score_weight_sentiment': score_weight_sentiment,
        'score_weight_keyword': score_weight_keyword,
        'score_weight_question': score_weight_question,
        'score_weight_exclamation': score_weight_exclamation,
        'score_weight_hook': score_weight_hook,
        'score_weight_face_factor': score_weight_face_factor,
        'score_weight_duration_bonus': score_weight_duration_bonus
    }

    st.sidebar.markdown("---")
    if st.sidebar.button("⚠️ Clear Caches & Temp Files"):
        clear_app_cache_and_temp()
        st.experimental_rerun() # Rerun to reflect cleared state

    # --- Main Content Area ---
    if not input_video_path:
        st.info("⬅️ Please upload a video or provide a YouTube link via the sidebar to begin.")
    elif not os.path.exists(input_video_path):
        st.error(f"The selected video path seems invalid or file is missing: {input_video_path}. Please re-select or re-download.")
    elif not whisper_model or not face_cascade or not vader_analyzer:
        st.error("🚫 One or more essential models (Whisper, Face Cascade, VADER) failed to load. Processing cannot start. Check error messages above.")
    else:
        st.header("🎬 Process Video")
        st.markdown(f"Ready to process: **{st.session_state.get('source_video_details',{}).get('filename','Selected Video')}**")
        
        if st.button("🚀 Find & Generate Viral Clips!", type="primary", use_container_width=True):
            if 'final_clips_info' in st.session_state: # Clear previous results before new run
                del st.session_state['final_clips_info']

            # Use a temporary directory for this specific processing session's intermediate files
            with tempfile.TemporaryDirectory(prefix="viralclip_session_", dir=TEMP_DIR_BASE) as session_temp_dir:
                st.markdown(f"Processing using session temp: `{session_temp_dir}`") # For debug

                all_sorted_clips = run_processing_pipeline(
                    input_video_path, whisper_model, face_cascade, vader_analyzer,
                    processing_settings, session_temp_dir
                )

                if all_sorted_clips:
                    top_clips_to_generate = all_sorted_clips[:MAX_CLIP_SUGGESTIONS_TO_PROCESS]
                    st.subheader(f"🏆 Top {len(top_clips_to_generate)} Potential Clips (max {MAX_CLIP_SUGGESTIONS_TO_PROCESS} shown for generation):")
                    
                    generated_clips_info = []
                    for i, clip_info in enumerate(top_clips_to_generate):
                        clip_base_name = f"clip_{i+1}_{source_video_name_for_files[:30]}_{uuid.uuid4().hex[:4]}.mp4"
                        output_clip_path = os.path.join(TEMP_DIR_BASE, clip_base_name) # Save final clips to persistent temp

                        actual_crop_mode = "center"
                        if processing_settings['crop_mode'] == "Smart Static Crop (Face Focus)" and clip_info.get('face_roi_for_smart_crop'):
                            actual_crop_mode = "smart_static"
                        
                        exp_title = f"Clip #{i+1} (Score: {clip_info['score']:.2f}) | {clip_info['duration']:.1f}s | {clip_info['start_time']:.1f}s - {clip_info['end_time']:.1f}s"
                        with st.expander(exp_title, expanded=(i==0)): # Expand first clip by default
                            st.markdown(f"**Text:** *{clip_info['text']}*")
                            st.markdown(f"**Scoring Reasons:** _{', '.join(clip_info['reasons'])}_")
                            if clip_info.get('face_roi_for_smart_crop'):
                                st.caption(f"Face ROI for smart crop: {clip_info['face_roi_for_smart_crop']}")
                            
                            with st.spinner(f"Generating vertical video for Clip #{i+1}... This can take time."):
                                generated_path = reframe_and_export_clip(
                                    st.session_state['source_video_details']['path'], # Use path from session state
                                    clip_info['start_time'],
                                    clip_info['end_time'],
                                    output_clip_path,
                                    crop_mode=actual_crop_mode,
                                    face_roi_abs=clip_info.get('face_roi_for_smart_crop'),
                                    original_dims=st.session_state['source_video_details']['dimensions']
                                )
                            if generated_path and os.path.exists(generated_path):
                                st.video(generated_path)
                                with open(generated_path, "rb") as fp_video:
                                    st.download_button(
                                        label=f"Download Clip #{i+1}",
                                        data=fp_video.read(), # Read bytes for download
                                        file_name=os.path.basename(generated_path),
                                        mime="video/mp4"
                                    )
                                generated_clips_info.append({"path": generated_path, "info": clip_info})
                            else:
                                st.error(f"Failed to generate Clip #{i+1}.")
                    
                    st.session_state['final_clips_info'] = generated_clips_info
                    st.success("All selected clips processed!")
                    if len(all_sorted_clips) > MAX_CLIP_SUGGESTIONS_TO_PROCESS:
                        st.info(f"{len(all_sorted_clips) - MAX_CLIP_SUGGESTIONS_TO_PROCESS} more clips were identified but not processed to save resources. Consider adjusting scoring or processing more clips if desired (by changing `MAX_CLIP_SUGGESTIONS_TO_PROCESS` in code).")

                else:
                    st.error("💥 Clip generation pipeline did not produce any viable clips. Try adjusting settings or check logs if any.")
        
        # Display previously generated clips if any in session
        elif 'final_clips_info' in st.session_state and st.session_state['final_clips_info']:
            st.subheader("Previously Generated Clips (this session):")
            for i, clip_data in enumerate(st.session_state['final_clips_info']):
                clip_info = clip_data['info']
                exp_title = f"Clip #{i+1} (Score: {clip_info['score']:.2f}) | {clip_info['duration']:.1f}s | {clip_info['start_time']:.1f}s - {clip_info['end_time']:.1f}s"
                with st.expander(exp_title, expanded=False):
                    st.markdown(f"**Text:** *{clip_info['text']}*")
                    st.markdown(f"**Scoring Reasons:** _{', '.join(clip_info['reasons'])}_")
                    if os.path.exists(clip_data['path']):
                        st.video(clip_data['path'])
                        with open(clip_data['path'], "rb") as fp_video_rep:
                            st.download_button(
                                label=f"Download Clip #{i+1}",
                                data=fp_video_rep.read(),
                                file_name=os.path.basename(clip_data['path']),
                                mime="video/mp4",
                                key=f"download_prev_{i}" # Unique key for rerun
                            )
                    else:
                        st.warning(f"Video file for clip #{i+1} not found at {clip_data['path']}. It might have been cleared.")


    st.sidebar.markdown("---")
    st.sidebar.info("Prototyped by an AI Assistant. Processing can be slow, especially for long videos or larger Whisper models on CPU-limited environments.")
    st.sidebar.markdown("Remember to clear cache if you encounter persistent issues or change models frequently.")


if __name__ == "__main__":
    main_app_ui()
