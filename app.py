# app.py
# AI-Powered Classroom Attendance & Lecture Notes Generator
# Production-ready | Hugging Face Spaces & Google Colab compatible

import os
import tempfile
import math
import time
from typing import List, Dict

import gradio as gr
from groq import Groq
from faster_whisper import WhisperModel
from pydub import AudioSegment

# =========================
# Configuration
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY is None or GROQ_API_KEY.strip() == "":
    GROQ_AVAILABLE = False
else:
    GROQ_AVAILABLE = True
    groq_client = Groq(api_key=GROQ_API_KEY)

# CPU-only Whisper model (lightweight)
WHISPER_MODEL_NAME = "base"
whisper_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8"
)

# Editable student list
STUDENTS = [

    "Rao Muhammad Faran Latif",
    "Muhammad Arslan Safdar",
    "Waqas Ahmad",
    "Barira Mazhar",
    "Ubaid Javaid",
    "Sobia Razzaq",
    "Saif Siddique",
    "Abdul Muneeb Bhatti",
    "Tuba Mukhtar",
    "Muhammad Shehzad",
    "Muhammad Waseem",
    "Mahnoor Rafique",
    "Shahid Abbas",
    "Warda Rafique",
    "Muhammad Aqib",
    "Nafeesa Maqsood",
    "Zain Iqbal",
    "Shahzaib",
    "Ali Raza Khan",
    "Abdullah Niaz",
    "Ahsan Mukhtiar",
    "Mahnoor",
    "Ahsin Majeed",
    "Muhammad Yousuf",
    "Muhammad Rouf",
    "Kousar Abbas",
]

# =========================
# Utility Functions
# =========================

def show_error(message: str) -> str:
    return f"❌ {message}"


def split_audio(audio_path: str, chunk_length_ms: int = 5 * 60 * 1000) -> List[str]:
    """Split long audio into manageable chunks"""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    total_chunks = math.ceil(len(audio) / chunk_length_ms)

    for i in range(total_chunks):
        start = i * chunk_length_ms
        end = start + chunk_length_ms
        chunk = audio[start:end]

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(tmp_file.name, format="wav")
        chunks.append(tmp_file.name)

    return chunks


def transcribe_audio(audio_path: str) -> str:
    """Speech-to-text using Whisper"""
    segments, _ = whisper_model.transcribe(audio_path)
    transcription = " ".join([segment.text for segment in segments])
    return transcription.strip()


def generate_notes(transcript: str) -> str:
    """Generate structured lecture notes using Groq LLM"""

    if not GROQ_AVAILABLE:
        return show_error("GROQ_API_KEY is missing. Please add it to your environment variables.")

    prompt = f"""
You are an expert academic note-taking AI.

Transform the following lecture transcript into clear, structured, and student-friendly lecture notes.

STRICT REQUIREMENTS:
- Do not add information not present in the transcript.
- Be concise but complete.
- Use clear headings and bullet points.
- Maintain academic tone.

OUTPUT FORMAT:
Title:
Short Summary:
Key Points (bulleted):
Important Definitions:
Examples (only if present in transcript):
Conclusion:

Lecture Transcript:
\"\"\"
{transcript}
\"\"\"
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1200,
    )

    return response.choices[0].message.content.strip()


# =========================
# Lecture Processing Logic
# =========================

def process_lecture(audio_file):
    if audio_file is None:
        return show_error("Please record or upload an audio file first.")

    try:
        audio_chunks = split_audio(audio_file)
        full_transcript = ""

        for chunk in audio_chunks:
            text = transcribe_audio(chunk)
            full_transcript += " " + text

        full_transcript = full_transcript.strip()

        if full_transcript == "":
            return show_error("Unable to transcribe audio. Please try a clearer recording.")

        notes = generate_notes(full_transcript)
        return notes

    except Exception:
        return show_error("Failed to process the lecture. Please try again with a supported audio format.")


# =========================
# Attendance Logic
# =========================

def init_attendance():
    return {name: "Not Marked" for name in STUDENTS}


def mark_attendance(attendance: Dict[str, str], student: str, status: str):
    attendance[student] = status
    return attendance, render_attendance(attendance)


def render_attendance(attendance: Dict[str, str]) -> str:
    total = len(attendance)
    present = sum(1 for v in attendance.values() if v == "Present")
    absent = sum(1 for v in attendance.values() if v == "Absent")

    rows = []
    for name, status in attendance.items():
        color = "green" if status == "Present" else "red" if status == "Absent" else "gray"
        rows.append(f"- **{name}**: <span style='color:{color}'>{status}</span>")

    summary = f"""
### 📊 Attendance Summary
- Total Students: **{total}**
- Present: **{present}**
- Absent: **{absent}**

### 🧑‍🎓 Student Status
""" + "\n".join(rows)

    return summary


# =========================
# Gradio UI
# =========================

custom_css = """
body {
    background-color: #f5f7fa;
}
.gr-button {
    font-weight: 600;
}
"""

with gr.Blocks(css=custom_css, title="AI Classroom Assistant") as demo:

    gr.Markdown(
        """
        # 🎓 AI-Powered Classroom Assistant
        Generate lecture notes automatically and manage classroom attendance with ease.
        """
    )

    with gr.Tab("🎤 Lecture Notes Generator"):
        gr.Markdown("### Record or Upload a Lecture Audio")

        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Lecture Audio (Mic or File)"
        )

        process_btn = gr.Button("Process Lecture", variant="primary")
        notes_output = gr.Markdown(label="Generated Lecture Notes")

        process_btn.click(
            fn=process_lecture,
            inputs=audio_input,
            outputs=notes_output
        )

    with gr.Tab("🧑‍🎓 Attendance System"):
        gr.Markdown("### Mark Student Attendance")

        attendance_state = gr.State(init_attendance())
        attendance_display = gr.Markdown(render_attendance(init_attendance()))

        for student in STUDENTS:
            with gr.Row():
                gr.Markdown(f"**{student}**")
                present_btn = gr.Button("✅ Present", size="sm")
                absent_btn = gr.Button("❌ Absent", size="sm")

                present_btn.click(
                    fn=lambda a, s=student: mark_attendance(a, s, "Present"),
                    inputs=attendance_state,
                    outputs=[attendance_state, attendance_display],
                )

                absent_btn.click(
                    fn=lambda a, s=student: mark_attendance(a, s, "Absent"),
                    inputs=attendance_state,
                    outputs=[attendance_state, attendance_display],
                )

    gr.Markdown(
        """
        ---
        **Deployment Notes:**
        - Python 3.10+
        - Gradio (latest)
        - faster-whisper
        - groq
        - pydub
        """
    )

demo.launch(
    server_name="0.0.0.0"
)
