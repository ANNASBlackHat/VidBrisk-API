import os
import re
import sys
import requests
import subprocess

def get_telegram_creds():
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        zshrc_path = os.path.expanduser("~/.zshrc")
        if os.path.exists(zshrc_path):
            with open(zshrc_path, "r", encoding="utf-8") as f:
                content = f.read()
                m_tok = re.search(r"TELEGRAM_TOKEN=([^\s\n\"']+)", content)
                m_chat = re.search(r"TELEGRAM_CHAT_ID=([^\s\n\"']+)", content)
                if m_tok:
                    token = m_tok.group(1).strip()
                if m_chat:
                    chat_id = m_chat.group(1).strip()
    return token, chat_id

def send_video(video_path: str, caption: str = ""):
    token, chat_id = get_telegram_creds()
    if not token or not chat_id:
        raise ValueError("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not found in environment or ~/.zshrc")
    
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Original video size: {file_size_mb:.2f} MB")
    
    upload_path = video_path
    if file_size_mb > 48.0:
        print("File exceeds Telegram 50MB bot upload limit. Encoding optimized version for Telegram...")
        base, ext = os.path.splitext(video_path)
        upload_path = f"{base}_telegram.mp4"

        # Probe duration
        dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        try:
            dur = float(subprocess.check_output(dur_cmd).decode().strip())
        except Exception:
            dur = 60.0

        # Target 42 MB total to stay safely under 50MB
        target_bitrate_kbps = max(500, int((42 * 8192) / dur) - 128)
        print(f"Targeting bitrate {target_bitrate_kbps}k for {dur:.1f}s video (<45MB)...")

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264", "-b:v", f"{target_bitrate_kbps}k",
            "-maxrate", f"{int(target_bitrate_kbps * 1.2)}k",
            "-bufsize", f"{target_bitrate_kbps * 2}k",
            "-preset", "faster",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            upload_path
        ]
        subprocess.run(cmd, check=True)
        new_size_mb = os.path.getsize(upload_path) / (1024 * 1024)
        print(f"Compressed video size: {new_size_mb:.2f} MB")

    url = f"https://api.telegram.org/bot{token}/sendVideo"
    print(f"Uploading {upload_path} to Telegram chat {chat_id}...")
    
    with open(upload_path, "rb") as video_file:
        files = {"video": video_file}
        data = {
            "chat_id": chat_id,
            "caption": caption,
            "supports_streaming": True,
            "parse_mode": "Markdown"
        }
        resp = requests.post(url, data=data, files=files, timeout=300)
    
    if resp.status_code == 200:
        print("✅ Successfully sent video to Telegram!")
        return resp.json()
    else:
        print(f"❌ Failed to send video: {resp.status_code} {resp.text}")
        resp.raise_for_status()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python send_to_telegram.py <path_to_video> [caption]")
        sys.exit(1)
    
    v_path = sys.argv[1]
    cap = sys.argv[2] if len(sys.argv) > 2 else "🎬 Final Rendered Video"
    send_video(v_path, cap)
