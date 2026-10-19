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
    
    # If video is larger than 10MB, compress for speedy Telegram upload
    if file_size_mb > 10.0:
        print("File is larger than 10MB. Optimizing for quick Telegram dispatch...")
        base, ext = os.path.splitext(video_path)
        upload_path = f"{base}_telegram.mp4"

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264", "-crf", "28",
            "-preset", "faster",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            upload_path
        ]
        subprocess.run(cmd, check=True)
        new_size_mb = os.path.getsize(upload_path) / (1024 * 1024)
        print(f"Optimized video size: {new_size_mb:.2f} MB")

    url = f"https://api.telegram.org/bot{token}/sendVideo"
    print(f"Uploading {upload_path} to Telegram chat {chat_id} via IPv4 curl...")
    
    # Use curl with -4 for rock-solid IPv4 network throughput
    curl_cmd = [
        "curl", "-4", "-s", "-X", "POST", url,
        "-F", f"chat_id={chat_id}",
        "-F", f"caption={caption}",
        "-F", f"video=@{upload_path}"
    ]
    res = subprocess.run(curl_cmd, capture_output=True, text=True)
    if res.returncode == 0 and '"ok":true' in res.stdout:
        print("✅ Successfully sent video to Telegram!")
        return res.stdout
    else:
        print(f"❌ Failed to send video: {res.stderr or res.stdout}")
        raise RuntimeError(f"Telegram upload failed: {res.stderr or res.stdout}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python send_to_telegram.py <path_to_video> [caption]")
        sys.exit(1)
    
    v_path = sys.argv[1]
    cap = sys.argv[2] if len(sys.argv) > 2 else "🎬 Final Rendered Video"
    send_video(v_path, cap)
