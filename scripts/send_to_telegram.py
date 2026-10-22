import os
import re
import sys
import time
import requests
import subprocess
from typing import Optional, Dict, Any, Tuple

def get_telegram_creds() -> Tuple[Optional[str], Optional[str]]:
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                content = f.read()
                m_tok = re.search(r"TELEGRAM_TOKEN=([^\s\n\"']+)", content)
                m_chat = re.search(r"TELEGRAM_CHAT_ID=([^\s\n\"']+)", content)
                if m_tok and not token:
                    token = m_tok.group(1).strip()
                if m_chat and not chat_id:
                    chat_id = m_chat.group(1).strip()

    if not token or not chat_id:
        zshrc_path = os.path.expanduser("~/.zshrc")
        if os.path.exists(zshrc_path):
            with open(zshrc_path, "r", encoding="utf-8") as f:
                content = f.read()
                m_tok = re.search(r"TELEGRAM_TOKEN=([^\s\n\"']+)", content)
                m_chat = re.search(r"TELEGRAM_CHAT_ID=([^\s\n\"']+)", content)
                if m_tok and not token:
                    token = m_tok.group(1).strip()
                if m_chat and not chat_id:
                    chat_id = m_chat.group(1).strip()
    return token, chat_id

def get_gofile_token() -> Optional[str]:
    token = os.environ.get("GOFILE_TOKEN")
    if not token:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                m_tok = re.search(r"GOFILE_TOKEN=([^\s\n\"']+)", f.read())
                if m_tok:
                    token = m_tok.group(1).strip()
    return token

def upload_to_gofile(file_path: str, token: Optional[str] = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Uploads a file to Gofile.io and returns file metadata including downloadPage."""
    if not os.path.exists(file_path):
        print(f"❌ File does not exist for Gofile upload: {file_path}")
        return None

    token = token or get_gofile_token()
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"🌐 Preparing Gofile upload ({file_size_mb:.2f} MB)...")

    for attempt in range(1, max_retries + 1):
        try:
            # 1. Fetch available upload server
            srv_resp = requests.get("https://api.gofile.io/servers", timeout=15).json()
            if srv_resp.get("status") != "ok" or not srv_resp.get("data", {}).get("servers"):
                print(f"⚠️ Gofile servers query attempt {attempt} failed: {srv_resp}")
                time.sleep(2)
                continue

            servers = srv_resp["data"]["servers"]
            server_name = servers[0]["name"]
            upload_url = f"https://{server_name}.gofile.io/contents/uploadfile"

            # 2. Upload file
            print(f"🚀 Uploading to Gofile server '{server_name}'...")
            data = {"token": token} if token else {}
            with open(file_path, "rb") as f:
                up_resp = requests.post(
                    upload_url,
                    files={"file": f},
                    data=data,
                    timeout=300,
                ).json()

            if up_resp.get("status") == "ok":
                data_res = up_resp.get("data", {})
                dl_page = data_res.get("downloadPage")
                print(f"✅ Gofile upload successful! Download URL: {dl_page}")
                return data_res
            else:
                print(f"⚠️ Gofile upload attempt {attempt} returned non-ok: {up_resp}")
        except Exception as e:
            print(f"⚠️ Gofile upload attempt {attempt} error: {e}")
        time.sleep(3)

    print("❌ All Gofile upload attempts failed.")
    return None

def create_preview_video(video_path: str, target_height: int = 320) -> str:
    """Creates a fast, highly compressed preview video for instant mobile Telegram viewing."""
    base, ext = os.path.splitext(video_path)
    preview_path = f"{base}_preview_{target_height}p.mp4"
    print(f"🎬 Creating {target_height}p preview: {preview_path}...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"scale=-2:'min({target_height},ih)'",
        "-c:v", "libx264", "-crf", "32",
        "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
        preview_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new_size_mb = os.path.getsize(preview_path) / (1024 * 1024)
    print(f"✅ Preview created ({new_size_mb:.2f} MB)")
    return preview_path

def send_video(
    video_path: str,
    caption: str = "",
    target_height: int = 320,
    max_direct_size_mb: float = 45.0,
    force_gofile: bool = False,
    gofile_fallback: bool = True,
) -> Dict[str, Any]:
    """Sends video to Telegram with intelligent file size routing and Gofile upload support.
    
    If file size <= max_direct_size_mb and not force_gofile:
        - Sends preview if file > 3MB for speed, or full video.
    If file size > max_direct_size_mb or force_gofile:
        - Uploads original uncompressed video to Gofile.io.
        - Generates a lightweight 320p preview.
        - Sends the 320p preview to Telegram with the Gofile download link in the caption.
    """
    token, chat_id = get_telegram_creds()
    if not token or not chat_id:
        raise ValueError("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not found in environment or ~/.zshrc")
    
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Original video size: {file_size_mb:.2f} MB")
    
    gofile_data = None
    gofile_url = None
    
    # Check if we should upload to Gofile
    needs_gofile = force_gofile or (file_size_mb > max_direct_size_mb)
    if needs_gofile and gofile_fallback:
        print(f"📦 File size ({file_size_mb:.2f} MB) exceeds Telegram direct limit ({max_direct_size_mb} MB) or gofile requested.")
        gofile_data = upload_to_gofile(video_path)
        if gofile_data:
            gofile_url = gofile_data.get("downloadPage")
    
    # Construct caption with Gofile download link if available
    final_caption = caption
    if gofile_url:
        link_line = f"\n\n📥 Full Video (1080p, {file_size_mb:.1f}MB):\n{gofile_url}"
        final_caption = f"{caption}{link_line}".strip()
    
    # Decide which file to upload to Telegram
    upload_path = video_path
    if file_size_mb > 3.0 or needs_gofile:
        upload_path = create_preview_video(video_path, target_height=target_height)

    url = f"https://api.telegram.org/bot{token}/sendVideo"
    print(f"Uploading {upload_path} to Telegram chat {chat_id} via IPv4 curl...")
    
    curl_cmd = [
        "curl", "-4", "-s", "-X", "POST", url,
        "-F", f"chat_id={chat_id}",
        "-F", f"caption={final_caption}",
        "-F", f"video=@{upload_path}"
    ]
    res = subprocess.run(curl_cmd, capture_output=True, text=True)
    if res.returncode == 0 and '"ok":true' in res.stdout:
        print("✅ Successfully sent video to Telegram!")
        return {
            "status": "ok",
            "gofile_url": gofile_url,
            "uploaded_file": upload_path,
            "telegram_response": res.stdout,
        }
    else:
        print(f"❌ Failed to send video: {res.stderr or res.stdout}")
        raise RuntimeError(f"Telegram upload failed: {res.stderr or res.stdout}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python send_to_telegram.py <path_to_video> [caption] [height_p] [--gofile]")
        sys.exit(1)
    
    v_path = sys.argv[1]
    cap = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "🎬 Final Rendered Video"
    h_p = 320
    force_gf = "--gofile" in sys.argv
    for arg in sys.argv[2:]:
        if arg.isdigit():
            h_p = int(arg)
    
    send_video(v_path, cap, target_height=h_p, force_gofile=force_gf)
