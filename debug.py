import os
import sys

# Try to import MoviePy with compatibility for v1.0 and v2.0
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    try:
        # Fallback for MoviePy v2.0+
        from moviepy import VideoFileClip
    except ImportError:
        print("\n❌ CRITICAL ERROR: 'moviepy' library is missing.")
        print("👉 Please install it by running the following command:\n")
        print("    pip install moviepy\n")
        sys.exit(1)

# List your reference videos here
video_files = [
    "data/custom/a.mp4",
    "data/custom/b.mp4",
    "data/custom/c.mp4",
    "data/custom/d.mp4"
]

print("🔄 Starting Video Repair for Web Compatibility...")

for file_path in video_files:
    if os.path.exists(file_path):
        try:
            print(f"Processing: {file_path}")
            
            # 1. Rename original to keep a backup
            backup_path = file_path.replace(".mp4", "_backup.mp4")
            if not os.path.exists(backup_path):
                os.rename(file_path, backup_path)
            
            # 2. Load the backup
            clip = VideoFileClip(backup_path)
            
            # 3. Write new file with H.264 codec (Web Standard)
            # preset='fast' speeds it up, crf=22 maintains quality
            clip.write_videofile(
                file_path, 
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True,
                logger=None
            )
            
            print(f"✅ Fixed: {file_path}")
            clip.close()
            
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
            # Restore backup if failed
            if os.path.exists(backup_path) and not os.path.exists(file_path):
                os.rename(backup_path, file_path)
    else:
        print(f"⚠️ Not found: {file_path}")

print("🎉 Done! Restart your Streamlit app.")