"""
Upload data to Modal Volume before training.
Usage: uv run modal run miniature_fm/sync_to_modal.py
"""

import modal
from pathlib import Path

app = modal.App("miniature-fm-upload")
volume = modal.Volume.from_name("2026-10-19-miniature-fm", create_if_missing=True)

@app.function(volumes={"/vol": volume})
def upload_files(files_data):
    """Upload files to volume"""
    import os

    print(f"Uploading {len(files_data)} files to volume...")

    for local_content, remote_path in files_data:
        # Create parent directory if needed
        remote_full = f"/vol/{remote_path}"
        os.makedirs(os.path.dirname(remote_full), exist_ok=True)

        # Write file
        with open(remote_full, 'wb') as f:
            f.write(local_content)

        print(f"  Uploaded: {remote_path}")

    volume.commit()
    print("Upload complete!")

@app.local_entrypoint()
def main():
    """Upload data to Modal volume"""
    print("Preparing to upload training data...")

    local_dir = Path("volumes/data/intermediates/visual_poetry")

    # Read files into memory
    files_data = []
    for local_path in local_dir.rglob("*"):
        if local_path.is_file():
            relative_path = local_path.relative_to(Path("volumes"))
            remote_path = str(relative_path)

            with open(local_path, 'rb') as f:
                content = f.read()

            files_data.append((content, remote_path))
            print(f"  Prepared: {local_path} ({len(content)} bytes)")

    print(f"\nUploading {len(files_data)} files to Modal volume...")
    upload_files.remote(files_data)
    print("All files uploaded successfully!")
