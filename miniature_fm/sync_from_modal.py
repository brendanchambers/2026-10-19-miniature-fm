"""
Download model outputs from Modal Volume after training.
Usage: uv run modal run miniature_fm/sync_from_modal.py <run_id>
Example: uv run modal run miniature_fm/sync_from_modal.py --run-id teensy_20261021_150000
"""

import modal
import sys
import os
from pathlib import Path

app = modal.App("miniature-fm-download")
volume = modal.Volume.from_name("2026-10-19-miniature-fm")

@app.function(volumes={"/volumes": volume})
def list_files(run_id: str):
    """List files in the run directory"""
    import os
    remote_dir = f"/volumes/models/{run_id}"

    if not os.path.exists(remote_dir):
        return None

    files = []
    for root, dirs, filenames in os.walk(remote_dir):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, "/volumes")
            files.append(rel_path)

    return files

@app.function(volumes={"/volumes": volume})
def read_file(filepath: str):
    """Read a file from the volume"""
    with open(f"/volumes/{filepath}", 'rb') as f:
        return f.read()

@app.local_entrypoint()
def main(run_id: str):
    """Download results from Modal volume"""
    print(f"Downloading results for run: {run_id}")

    # List files
    files = list_files.remote(run_id)

    if files is None:
        print(f"Error: Run '{run_id}' not found in Modal volume")
        print(f"Check that the run completed successfully")
        return

    print(f"Found {len(files)} files to download")

    # Download each file
    for filepath in files:
        local_path = Path("volumes") / filepath
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading: {filepath}")
        content = read_file.remote(filepath)

        with open(local_path, 'wb') as f:
            f.write(content)

    print(f"\nDownload complete! Results saved to volumes/models/{run_id}/")
    print(f"View metrics: cat volumes/models/{run_id}/loss_plots/metrics.jsonl")
    print(f"View plot: open volumes/models/{run_id}/loss_plots/losses_final.png")
