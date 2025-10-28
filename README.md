# Jumbled Video Fixing

Restore video files whose frames are in the wrong order. This project uses a pre-trained ResNet50 model to understand frame content, builds a similarity matrix, and reconstructs the most likely chronological sequence before exporting a corrected MP4.

## Features

- Handles completely shuffled frame orderings
- No training required – leverages ImageNet-pretrained ResNet50
- Two-step command line workflow (prepare once, fix many times)
- Optional legacy script for single-command processing

## Requirements

- Python 3.9 or newer
- `pip install -r requirements.txt`
- Input video readable by OpenCV (MP4 recommended)
- CUDA-compatible GPU optional (CPU works but is slower)

## Quick Start

```bash
# clone and install
git clone https://github.com/gaur-avvv/Jumbled-Video-Fixing.git
cd Jumbled-Video-Fixing
pip install -r requirements.txt

# step 1 – extract frames and compute features
python prepare_model.py jumbled_video.mp4
# -> creates ./frames/ with frame images, features.npy, video_info.pkl

# step 2 – rebuild the correct order and export
python fix_video.py ./frames fixed_video.mp4
# add --cleanup to delete the frames folder afterwards if you wish
```

#### All-in-One Command (optional)

For one-shot processing you can still run:

```bash
python fix_jumbled_video.py input.mp4 output.mp4
```

Splitting the workflow, however, saves time when you want to experiment with reconstruction settings.

## How It Works

1. **Frame Extraction** – OpenCV saves each frame as `frameXXXX.jpg`.
2. **Feature Extraction** – ResNet50 (classifier head removed) converts frames into 2048-D embeddings.
3. **Similarity Matrix** – cosine similarity compares every pair of frames.
4. **Endpoint Detection** – frames with the fewest strong neighbors are candidate starts/ends.
5. **Greedy Reconstruction** – iteratively append the most similar unused frame from each candidate endpoint and keep the best-scoring sequence.
6. **Direction Check** – simple trend analysis flips the sequence if the best path is reversed.
7. **Video Export** – reordered frames are encoded back to MP4 with the original FPS and resolution.

## Tips

- First execution downloads the ResNet50 weights (~100 MB).
- GPU acceleration cuts feature extraction time dramatically; scripts fall back to CPU automatically.
- Keep the `frames/` folder if you plan more experiments, or use `--cleanup` to remove it.
- Typical results reach 95–99% frame similarity on ~300-frame videos.

## Repository Layout

```
README.md                    # Project guide
prepare_model.py             # Step 1 script
fix_video.py                 # Step 2 script
fix_jumbled_video.py         # Legacy single-command script
Video_Reconstruction_Tutorial.ipynb  # Notebook / Colab workflow
requirements.txt             # Python dependencies
```

## Optional VS Code Enhancement

To preview reconstructed videos inside VS Code, install the **Video Preview** extension (`digitallyinduced.video-preview`). Right-click the MP4 file and choose **Open with Video Preview**.

## License & Contributions

Open source and contributions are welcome. Please open an issue or submit a pull request with improvements or questions.
