# VisionLab – Computer Vision Playground

## Run on Windows (PowerShell)

```powershell
cd ComputerVisionLab
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Current backend milestone
- Flask upload and processing API is connected.
- Image and video files are validated.
- Processed outputs are previewable and downloadable.
- Grayscale conversion is implemented as the first end-to-end algorithm.
- The remaining algorithms will be added incrementally.
- Maximum upload size: 100 MB.
