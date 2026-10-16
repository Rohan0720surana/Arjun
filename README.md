<div align="center">

# ARJUN

### Campus Guardian

*“Arjuna saw only the eye of the bird.”*

**Normal CCTV records thousands of innocent students to catch one incident.
ARJUN watches no one, until someone gives it a reason to.**

</div>

---

> **Status: build in progress.** This README is filled in as each subsystem
> lands. The complete document — architecture diagram, detection rules, privacy
> design, setup, demo guide, ethics, credits — is written at the end of the
> build. See `docs/MASTER_PROMPT.md` for the brief this project is built to.

## Quick start

```bash
python -m pip install -r requirements.txt
cp .env.example .env
python run.py
```

Then open <http://localhost:8000>.

## CREDITS

Maintained as dependencies are added. Full licence notes land in the final
README.

| Component | Used for |
|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | person/object detection, pose, bundled ByteTrack |
| [COCO dataset](https://cocodataset.org) | the classes the pretrained weights know |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) (MIT) | multi-object tracking, via Ultralytics |
| [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache-2.0) | hand landmarks, face mesh |
| [OpenCV](https://opencv.org) (Apache-2.0) | capture, drawing, clip encoding |
| [NumPy](https://numpy.org) (BSD) | geometry and the heatmap |
| [FastAPI](https://fastapi.tiangolo.com) (MIT) | HTTP + WebSocket backend |
| [Uvicorn](https://www.uvicorn.org) (BSD) | ASGI server |
| [python-dotenv](https://github.com/theskumar/python-dotenv) (BSD) | configuration |
| [Requests](https://requests.readthedocs.io) (Apache-2.0) | Telegram + Twilio calls |
| [Anthropic Claude API](https://www.anthropic.com) | natural-language search, incident reports |
| [Telegram Bot API](https://core.telegram.org/bots/api) | push alerts |
| [Twilio WhatsApp API](https://www.twilio.com/whatsapp) | optional push alerts |
| [Chart.js](https://www.chartjs.org) (MIT) | dashboard analytics |
| [Google Fonts](https://fonts.google.com) (OFL) | Cormorant Garamond, Inter, JetBrains Mono |
| [pytest](https://pytest.org) (MIT) | the detector test suite |
