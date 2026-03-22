# Bird Watcher Skill — Launch Posts

Draft posts for each platform. Adjust as needed before posting.

---

## 1. ClawHub Submission

**Skill Name:** bird-watcher
**Repo:** https://github.com/MS-707/bird-watcher-skill
**Tags:** computer-vision, yolo, wildlife, camera, moondream, streaming

**Description:**
Real-time bird detection using YOLOv11 with Moondream VLM species identification. Streams an annotated live video feed to any device on your local network. Detects birds with bounding boxes at 15fps on Apple Silicon, identifies species via local VLM, saves every detection frame, and logs to wildlife census. Everything runs locally — no cloud, no subscriptions. Works with any Mac webcam.

---

## 2. OpenClaw Discord (#skills / #showcase)

**Post:**

🐦 **New Skill: Bird Watcher — Real-time YOLO bird detection with species ID**

Just published a skill that turns any Mac with a webcam into a live bird detection station.

**What it does:**
- YOLOv11 scans every frame for birds at ~15fps on M1
- When a bird is detected, Moondream VLM crops and identifies the species
- Streams live annotated video to your phone via MJPEG (open a URL, that's it)
- AirPlay to your TV for a backyard bird cam on the big screen
- Saves every detection frame automatically
- Integrates with wildlife-census and birdnet-audio skills if you have them

**Architecture:** Camera and YOLO run in separate threads so the feed is always smooth. YOLO boxes overlay independently. Moondream only fires on detections with a 5s cooldown so it doesn't hog resources.

**Security:** Auth token generated on each startup. Local network only. No data leaves your machine.

📦 `git clone https://github.com/MS-707/bird-watcher-skill.git`

[ATTACH DEMO VIDEO HERE]

Would love feedback — especially if anyone tries it on Intel Macs or with USB cameras. First skill I've published, so roast the README if something's unclear.

---

## 3. Reddit — r/selfhosted

**Title:** I created a local-first AI bird detection system that streams to any device on your network. No cloud, no subscriptions, runs on a MacBook.

**Post:**

I wanted a bird camera for my porch feeder but every commercial option sends video to someone else's cloud. So I made my own.

**What it does:**
- MacBook webcam pointed at bird feeder
- YOLOv11 runs on every frame detecting birds in real-time (~15fps on M1)
- Green bounding boxes appear instantly when a bird enters the frame
- Local VLM (Moondream) identifies the species by cropping the detected bird
- Streams live annotated video via MJPEG to any device on your WiFi
- Auth token required to view — generated fresh each session
- Saves every detection frame to disk with timestamps
- Everything runs locally. Zero cloud calls. Zero accounts. Zero monthly fees.

**Stack:** Python, OpenCV, YOLOv11 (Ultralytics), Moondream VLM, Flask for MJPEG streaming

**Hardware:** MacBook Air M1 with built-in camera. That's it.

The camera thread and YOLO thread are decoupled so the video feed is always smooth regardless of detection processing speed. Moondream only fires when YOLO spots a bird, with a 5-second cooldown.

It started as a weekend project with my AI assistant (OpenClaw) and turned into something my partner and I actually watch on the TV via AirPlay. There's something weirdly satisfying about seeing a green box pop up on a scrub jay in real-time.

Open source, MIT licensed: https://github.com/MS-707/bird-watcher-skill

[ATTACH DEMO VIDEO OR GIF HERE]

Happy to answer questions about the architecture or YOLO performance on different hardware.

---

## 4. Reddit — r/birding

**Title:** Built a real-time AI bird detection camera for my backyard feeder — identifies species automatically

**Post:**

I'm not a developer by trade (I work in environmental health & safety) but I've been tinkering with AI tools and wanted a smarter way to know what's visiting my feeder in Sonoma County.

**What I made:**
- Webcam pointed at my feeder streams live video to my phone
- AI (YOLOv11) draws green bounding boxes around birds in real-time
- When a bird is detected, a vision model crops the image and tries to identify the species
- Every detection is saved as a timestamped photo — both the original and the annotated version
- Audio detection (BirdNET by Cornell Lab) runs in parallel to catch calls

It's not perfect — the species ID is a general-purpose AI, not a dedicated bird model, so it gets common backyard birds right (jays, sparrows, finches, robins) but might struggle with rarer species. The visual detection itself (finding birds in the frame) is very reliable though.

Today it caught 42 detections in an afternoon session. Mostly scrub jays and finches at my feeder.

Everything runs locally on my laptop — no cloud, no subscription, no data goes anywhere. Open source if anyone wants to try it: https://github.com/MS-707/bird-watcher-skill

Requires a Mac with a webcam and some comfort with Terminal commands. Happy to help anyone set it up.

[ATTACH BEST DETECTION PHOTO HERE]

What species are you all seeing at your feeders this spring?

---

## 5. Reddit — r/computervision

**Title:** YOLOv11 + VLM pipeline for real-time bird detection and species classification — decoupled threading architecture

**Post:**

Built a real-time detection system that combines YOLOv11 for fast object detection with Moondream VLM for fine-grained classification. Sharing the architecture in case it's useful for anyone doing similar multi-model pipelines.

**Problem:** I wanted to detect and identify birds at a feeder in real-time, stream the annotated feed to other devices, and save detection frames — all on a MacBook Air M1 with no cloud.

**Architecture:**
```
Camera Thread (30fps) → captures frames, overlays latest YOLO boxes, encodes MJPEG
    ↓ shares frame via threading.Lock
YOLO Thread (~10-15fps) → runs YOLOv11s on latest frame, bird class only (COCO #14)
    ↓ on detection, 5s cooldown
VLM Thread → crops detected region + 25% padding, sends to local Moondream for classification
```

Key design decisions:
- **Decoupled threads** — camera never waits for YOLO, YOLO never waits for VLM. Feed is always smooth.
- **YOLO as gatekeeper** — VLM is expensive (~500ms per inference), so it only fires when YOLO detects something. 5s cooldown prevents spam.
- **Bounding box persistence** — boxes stay visible for 3s after last detection so fast-moving subjects don't just flash for one frame.
- **Detection frame saving** — both original (for retraining) and annotated (for review) saved with microsecond timestamps.

**Performance on M1 Air:**
- Camera: 30fps native
- YOLOv11n: ~15-20fps processing
- YOLOv11s: ~10-15fps (recommended, better recall on small birds)
- YOLOv11m: ~5-8fps
- Moondream: ~500ms per classification (background thread, doesn't affect stream)

**Limitations:**
- COCO's "bird" class is coarse — no species differentiation at the YOLO level
- Moondream is a general VLM, not bird-specific — gets common species right but struggles with similar-looking species
- Would benefit from a fine-tuned bird detection model (like Birds-YOLO from this paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12650164/)

Next steps I'm considering:
- Fine-tuning on CUB-200-2011 or NABirds dataset for species-level YOLO detection
- Adding motion detection pre-filter to avoid running YOLO on static frames
- WebRTC instead of MJPEG for lower latency streaming

MIT licensed: https://github.com/MS-707/bird-watcher-skill

Interested in feedback on the threading architecture and whether anyone has experience with bird-specific YOLO fine-tuning.

---

## 6. Hacker News — Show HN

**Title:** Show HN: Bird Watcher – Real-time YOLO bird detection that streams to your phone, runs locally

**Post:**

I created a bird detection system that turns a Mac webcam into a live AI-annotated bird camera. YOLOv11 detects birds with bounding boxes at ~15fps, a local VLM (Moondream) identifies the species, and the whole thing streams to any device on your network via MJPEG.

Everything runs locally — no cloud, no accounts, no data leaves your machine. Auth token generated per session.

The camera and YOLO run in separate threads so the video feed is always smooth. The VLM only fires when a bird is detected, with a cooldown to avoid burning compute.

I'm not a software engineer — I work in environmental health & safety. This started as a weekend project with my AI assistant (OpenClaw) and turned into something my partner and I actually watch on our TV via AirPlay.

https://github.com/MS-707/bird-watcher-skill

---

## 7. Product Hunt

**Tagline:** Real-time AI bird detection for your backyard — runs locally, streams to any device

**Description:**
Bird Watcher turns any Mac with a webcam into a live bird detection station. Point it at your feeder, open a URL on your phone, and watch AI draw bounding boxes around every bird in real-time. Species identification happens automatically via local VLM. No cloud. No subscription. No data leaves your network.

**Topics:** Artificial Intelligence, Open Source, Privacy, Nature, Smart Home

---

## Notes

- All posts need the demo video/GIF before publishing — that's the most important asset
- r/birding post is intentionally less technical — focus on the birding community angle
- r/computervision post is intentionally more technical — architecture and performance focus
- HN post is brief — they prefer concise Show HN posts with the details in the repo
- Adjust tone for each community — don't cross-post the same text
- Wait 24-48 hours between platform posts to avoid looking spammy
