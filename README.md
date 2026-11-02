# GazeTurn Lab

A test bench for one question: **can an ordinary laptop webcam turn e-book pages reliably enough for someone who can't use their hands?**

It tracks your eyes with the webcam, learns how they map to the screen, and opens a full-screen book. A page turns in one of two ways:

- **Last word:** you rest your eyes on the page's last word for about half a second. The word glows as the timer fills.
- **Look up:** you finish the page and look up to where it begins.

The default, **Both**, turns on whichever comes first. Every test is scored, so you get numbers instead of a feeling.

## Three trackers, one calibration

Each calibration trains three trackers and scores all of them on the same five check dots. The most accurate one is used automatically:

- **Classic:** a regression on iris position, eyelids and head pose from Google MediaPipe's face mesh.
- **Neural:** BlazeGaze, a small neural network from WebEyeTrack (Vanderbilt, 2025) that reads an image of both eyes plus head pose. It is then fitted to your eyes from the calibration.
- **Fusion:** both sets of signals in one model.

The calibration screen shows all three numbers. Click one to switch, or choose in Settings › Tracker. Calibration also redoes any dot that disagrees with the rest (usually a blink or a glance away).

Everything runs on your laptop. No video is recorded or uploaded.

## Run it (Mac)

1. Unzip the folder (for example into Downloads).
2. Open **Terminal** and run:
   ```
   cd ~/Downloads/gazeturn-lab
   node server.js
   ```
   Chrome opens at `http://localhost:8765`. Keep the Terminal window open while you use the lab, and press `Ctrl+C` there to stop.
   - If it says `command not found: node`, install Node.js (LTS) from nodejs.org, or run `python3 -m http.server 8765` in the same folder instead and open `http://localhost:8765` in Chrome yourself.
   - Alternatively, double-click `start.command`. macOS may block it the first time because it was downloaded: go to System Settings › Privacy & Security and choose **Open Anyway**.
3. When Chrome asks for the camera, allow it. If nothing happens, check System Settings › Privacy & Security › Camera › Google Chrome.

The first load needs internet: the face-tracking library (Google MediaPipe) and its model (a few MB) come from a CDN. The neural model and TensorFlow.js are included in the folder. Use Chrome; Safari is not supported yet. The gear icon (top right) holds settings: three looks (Keynote, Glass, Halo), viewing distance, face gestures and detector tuning.

## How to test the potential

Do this at a table with the laptop flat (not on your lap), light in front of you rather than behind, and the same posture each time.

1. **Setup.** Start the camera and fix anything red in Checks. Pick your MacBook model. For better degree numbers, measure eye-to-screen distance with a tape measure and enter it.
2. **Calibrate.** 13 dots, then a light that glides across four rows like lines of text, then 5 check dots. Note the accuracy. Under about 2.5° is normal for a webcam; above 3.5°, fix the lighting or position and redo it.
3. **Cursor test (Check step).** The mouse cursor is the ground truth. Rest it on something, look at its tip and hold your gaze, then move on. Cover the whole screen for a minute or two. Hold `Space` to record while moving it.
   - The top shows the error now, the typical error, and which way the gaze is biased ("sits 1.2° above").
   - **Map** shades each part of the screen by its error.
   - **Compare** shows all three trackers at once, each as its own ring.
   - **Learn** (`L`) retrains every tracker on what you just recorded and shows before/after accuracy on samples it didn't train on. Press **Use** only if it's better.
   - **Export** saves the raw data as JSON, for offline analysis and tuning.
4. **Shadow test, 2–3 times.** Read the built-in story (about 8 pages) and turn pages yourself with `→`. The lab predicts each turn in the background. This tells you how accurate the detector is without it interfering.
5. **Live test, automatic.** Now the lab turns the pages. When you reach the last word, rest your eyes on it until the underline fills. Or look up to the page's first line. Press `←` only if it turned too early, and `→` if it missed.
6. **Compare.** Run shadow tests with each Detect setting (Both, Last word, Look up), then live tests with *Auto + gesture* and *Look at Next*. These are the alternatives for users whose reading patterns don't suit the automatic mode.
7. **Repeat in worse conditions:** with glasses, in dim light, leaning back. Real users won't sit perfectly.

Recalibrate whenever you move your laptop or chair, or when the top bar says "moved". **Auto re-centre** also corrects small drift on every page. Your eyes visit two known places each page, the first word and the last, and the lab uses them to pull the gaze back onto the text.

Why "rest on the last word" rather than "turn as soon as I reach it": webcam tracking is off by about one or two lines vertically. The difference between the last line and the line above is behaviour, not position. At the end of an ordinary line your eyes jump back left within about 0.3 s. On the last word they stay.

### Record test data and score offline

Check › **Record test data** takes about 3 minutes:
1. A light visits 25 spots.
2. It glides along 7 lines.
3. You sit back and it visits 25 spots again.
4. You read one page.

It saves one `.zip` to Downloads. The file holds your face-landmark points, head pose and small pictures of your eyes, each tagged with where the light was. It stays on your computer unless you share it.

`tools/bench.js` replays a recording through every tracker using the site's own code, and scores each one on spots it never trained on:

```
cd tools && npm install && cd ..
node tools/bench.js ~/Downloads/gazeturn-capture-….zip
```

It prints a leaderboard in degrees for two tests: the same sitting (train on half the first 25 spots, test on the other half) and a new posture (test on the second 25 spots). For each tracker it shows the error per frame and per spot, the error at the top, middle and bottom of the screen, and the left–right and up–down bias. Because one recording can be replayed any number of times, every idea for improving accuracy is measured on the same data.

### Reading the scores

| Score | What it means | Good | Worrying |
|---|---|---|---|
| Turns caught | Share of real page ends the lab detected | ≥ 90% | < 70% |
| False turns / hour | Turns you didn't want. The most important number: a wrong turn is worse than no tool | ≤ 2 | > 6 |
| Line counting | Line returns counted vs the real count | 85–110% | < 70% |
| Timing (shadow) | How long after your own press the lab would have turned | under +1.5 s | over +2 s |
| Calibration | Average gaze error in degrees of visual angle | ≤ 2° | > 3.5° |

Verdicts: **Promising** means ≥ 90% caught with ≤ 2 false turns/hour. **Workable** means ≥ 70% and ≤ 6/hour. Anything else is **Not yet**. These cut-offs are a working choice for a first prototype, not an accessibility standard.

When a test underperforms, the Results page shows why. The *page-end candidates* line explains every rejected look-up. The *line returns* chart shows whether line counting is the weak step. Change one setting in **Tuning**, then press **Re-score** on a saved shadow test to see the effect without reading again.

**Raw data:** in Results, *Export session (JSON)* saves every gaze sample and event of a test, so it can be analysed or re-scored later.

## What's in the folder

```
index.html           page structure
css/style.css        styling (dark, three looks)
js/app.js            camera, calibration, reader, scoring UI
js/core/stats.js     median, mean, quantile
js/core/onefilter.js One Euro smoothing filter (Casiez et al., CHI 2012)
js/core/ridge.js     ridge regression with leave-one-dot-out cross-validation
js/core/features.js  eye and head features from MediaPipe's 478 face landmarks
js/core/blazeprep.js eye image, head vector and 3D eye position for the neural model
js/neural.js         runs the neural model (TensorFlow.js)
vendor/              TensorFlow.js and the BlazeGaze weights (licences inside)
js/core/detector.js  last-word, look-up and combined detectors, gestures, dwell, scoring, replay
js/sample-book.js    the built-in test story
server.js            tiny local web server (Node built-ins only)
start.command        double-click launcher for macOS
tools/bench.js       offline test bench for recordings (needs `npm install` in tools/)
js/capture.js        Record test data
js/zip.js            tiny zip writer/reader for recordings
tests/               Node tests, a simulated noisy reader, and a check of the neural
                     preprocessing against the original WebEyeTrack code
```

Run the tests with `node tests/run-tests.js`. The simulation reads 10 pages at increasing noise levels. With Both + auto re-centre, it catches 100% of turns with no false turns up to 2–3° of noise. Last word on its own is excellent up to about 1° but falls off past 1.5°, which is why Both is the default. Simulations are optimistic, though. Your real tests are the actual measurement.

## Known limits

- It only turns pages inside its own reader. Controlling Kindle or a PDF app (by sending a keypress) is the next step, once these tests show the detector is worth it.
- Calibration is tied to the window size and your seating position. The reader warns you if the window size differs from calibration.
- Glasses glare, low light, and heavy head movement all hurt webcam tracking.
- Pausing to think at the end of a line near the bottom for more than about 0.65 s can turn the page early in Last word mode. Raise *Rest on the last word for* in Tuning if that happens.
- The distance estimate from the camera is rough (±20%). It only affects how errors are converted to degrees, not the page turning.

## Credits

- Face landmarks: Google MediaPipe Face Landmarker (Apache-2.0).
- Neural gaze model: BlazeGaze from WebEyeTrack by E. Davalos et al., Vanderbilt University, MIT licence. Paper: arXiv:2508.19544. Code: github.com/RedForestAI/WebEyeTrack.
- TensorFlow.js (Apache-2.0).
