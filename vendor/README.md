# Third-party files

| Folder | What | Licence |
|---|---|---|
| `tfjs/tf.min.js` | TensorFlow.js 4.22.0 (Google), runs the neural gaze model in the browser | Apache-2.0 (header inside the file) |
| `blazegaze/` | BlazeGaze trained weights from WebEyeTrack (Davalos et al., 2025, github.com/RedForestAI/WebEyeTrack) | MIT (see `blazegaze/LICENSE`) |

Paper: E. Davalos et al., "WebEyeTrack: Scalable Eye-Tracking for the Browser via On-Device Few-Shot Personalization", arXiv:2508.19544, 2025.

`js/core/blazeprep.js` is a plain-JavaScript port of WebEyeTrack's input preprocessing (eye patch, head vector, face origin). It is checked against the original on a real face: identical head vector and 3D origin, and 99.5% identical eye-patch pixels.
