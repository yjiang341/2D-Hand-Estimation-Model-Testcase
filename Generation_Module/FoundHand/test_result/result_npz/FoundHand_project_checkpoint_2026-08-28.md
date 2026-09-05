# 2D-Hand / FoundHand Project Checkpoint

Date: 2026-08-28

## Current status

The current pipeline is working through the FSK transport layer:

```text
Webcam / MediaPipe
    ↓
2D hand landmarks + handedness
    ↓
3D palm orientation detector
    ↓
EMA + hysteresis → stable PALM / BACK
    ↓
Pose Packet v3
    ↓
FSK modulation
    ↓
WAV
    ↓
FSK receiver
    ↓
Recovered 2D pose + hand presence + orientation
    ↓
FoundHand
```

### Confirmed working

- MediaPipe right/left hand detection
- Fixed RIGHT / LEFT packet slots
- 3D PALM / BACK orientation detector
- EDGE handling through EMA + hysteresis
- Duplicate same-handedness candidate suppression
- One-frame hand miss HOLD logic
- Pose Packet v3
- 104-byte packet size retained
- PALM/BACK metadata transmission
- FSK modulation / demodulation
- Receiver protocol-v3 orientation recovery
- Sender ↔ receiver hand-presence metadata: exact
- Sender ↔ receiver orientation metadata: exact
- Sender raw pose ↔ FSK recovered pose geometry: sub-pixel error

Latest corrected raw-vs-v3 comparison:

```text
20 sender frames
20 receiver frames
20 common frame IDs
0 missing frames

TX hand presence exact: 20/20
Orientation exact:      20/20

40 detected hand-frames compared

Mean landmark error:    ~0.394 px
Mean RMS error:         ~0.417 px
Worst point error:      ~0.692 px
```

Conclusion:

> Packet v3 + FSK transport preserves the transmitted 2D hand geometry.  
> Missing-looking fingers / finger fusion in generated images are not caused by the FSK transport layer.

---

## Important experimental findings

### 1. `MIRROR_X` is not the main problem

Both:

```python
BASELINE_MIRROR_X = True
```

and:

```python
BASELINE_MIRROR_X = False
```

still produce anatomy problems in some frames.

So do not spend much more time tuning this variable for now.

### 2. Personal IMG_1088 references performed poorly

Personal references were successfully prepared as:

```text
0000 = RIGHT_PALM
0001 = RIGHT_BACK
0002 = RIGHT_EDGE

0003 = LEFT_PALM
0004 = LEFT_BACK
0005 = LEFT_EDGE
```

However, using those references caused severe background / composition leakage in FoundHand generation.

Likely causes:

- cluttered room background
- chair / curtain / furniture structure leaking into generation
- reference-domain mismatch with FoundHand
- changing hand scale / placement between reference images

Current recommendation:

> Do not use IMG_1088 personal references for the next experiment.

### 3. Official IMG_1087 reference removes background leakage

Using:

```text
IMG_1087 frame 6
```

as a fixed official reference produces much cleaner backgrounds.

However, generated hand anatomy can still be inaccurate:

- finger fusion
- missing-looking fingers
- inaccurate side-view hand structure
- PALM/BACK surface ambiguity in some poses

Therefore the remaining problem is downstream of transport.

### 4. PALM/BACK orientation detector is still useful

The detector / protocol is working correctly.

But PALM/BACK alone does not solve:

- finger depth ordering
- self-occlusion
- side-view overlap
- 3D finger geometry

A 2D skeleton can contain all 21 keypoints while still being ambiguous to FoundHand.

---

# NEXT STEP

## Main next experiment: `test_official_vs_fsk_target.py`

Purpose:

Determine whether the hand-generation anatomy errors are mainly caused by:

1. **FoundHand itself**, or
2. **our FSK → FoundHand pose adapter / retarget distribution**

### A/B design

Keep all of these fixed:

```text
Reference image:       IMG_1087 frame 6
Reference pose:        IMG_1087 frame 6
Model checkpoint:      same
VAE:                   same
Sampling steps:        100
CFG scale:             2.5
Initial latent z:      same
Sampling noise:        same
Previous generated refs: OFF
```

Only change target-pose source.

### A — official FoundHand target

```text
official IMG_1087 target pose
        ↓
FoundHand
```

Choose one or more official target poses resembling the difficult FSK poses:

- open palm
- side view
- overlapping fingers

### B — FSK-derived target

```text
v3_recovered_pose.npz
        ↓
fsk_retarget
        ↓
FoundHand
```

Use comparable difficult FSK frames, e.g.:

```text
frame 0
frame 6
frame 12
frame 18
```

### Interpretation

If:

```text
official target → clean 5-finger result
FSK target      → finger fusion / missing-looking finger
```

then the main issue is likely:

```text
FSK pose adapter / retarget distribution mismatch
```

If:

```text
official target → similar anatomy failure
FSK target      → similar anatomy failure
```

then the main issue is likely:

```text
FoundHand model limitation for that pose/viewpoint
```

This A/B experiment should be done before inventing another retarget method.

---

# After the A/B test

Depending on the result:

## If adapter / retarget is the problem

Investigate:

- distribution matching to official FoundHand poses
- reference-pose similarity selection
- simpler normalization rather than more complex kinematic transforms
- whether target pose should be transformed differently before heatmap generation

Avoid immediately returning to:

- Procrustes
- kinematic reconstruction
- relative-angle reconstruction

Those previous experiments did not improve results.

## If FoundHand itself is the problem

Document model limitations:

- edge-on hands
- finger overlap
- open-palm geometry
- strong viewpoint changes
- 2D depth ambiguity

Then consider:

- another generation model from the original proposal
- richer conditioning (depth / 3D)
- model comparison as part of final evaluation

---

# Later project tasks

After the official-vs-FSK target A/B:

1. Quantitative generated-image pose evaluation
   - re-run MediaPipe on generated images
   - compare detected output landmarks to conditioning landmarks

2. Temporal video evaluation
   - pose interpolation
   - flicker / appearance stability
   - frame-to-frame consistency

3. Proposal-completeness work
   - compare at least one additional pose-estimation baseline if feasible
   - compare or discuss another hand-generation model
   - measure latency / FPS / runtime per pipeline stage

4. Final report metrics
   - FSK pose fidelity
   - hand-presence accuracy
   - orientation transport accuracy
   - generated pose preservation
   - generation success / failure categories
   - inference / transmission / generation latency

---

# Important files

## Sender / protocol

```text
Estimation_Module\Meeting_Bridge_Module\sender\meeting_sender.py
Estimation_Module\Pose_PacketUp\packet\pose_packet.py
```

## Receiver

```text
Estimation_Module\FSK_Module\fsk_receiver_main.py
Estimation_Module\FSK_Module\receiver\fsk_receiver.py
```

## Current recovered pose

```text
D:\Project\2D-Hand\logs\v3_recovered_pose.npz
```

## Sender raw diagnostic

```text
D:\Project\2D-Hand\local_sender_copy\raw_mediapipe_pose.npz
```

## Transport validation

```text
D:\Project\2D-Hand\compare_raw_vs_v3_pose.py
```

## FoundHand generation

```text
Generation_Module\FoundHand\my_demos\test_fsk_single_generation.py
Generation_Module\FoundHand\my_demos\test_image2video.py
```

## Official reference currently preferred

```text
FoundHand\test_data\iphone_video\IMG_1087\0006.jpg
FoundHand\test_data\iphone_video\IMG_1087.pkl
```

---

# Current project state summary

```text
MediaPipe detection                    ✅
RIGHT / LEFT slot mapping              ✅
3D PALM/BACK detector                  ✅
EMA + hysteresis                       ✅
single-frame miss HOLD                 ✅
Packet v3                              ✅
FSK modulation / recovery              ✅
hand-presence metadata transport       ✅ 100%
orientation metadata transport         ✅ 100%
2D geometry transport                  ✅ sub-pixel

personal reference experiment          ❌ background/domain leakage
MIRROR_X as main root cause            ❌ largely ruled out

─────────────────────────────────────────────
Official-target vs FSK-target A/B      ← NEXT
FoundHand anatomy limitation analysis  ⏭️
Generated-pose quantitative eval       ⏭️
Temporal / video refinement            ⏭️
```
