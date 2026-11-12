# From this demo to a real grading line

This project builds the vision and grading core. A working line needs several
more parts, which a Knowledge Transfer Partnership would develop on real
equipment with academic support. This note is an honest map of what is here and
what is not, and how they would join up.

## What is built here

- A synthetic fruit renderer with exact, automatic grade labels and defect masks.
- A CNN grader trained on synthetic data alone, shown to hold up on unseen
  cameras once colour normalisation and domain randomisation are used.
- Calibrated confidence, Grad-CAM explanations with a grounding check, and a
  grade-to-lane sorting decision with a value model.
- An operator dashboard.

## What a real build adds, and where it meets this code

**Real training images.** The synthetic renderer stands in for a photorealistic
pipeline (for example Blender or KeyShot) and for real captures from the line.
The model does not care where images come from: `fg/render.py` would be joined
or replaced by real and photorealistic images, and the same `fg/model.py` trains
on them. Domain randomisation (`fg/augment.py`) stays, because it is what lets
synthetic and real images be mixed.

**A proper camera setup.** Real lighting and a fixed, calibrated camera remove
much of the variation this demo has to randomise over. The white-balance and
contrast step in `fg/cam.py` is the hook for a real colour-calibration target.

**Discrete-element simulation of fruit on the belt.** Where fruit sits, how it
tumbles and how it presents to the camera affects grading. A DEM model would
generate realistic poses and occlusions to render and train on. It would feed
the renderer, not the model, so nothing downstream changes.

**The robot.** The sorter in `fg/sort.py` outputs a lane and an action per
fruit. On a real line that decision, with the fruit's position and a timestamp,
goes to the robot controller (for example a Fanuc delta) as a pick instruction.
The model already runs fast enough per frame for a belt; the open work is the
hand-off protocol, timing and safety interlocks, not the grading.

**Validation on real fruit.** The honest test is accuracy against human graders
on real produce, and a measured effect on throughput and value recovery. The
metrics here (accuracy, calibration, grounding, value) are the same ones that
would be reported then, on real data.

## The order it would be done

1. Capture and label a first batch of real fruit; confirm the grades with the
   packhouse against their specification.
2. Train on synthetic plus real, keeping domain randomisation; measure on held-
   out real fruit.
3. Build the DEM and photorealistic rendering to grow the synthetic set cheaply
   for rare defects.
4. Integrate the sorter decision with the robot controller on the training rig,
   away from live production.
5. Validate against human graders, then move to live grading with a human over
   the low-confidence cases.
