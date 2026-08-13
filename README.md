# Reconstruction of 6D Phase Space via Machine Learning

**Fermi National Accelerator Laboratory — Summer 2026**  
**Jermaine Shirlee | Supervisor: Dr. Katsuya Yonehara**

---

## Overview

Particle beams at Fermilab exist in a six-dimensional (6D) phase space defined 
by three positional components (x, y, z) and three momentum components 
(px, py, pz). Beam detectors are limited to recording spatial distribution 
making it computationally difficult measure momentum.

This project tests the capabilities Machine Learning to reconstruct the initial 
momentum  magnitude (p0) of a simulated muon beam from spatial distribution data alone, 
demonstrating an efficient computational approach to momentum measurement.

---

## Pipeline

```mermaid
graph TD
    A[G4Beamline Simulation] --> B[Text File Output:<br><i> x, y, z, px, py, pz per event</i>]
    B --> C[C++ Data Pipeline:<br> <br>Random 100-event sampling &rarr; 2D histogram PNG</i>]
    C --> D[ResNet18 CNN: <br><i>Trained on histogram images &rarr; predicts p0</i>]
    D --> E([Momentum Prediction<br><b>Mean Absolute Error < 1 MeV/c</b>])

```


