## README.md
# COSMIC-2 Back-Propagation Geolocation Reproduction

This repository reproduces the methodology from:

**“Geolocation of the Ionospheric Irregularities in the Equatorial F Layer by Back Propagation of COSMIC-2 Radio Occultation Signals.”**

The goal is to reproduce the paper’s back-propagation, BP, geolocation method for ionospheric F-region plasma irregularities using COSMIC-2 high-rate GNSS radio occultation scintillation data.

The implementation has two main modes:

1. **Synthetic validation experiments**  
   Reproduce the paper’s numerical modeling of forward propagation, back propagation, phase screens, scan-angle sensitivity, two-screen cases, thermal noise, and multi-valued geolocation behavior.

2. **Real COSMIC-2 processing**  
   Process COSMIC-2 high-rate phase and amplitude/SNR data from POD antennas, apply IGRF-13-based BP geolocation, quality control, and generate geolocation products such as monthly maps and L1/L2 comparison statistics.

No machine learning training is involved. The configuration explicitly states:

