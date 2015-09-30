---
layout: page
title: COS 513 Project
---


<!-- example of the message class
<p class="message">
  My name is Kiran Vodrahalli. 
</p>
-->

## Overview


### Goals

We are interested in exploring the relationships between different measurements of brain activity, namely, fMRI, MEG, and EEG. fMRI data is based on a time series of BOLD responses and takes the form of a 4-vector with high spatial resolution, but low temporal resolution. On the other hand, both MEG (recordings of magnetic fields induced by electrical currents in the brain) and EEG (recording voltage fluctuations due to ion movement inside neurons) data have high temporal resolution, but less spatial resolution. Therefore, the primary question we ask for our project is: Can we utilize fMRI and (EEG or MEG) data, recorded simultaneously on a set of subjects performing some task, to jointly create a new signal that is able to retain both spatial and temporal resolution from the inputs to achieve a higher resolution on both axes?

We would also like to validate our multimodal representation with results on tasks like predicting the object a person is looking at, in order to demonstrate an improvement over the unimodal representations.


### The Dataset

We are using the Auditory and Visual Oddball EEG-fMRI dataset, available for download at the following <a href= "https://openfmri.org/dataset/ds000116" title= "https://openfmri.org/dataset/ds000116"> link </a>. The complete description of the data is available at the provided link. We summarize the main points here: 

- 17 healthy subjects performed separate but analogous auditory and visual oddball tasks (interleaved), while simultaneous EEG-fMRI was recorded.  

- The Oddball paradigm: There were 3 runs each of separate auditory (``task 1") and visual (``task 2") tasks. Each run consisted of 125 total stimuli (each 200 ms): 20\% target stimuli (requiring button response), 80\% standard stimuli (to be ignored). First 2 stimuli constrained to be standards. Intertrial interval is 2-3 sec, uniformly distributed. 

	- Task 1: The target stimulus was a broadband “laser gun” sound, and the standard stimulus was a 390 Hz tone.

	- Task 2: The target stimulus was large red circle on isoluminant grey background at a 3.45 degree visual angle, and the standard stimulus was a small green circle on isoluminant grey background at a 1.15 degree visual angle.

- The EEG data was collected at a 1000 Hz sampling rate across 49 channels. The start of the scanning was triggered by the fMRI scan start. The EEG clock was synced with the scanner clock on each TR (repetition time).

- The BOLD fMRI Data was an EPI sequence with 170 TRs per run, with 2 sec TR and 25 ms TE (echo time). There are 32 slices, and no slice gap. The spatial resolution  is 3mm x 3mm x 4mm with AC-PC alignment.


### Related Papers

### Preliminary Analysis
