---
layout: page
title: COS 513 Project
---
<p class="message">
Team: Lydia Liu, Niranjani Prasad, Kiran Vodrahalli
</p>


## Overview


### Goals

We are interested in exploring the relationships between different measurements of brain activity, namely, fMRI, MEG, and EEG. fMRI data is based on a time series of BOLD responses and takes the form of a 4-vector with high spatial resolution, but low temporal resolution. On the other hand, both MEG (recordings of magnetic fields induced by electrical currents in the brain) and EEG (recording voltage fluctuations due to ion movement inside neurons) data have high temporal resolution, but less spatial resolution. Therefore, the primary question we ask for our project is: Can we utilize fMRI and (EEG or MEG) data, recorded simultaneously on a set of subjects performing some task, to jointly create a new signal that is able to retain both spatial and temporal resolution from the inputs to achieve a higher resolution on both axes?
If we take X to be EEG-space and Y to be fmRI space, we essentially wish to find two functions
$$ f_1(X, Y) \sim X' \\ f_2(X, Y) \sim Y'$$
such that X' and Y' are better representations of the underlying information in the brain data. 

We would also like to validate our multimodal representation with results on tasks like predicting the object a person is looking at, in order to demonstrate an improvement over the unimodal representations.


### The Dataset

We are currently using the Auditory and Visual Oddball EEG-fMRI dataset, available for download at the following <a href= "https://openfmri.org/dataset/ds000116" title= "https://openfmri.org/dataset/ds000116"> link </a>. The complete description of the data is available at the provided link. We summarize the main points here: 

- 17 healthy subjects performed separate but analogous auditory and visual oddball tasks (interleaved) while simultaneous EEG-fMRI was recorded.  

- The Oddball experiment paradigm: There were 3 runs each of separate auditory and visual tasks. Each run consisted of 125 total stimuli (each 200 ms): 20% were target stimuli (requiring a button response) and 80% were standard stimuli (to be ignored). The first two stimuli in the time course are constrained to be standard stimuli. The inter-trial interval is assumed to be uniformly distributed over 2-3 seconds.

	- Task 1 (*Auditory*): The target stimulus was a broadband “laser gun” sound, and the standard stimulus was a 390 Hz tone.

	- Task 2 (*Visual*): The target stimulus was large red circle on isoluminant grey background at a 3.45 degree visual angle, and the standard stimulus was a small green circle on isoluminant grey background at a 1.15 degree visual angle.

- The EEG data was collected at a 1000 Hz sampling rate across 49 channels. The start of the scanning was triggered by the fMRI scan start. The EEG clock was synced with the scanner clock on each TR (repetition time).

- The BOLD fMRI Data was an EPI sequence with 170 TRs per run, with 2 sec TR and 25 ms TE (echo time). There are 32 slices, and no slice gap. The spatial resolution is 3mm x 3mm x 4mm with AC-PC alignment.


### Related Papers

- <a href= "http://www.ncbi.nlm.nih.gov/pubmed/25797833" title= "[walz2015]"> Prestimulus EEG alpha oscillations modulate task-related fMRI BOLD responses to auditory stimuli [Walz et. al, 2015]</a>

- <a href= "http://www.ncbi.nlm.nih.gov/pubmed/23962956" title= "[walz2014]"> Simultaneous EEG-fMRI reveals a temporal cascade of task-related and default-mode activations during a simple target detection task [Walz et. al, 2014]</a>

- <a href= "http://www.ncbi.nlm.nih.gov/pubmed/24244465" title= "[conroy2013]"> Fast bootstrapping and permutation testing for assessing reproducibility and interpretability of multivariate fMRI decoding models [Conroy et. al, 2013]</a>

- <a href= "http://www.ncbi.nlm.nih.gov/pubmed/24305817" title= "[walz2013]"> Simultaneous EEG-fMRI reveals temporal evolution of coupling between supramodal cortical attention networks and the brainstem [Walz et. al, 2013]</a>

- <a href= "http://www.ncbi.nlm.nih.gov/pubmed/17688965" title= "[moosmann2007]"> Joint independent component analysis for simultaneous EEG-fMRI: principle and simulation [Moosmann et. al, 2007]</a>

- <a href= "http://onlinelibrary.wiley.com/doi/10.1002/hbm.22623/abstract" title= "[murta2015]"> Electrophysiological correlates of the BOLD signal for EEG-informed fMRI [Murta et. al., 2015] </a>

- <a href= "http://journal.frontiersin.org/article/10.3389/fnins.2014.00175/abstract" title= "[schmuser2014]"> Data-driven analysis of simultaneous EEG/fMRI using an ICA approach [Schmüser et. al., 2014] </a>

- <a href= "http://www.ncbi.nlm.nih.gov/pubmed/25514112" title= "[assecondi2014]"> Reliability of information-based integration of EEG and fMRI data: a simulation study [Assecondi et. al, 2014]</a>

### Preliminary Analysis

#### Exploratory Analysis of the EEG Data

Below is the raw EEG data from the experiment, where each horizontal time series is an EEG-source with a few exceptions. The first 43 channels are EEG electrodes, channel 44 and 45 are EOG (eye movement), channel 46 and 47 is ECG (heart), and channel 48 and 49 are the stimulus and the behavioral response event markers, respectively. 
<img src="{{ site.baseurl }}/projects/cos513/eeg-raw.jpg" />

We plot the same data after performing Independent Components Analysis (ICA) along the time series. We can see that the first few channels of EEG explain a lot of the variance. 
<img src="{{ site.baseurl }}/projects/cos513/eeg-ica.jpg" />

Here we examine the first component after ICA is performed on the EEG data. The first plot is of the variation in EEG response values. We see that for the first component, the response is roughly Gaussian.
This approximation roughly holds across the other components, though the kurtosis varies. 
<img src="{{ site.baseurl }}/projects/cos513/eegICAcompstats.png" />

Here we plot a correlation matrix of the EEG channels for the auditory task. The value of entry (i, j) is the covariance between two EEG channels summed over time. There appears to be an interesting block structure, suggesting that some channels are highly correlated with each other, while others are not. This correlation matrix allows us to check how we might reduce dimension later on.
<img src="{{ site.baseurl }}/projects/cos513/EEGcorr.png" />

Here, we plot a correlation matrix of the EEG channels across both the auditory and visual tasks for the first run in both tasks. We take covariances by summing over time for each channel pair, where one channel is the EEG response over auditory and the other channel is the EEG response over visual. This correlation matrix allows us to notice that across experiments, most of the EEG channels are not correlated with each other. Perhaps the channels that remain correlated signify more information about responses to stimuli (of various types) than the others. 
<img src="{{ site.baseurl }}/projects/cos513/EEGcrossCov.png" />

#### Exploratory Analysis of the fMRI Data

Here we have a nice visualization of the fMRI data. Notice that it looks like a brain!
<img src="{{ site.baseurl }}/projects/cos513/fmri-brain-vis.jpg" />

Then we simply binned the values of the fMRI data to get a general idea of what responses looked like. Note that a lot of the fMRI values are 0. When subsampling to plot covariances and so on, we drew voxels from the non-zero space.  
<img src="{{ site.baseurl }}/projects/cos513/BOLD_hist_s2t1r2.png" />

Here we plotted the mean BOLD response and the first two principal components. 
We performed PCA to reduce the voxel space dimension on a single subject’s fMRI data for task 1, run 2. The resulting components are therefore time vectors: The horizontal axis of the plot is time, and the vertical axis is response. The color scheme is as follows: blue is the mean, red is the first principal component and green is the second principal component. The first PC appears very close to the mean and it appears to explain 99% of the variance. This result is not what we expected, and possibly suggests that applying PCA directly may not be the best way to reduce the dimension of the data, due to the highly non-linear variation of the BOLD values of fMRI voxels. Thus we may explore nonlinear dimension reduction approaches like Local Linear Embedding or Isomap. Since we may be looking for signal in a very small percentage of the voxels, another interpretation of the result may be that our confusing results are due to the strong effect of noise (i.e. the voxels we are not interested in) on the results of PCA in high dimension. Thus, the 99% explained by the first principal component may only be explaining variance in voxels we are not interested in (also recall that from the histogram, most of the voxels are 0).
<img src="{{ site.baseurl }}/projects/cos513/Mean_bold_and_pcs.png" />

Here we plot a correlation matrix for each TR: At each time step, we calculate covariance by summing over voxels. 
<img src="{{ site.baseurl }}/projects/cos513/fmriCorr_time.png" />

Here we plot a correlation matrix for a subset of the voxels from the center of brain. Here, for each voxel pair, we calculate covariance by summing over time. 
<img src="{{ site.baseurl }}/projects/cos513/corr_BOLD_subset_of_voxels.png" />



#### Future Directions

- Investigate cross-correlation of the EEG and fMRI data at full scale. We did not implement this cross-modality analysis since we felt it would be less meaningful if we only sampled a small portion of the fMRI voxels while comparing to EEG sources, as running covariance calculations for large number of voxels crashed the software we were using to generate them (which is why we display a correlation matrix with only a small number of voxels on each axis). We will use a cluster to investigate the full correlation matrix of the fMRI voxel data, as well as the correlation between the EEG and fMRI data. 

- Apply more dimension reduction methods and plot time courses in two dimensions to see if any patterns arise. 

- Apply community detection analysis to the fMRI data. 




