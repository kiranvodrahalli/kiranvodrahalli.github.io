---
layout: page
title: COS 513 Project
---
<p class="message">
Team: Lydia Liu, Niranjani Prasad, Kiran Vodrahalli
</p>




## Background Survey

### Our goals (\\(X\\) = EEG, \\(Y\\) = fMRI)
- find \\(f(X) = Y\\)
- find \\(f(Y) = X\\)
- find a low-dimensional mapping \\(f(X, Y) \to X'\\) where \\(X'\\) is low-dim EEG
- find a low-dimensional mapping \\(f(X, Y) \to Y'\\) where \\(Y'\\) is low-dim fMRI
- have a probabilistic generative model for \\(X'\\) and \\(Y'\\) (perhaps used in \\(f\\))

### Assessing the quality of \\(X', Y'\\)
- Supervised (predictive) tests
	- predict EEG signal from fMRI (accuracy)
	- predict fMRI signal from EEG (accuracy)
	- predict modality of signal type (auditory or visual) from oddball data
	- predict oddball signal on both auditory and visual data
	- how well does generative model do for covarying with \\(X'\\) and \\(Y'\\)?
- Unsupervised tests
	- analyze covariation of low-dimensional signal with high-dimensional signals
	- comment on correlations
	- goodness of generative model: compare with actual data (standard statistics) 

### Prior Methods and Approaches

Generally, prior approaches either use both fMRI and EEG in conjunction as separate information sources to verify neuroscientific claims, or map them into the same space with joint-ICA or CCA, potentially fitting a basic linear model with features from EEG to fMRI. 

#### Data Analysis Approaches

- General Overview of Approaches
	- fMRI-informed EEG aims to localize the source of the EEG data by using fMRI to construct brain model
EEG-informed fMRI: extract specific EEG feature, assuming its fluctuations over time covaries 

	- Neurogenerative modeling: similar to EEG-source modeling; inverse modeling based on the simulation of biophysical processes (known from neuroscience) involved in generating EEG and fMRI signals.

	- Data fusion [What we are most interested in]: Using supervised or unsupervised machine learning algorithms to combine multimodal datasets

- Late fusion methods [Multivariate Machine Learning Methods for Fusing Multimodal Functional Neuroimaging Data - Biessmann et al]

	- Supervised methods (either using an external target signal or asymmetric fusion where features from one modality are used as labels/regressors to extract factors from another modality) vs unsupervised (relying on data stats)

	- Unsupervised: PCA (maximal variance, decorrelated components), ICA (statistically independent components; temporal ICA for EEG, spatial for fMRI) Studies have found PCA works as well as ICA, with less to tune.

	- Supervised: regression and classification using external target signal (e.g stimulus type/intensity/latency, response time, artifactual information - linear regression, LDA) or using band-power features.

- Early fusion methods [Multivariate Machine Learning Methods for Fusing Multimodal Functional Neuroimaging Data - Biessmann et al]: First forming a feature set from each dataset followed by exploration of connections among the features.

	- *Multimodal ICA*: joint ICA (features from multiple modalities simply concatenated); parallel ICA (a user specified similarity relation between components from the different modalities is optimised simultaneously with modality-specific un-mixing matrices); linked ICA (Bayesian)

	- *CCA* and *PLS*: find the transformations for each modality that maximise the correlation between the time courses of the extracted components. Partial least squares aims to find maximally covarying components, CCA finds maximally correlating components. Relaxes independent component assumption of jICA, does not constrain component activation patterns to be the same for both modalities.

	- *mSPoC*: co-modulation between component power and a scalar target variable z can be modeled using the SPoC objective function;  in multimodal case, assume the target function z is the time-course of a component that is to be extracted from the other modality



#### Previous Results (Successes and Failures)

- EEG + fMRI (Walz et. al., 2013) on the Oddball dataset (our dataset)
	- EEG data \\(\to\\) training linear classifier to maximally discriminate standard and target trials \\(\to\\) create an EEG regressor out of demeaned classifier output (convolved with HRF) \\(\to\\) use the EEG regressor (and other event or response time related regressors) to fit a linear model to fMRI data \\(\to\\) comment on the correlation based on the coefficients (positive or negative, p-values). Also, they manually looked at fMRI images at TRs with high degree of correlation with the regressors.

	- How did they validate their performance?
		- They use the p-values of the coefficients to filter out correlates that are insignificant
		- Qualitative images and 'eyeballing it'-based analyses; comparing to previous known work to demonstrate that the data validates a neuroscientific model
		- P1 response, P300 response, etc (Calhoun paper)

- MEG + fMRI (Cichy et. al., 2014)
	- Used MEG and fMRI data to analyze the hierarchy of the visual pathway in the brain applied to object recognition (i.e., At what stage of visual processing does the brain disambiguate between human and non-human faces? How about man-made versus natural objects?)
	- How did they validate their performance?
		- Made plots of predictive power based on (MEG signal at each time point) over time, notice (with eyes) that peaks correspond to neuroscientifically-known time points in the visual process
		- Correlate the human fMRI with a monkey fMRI and report correlations for the same task
		- Generally, look at covariance plots and describe statistics for the time-points which are neuroscientifically known to be relevant for the neural visual processing pipeline

- mCCA vs jICA (2008)
	- Canonical Correlation Analysis for Feature-Based Fusion of Biomedical Imaging Modalities and Its Application to Detection of Associative Networks in Schizophrenia - Correa et al
	- Uses CCA to make inferences about brain activity in schizophrenia (found patients with schizophrenia showing more functional activity in motor areas and less activity in temporal areas associated with less gray matter as compared to healthy controls), general brain function (fMRI and EEG data collected for an auditory oddball task reveal associations of the temporal and motor areas with the N2 and P3 peaks)
	- The multimodal CCA (mCCA) method, we introduce is based on a linear mixing model in which each feature dataset is decomposed into a set of components (such as spatial areas for fMRI/sMRI or temporal segments for EEG), which have varying levels of activations for different subjects. A pair of components, one from each modality, are linked if they modulate similarly across subjects.
	- How did they validate their performance? 
		- Generated a simulated fMRI like set of components and an ERP-like set of components and mix each set with a different set of modulation profiles to obtain two sets of mixtures. The modulation profiles are chosen from a random normal distribution. The profiles are kept orthogonal within each set. Connections between the two modalities are simulated by generating correlation between profile pairs formed across modalities
	- Compares mCCA with jICA:
		- jICA examines the common connection between independent networks in both modalities while mCCA allows for common as well as distinct components and describes the level of connection between the two modalities
		- jICA model requires the two datasets to be normalized before being entered into a joint analysis underlying assumption of made in jICA is more reasonable when fusing information from two datasets that originate from the same modality
		- independence assumption in jICA; but utilizes higher order statistical information
		- mCCA jointly analyzes the two modalities to fuse information without giving preference to either modality; does not assume a common mixing matrix and does not require the data to be preprocessed to ensure equal contribution from both modalities
		- mCCA assumes that the components are linearly mixed across subjects
- Deligianni et. al (2014): Relating resting-state fMRI and EEG whole-brain connectomes across frequency bands
	- Apply sparse-CCA with randomized Lasso to fMRI-connectome and EEG-connectome for resting-state data (i.e., no supervised task) to identify the connections which provide most signal
	- Analyze the distance between precision matrices of the Hilbert envelopes (for fMRI and EEG)
		- Assuming brain activity patterns are described by a Gaussian multidimensional stationary process,  the covariance matrix fully characterizes the statistical dependencies among the underlying signals
	- They estimate prediction error via cross-validation for a function \\(f(\Omega_F) \approx \Omega_E\\) 



### Our Points of Novelty

- Most people don't think the fMRI and EEG data is low-dimensional, and thus don't go beyond vanilla approaches to find structure, with the exception of the Deligianni paper

- We can take advantage of the structure over time and space in fMRI and EEG data to reduce dimension and induce sparsity 

- Most people do not focus on coming up with generative models for fMRI / EEG data (except for 4 groups or so: John Cunningham, Jonathan Pillow and two others).

- We can go beyond Gaussian process assumptions to potentially create a more realistic view of the signals

- Use information-theoretic features in addition to typical neuroscientific features; validate results of the Assecondi paper

### Details of Our Approach

- What methods will you use to analyze these data?
	- How will features be represented?
		- as vectors in space and time 
			- vector in space varying over time (fMRI representations are sparse)
			- vector in time varying over space (EEG representations are sparse)
		- we can leverage a variety of dimension reduction approaches (both linear and non-linear)
		- use mutual information \\(\mc{I}\\) between points as a feature
		- use entropy \\(H\\) as a feature
	- What probabilistic models can we use to capture important signal in these data?
		- GLM (what everyone uses) with sparse features (after dimension reduction - sparse PCA?)
		- sparse CCA (following Deligianni et. al. but applied to predictive models)
		- Generative model with general non-gaussian distributions (for instance, fourth moment not \\(3\\))
			- Common underlying sources generate EEG and fMRI by mixing with EEG-specific noise sources and fMRI-specific noise sources
		- Can apply similar approach from Cichy et. al (2014)
			- Use machine learning classifier as a proxy for the predictive power of EEG signal at a given state
	- Information-theoretic models 
		- Most people using simultaneous EEG-fMRI rely on fitting a general linear model (in most cases, this is just a plain linear model with a linear mixing matrix etc) to describe the correlation between EEG and fMRI data. 
		- Thus most people only focus on linear correlation
		- Our alternative: Use mutual information and entropy to measure correlation, since these incorporate higher order correlations present in the data
		- Potential drawback: need high quality estimate of underlying probability distribution EEG/fMRI features are usually modeled as Gaussian - this assumption is way too broad
		- We can use different bias correction techniques based on number of samples, amount of correlation, and binning strategy.

- What assumptions are made about the data in the model (explicit and implicit)?
	- There is a high degree of correlation between EEG and fMRI data
	- Sparsity in the components (namely, that the thought signal lives in a much lower dimension, particularly true for the oddball task) 

- What will you do if those violated assumptions hurt performance?
	- 
- How will you fit the model to the data? Will this be computationally tractable?
	- CCA with regularization to induce sparsity in components.
	- Variational inference (for fitting generative model)
- How will you validate performance?
	 - We have some supervised tasks (as mentioned before); here, we can just assess predictive error
	 - We can also plot the data in interesting formats (over time, in low dimension, etc.) to assess qualitatively if the results are meaningful 
	 - We can compare our results to neuroscientific literature to see if our models hold up to previously known results 
- Is it feasible to compare our methods against other methods and to compare models? 
	- Yes, it is very feasible; we can compare with normal joint-ICA and non-sparse CCA

- Do reasonable implementations of the methods exist?
	- sparse-CCA has a reasonable implementation
	- variational inference fitting will depend on the specific assumptions we make about the priors
		- David Blei's group at Columbia is working on message-passing algorithms for plug-and-chug variational inference


### Links 

 - <a href=  "http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7182735" title= "http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7182735"> Multivariate Machine Learning Methods for Fusing Multimodal Functional Neuroimaging Data [Dähne et. al., 2015] </a>
 - <a href= "http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4740317" title="http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4740317"> Canonical Correlation Analysis for Feature-Based Fusion of Biomedical Imaging Modalities and Its Application to Detection of Associative Networks in Schizophrenia [Correa et. al., 2008]</a> 
 - <a href= "http://www.ncbi.nlm.nih.gov/pubmed/25221467" title="http://www.ncbi.nlm.nih.gov/pubmed/25221467"> Relating resting-state fMRI and EEG whole-brain connectomes across frequency bands [Deligianni et. al, 2014] </a>
 - <a href= "http://www.ncbi.nlm.nih.gov/pubmed/24305817" title= "http://www.ncbi.nlm.nih.gov/pubmed/24305817"> Simultaneous EEG-fMRI reveals temporal evolution of coupling between supramodal cortical attention networks and the brainstem [Walz et. al., 2013]</a>
 - <a href="http://www.ncbi.nlm.nih.gov/pubmed/16246587" title="http://www.ncbi.nlm.nih.gov/pubmed/16246587" > Neuronal chronometry of target detection: fusion of hemodynamic and event-related potential data [Calhoun et. al., 2006] </a>
 - <a href= "http://www.nature.com/neuro/journal/v17/n3/full/nn.3635.html" title="cichy"> Resolving human object recognition in space and time [Cichy et. al., 2014] </a>
 - <a href= "http://www.ncbi.nlm.nih.gov/pubmed/22553012" title= "huster"> Methods for simultaneous EEG-fMRI: an introductory review [Huster. et. al, 2012] </a> 
 - <a href="http://www.mitpressjournals.org/doi/pdf/10.1162/NECO_a_00695" title="info-theory"> Reliability of Information-Based Integration of EEG and fMRI Data: A Simulation Study [Assecondi et. al., 2015] </a> 



## Overview


### Goals

We are interested in exploring the relationships between different measurements of brain activity, namely, fMRI, MEG, and EEG. fMRI data is based on a time series of BOLD responses and takes the form of a 4-vector with high spatial resolution, but low temporal resolution. On the other hand, both MEG (recordings of magnetic fields induced by electrical currents in the brain) and EEG (recording voltage fluctuations due to ion movement inside neurons) data have high temporal resolution, but less spatial resolution. Therefore, the primary question we ask for our project is: Can we utilize fMRI and (EEG or MEG) data, recorded simultaneously on a set of subjects performing some task, to jointly create a new signal that is able to retain both spatial and temporal resolution from the inputs to achieve a higher resolution on both axes?
If we take \\(X\\) to be EEG-space and \\(Y\\) to be fMRI space, we essentially wish to find two functions 

$$
f_1(X, Y) = X'
$$ 

$$
f_2(X, Y) = Y'
$$ 

such that \\(X'\\) and \\(Y'\\) are better representations of the underlying information in the brain data. 

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




