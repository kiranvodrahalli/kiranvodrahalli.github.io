---
layout: page
title: research
---


<!-- example of the message class
<p class="message">
  My name is Kiran Vodrahalli. 
</p>
-->

* TOC
{:toc}

## Publications 


## Undergraduate Work 

### Low-dimensional Representations of Semantic Drift in Thought-Space
My senior thesis, advised by Prof. Arora and Prof. Norman (from Princeton Neuroscience Institute), aims to build a model for extracting low-dimensional representations of the semantic content of an fMRI signal recorded during the presentation of stories in various formats. I am developed novel models to represent the context of a story as it \\(\textbf{changes over time}\\), and verifying my representations with predictive zero-shot brain decoding tasks. 

### Estimating Trending Twitter Topics With Count-Min Sketch <a href= "{{ site.baseurl }}/research/cos521paper.pdf" title= "cos521"> [pdf] </a>
My final project for <a href= "http://www.cs.princeton.edu/courses/archive/fall14/cos521/" title= "cos521"> COS 521: Advanced Algorithms</a> was joint with Evan Miller (Princeton COS '16) and Albert Lee (Princeton COS '16). We attempted to solve the following problem: Given a time series of Twitter data, can we infer current trending topics on Twitter while appropriately discounting past tweets using a sketch-based approach? We tweaked the Hokusai data structure <a href= "http://www.auai.org/uai2012/papers/231.pdf" title= "Hokusai"> [Matusevych et al 2012]</a> and implemented it, then ran experiments on Twitter data. 

### Comparing Hebbian Semantic Vectors Across Language <a href= "{{ site.baseurl }}/research/neu330paper.pdf" title= "neu330"> [pdf] </a>
My final project for NEU 330, Connectionist Models, focused on building Hebbian neural network word vector models for parallel corpora, with the purpose of evaluating the word vectors based on how similarly the word vectors for translation pairs behaved in their respective corpora. The principle I held throughout the project was simply that changing language should essentially not effect the representation of a word/concept in a high-dimensional vector space. I both proposed methods of evaluation and made use of previously used metrics to evaluate the 9 models considered. The texts used to form the word vectors were Harry Potter and The Philosopher's Stone and its French counterpart.

### Sparse, Low-dimensional and Multimodal Representations of Time Series for Mind-Reading <a href= "{{ site.baseurl }}/research/cos513/" title= "cos513"> [COS 513 Blog] </a> 

This work is joint with Lydia Liu (Princeton ORFE '17) and Niranjani Prasad (Grad Student, Princeton CS Department). We investigated the application of sparse canonical correlation analysis (sCCA) as a tool for creating low-dimensional combined representations of EEG/MEG and fMRI brain data. We used two experiments to demonstrate that our low-dimensional representation retained useful information by testing on two datasets: One was a paired EEG-fMRI time series oddball response dataset and the other was a paired MEG-fMRI time series dataset of subjects looking at various types of objects (<a href = "http://people.csail.mit.edu/rmcichy/publication_pdfs/Cichy_et_al_NN_2014.pdf" title="cichy2014"> Resolving human object recognition in space and time (Cichy et. al, 2014) </a>). In both instances we outperformed other traditional methods of low-dimensional representation, including PCA and ICA. We submitted our work as a project for <a href= "http://www.cs.princeton.edu/~bee/courses/cos513.html" title= "cos513web"> COS 513: Foundations of Probabilistic Modeling</a>, taught by Prof. Barbara Engelhardt.

### Learning Shifting Communities Online in the Adversarial Block Model

For the final project in APC 529: Coding Theory and Random Graphs, taught by Professor Emmanuel Abbé, I analyzed the Stochastic Block Model (SBM) from the perpsective of online optimization, making use of recent results in the online learning of eigenvectors and the exact recovery setting of the SBM to build a framework for learning communities as they change over time with guaranteed regret bounds. 

### Solving Word Analogies With Convex Optimization
My final project for <a href= "http://www.cs.princeton.edu/courses/archive/spring15/cos511/" title= "cos511"> COS 511: Theoretical Machine Learning</a> investigated convex loss functions for learning word vectors to solve word analogy problems. Word analogies are of the form king:man :: queen:woman. Given three of the four words, the task is to correctly identify the fourth. Traditionally, this problem is approached in the unsupervised setting and texts are used to learn which words are most relevant. Word vectors, word representations in high-dimensional real space, are often used (particularly in the past few years) as a solution to the analogy problem via dot-product queries, an approach which has recently been validated by <a href= "http://arxiv.org/abs/1502.03520" title= "random_walks_semantic_space"> [Arora et al (2015)]</a>. I formulated a convex loss with which to train word vectors that in principle keeps the spirit of the dot product query intuition, implemented AdaGrad <a href= "http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf" title= "AdaGrad"> [Duchi et al (2011)]</a>, and trained on word pairs. 

### Noun Compounds in Semantic Quad-Space <a href="{{ site.baseurl}}/research/iw2014paper.pdf" title= "iw2014"> [pdf] </a>
My junior independent work with Dr. Christiane Fellbaum aimed to build a model for analyzing the similarity between noun compounds, which consist of a modifier noun and a head noun, like "life force." Accurate parsing can greatly improve question answering systems for various knowledge bases. For example, medical QA systems must correctly parse noun compounds like "human colon cancer line" to answer questions accurately. I looked at several approaches to analyzing the similarity of noun compounds and built a vector space model of noun compounds, inspired by the papers of Turney <a href= "http://arxiv.org/abs/1309.4035" title="Domain_and_function"> [Turney 2013]</a> and Fyshe <a href= "http://www.aclweb.org/anthology/W13-3510" title="fyshe_paper"> [Fyshe et al 2013] </a>. I extended Turney's dual-space model to a quad space model and ran it on two large corpora, <a href= "http://corpus.byu.edu/coca/" title="coca"> COCA </a>  and <a href= "http://corpus.byu.edu/glowbe/" title="glowbe"> GloWbE </a>. I then evaluated the results by comparing to a ground truth provided by Mechanical Turk workers.  

### Characterizing Intellectual Interests with SVM
In fall 2013, I began working with Professor Sam Wang of Princeton Neuroscience Institute (PNI) on applying machine learning to an intellectual interest survey, which attempts to identify the discipline and intensity of academic interest in survey respondents. The goal of the project was to investigate intellectual interest as a potential phenotypic marker for autism. In order to study whether this hypothesis was plausible, we had survey responses from two groups of people. The Simons Simplex Collection (SSC) dataset is a repository of genetic samples from families where one child is affected with an autism spectrum disorder. We had survey responses from simplex members, the parents of autistic children. The other responses were obtained by polling readers of Professor Wang's political blog. My role in this project was to create a classifier which given a survey response could output a score indicating certainty that the survey respondent had a particular intellectual interest; for instance, the humanities. This project was my first exposure to the difficulty of munging through data and the application of machine learning to problems in cognitive science. The classifier I eventually trained had \\(94\\)% accuracy for determining intellectual interest, making the survey-classifier pair potentially useful as a tool.


## Coding research

My Github repository is located at <a href = "https://github.com/kiranvodrahalli" title="github"> https://github.com/kiranvodrahalli </a>. Code for various research listed above can be found on my Github, as well as some random for-fun research. 

I plan to re-add the code for several of the above research to my Github soon, as well as clean up some of the code from COS 598C: Neural Networks as taught by Sebastian Seung in Spring 2015, including my ImageNet deep CNN classifier written in \\(\texttt{theano}\\) and \\(\texttt{lasagne}\\). 

