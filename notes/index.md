---
layout: page
title: notes
---


<!-- example of the message class
<p class="message">
  My name is Kiran Vodrahalli. 
</p>
-->

As a general note, please contact me if you find errors in my notes so that I may fix them.

## Talks

I attend various academic talks and seminars and sometimes scribe them.

### Zaid Harchaoui on Convolutional Kernel Neural Networks <a href = "{{ site.baseurl }}/notes/convolutional_kernel_nns.pdf">[pdf]</a>

Professor Zaid Harchaoui from NYU's Courant Institute discuss Convolutional Kernel Neural Nets, which can be trained in an unsupervised manner and manage to outperform well-known CNN architectures with a fraction of the data. The formalism takes advantage of kernels to define an approximate minimization problem, adding more insight into how to tackle the problem of explaining convolutional networks theoretically.

### Lester Mackey on Divide-and-Conquer Matrix Completion <a href = "{{ site.baseurl }}/notes/lester_mackey_matrix_completion_concentration.pdf">[pdf]</a>

Professor Lester Mackey from Stanford talks on his recent work for the fast-parallelization of matrix completion under a novel divide and conquer framework.

### Gillat Kol on Interactive Information Theory <a href = "{{ site.baseurl }}/notes/interactive_info_theory_gillat_kol.pdf">[pdf]</a>

Dr. Gillat Kol from IAS gives a talk on information theoretic measures and bounds on the compression of information transmission in the setting of interactive information theory, where no player knows the whole message being compressed. 

### Sanjeev Arora on Reversible Deep Nets <a href = "{{ site.baseurl }}/notes/deepnets-reversible-arora.pdf" title="deepnets-reversible-arora">[pdf]</a>

Professor Arora, Tengyu Ma, and Yingyu Liang discuss their results on giving a generative model for reversible deep nets. They also give a better training method that improves upon Dropout. 

### Santosh Vempala on the Complexity of Detecting Planted Solutions <a href = "{{ site.baseurl }}/notes/planted-graph-vempala.pdf" title="vempala">[pdf]</a>

Professor Santosh Vempala of Georgia Tech gave a talk on showing that for statistical algorithms (e.g. PCA, EM, and so on), solving planted clique and planted \\(k\\)-SAT is at least exponential time in the size of the input.

### Amir Ali Ahmadi on Optimizing over Nonnegative Polynomials <a href= "{{ site.baseurl }}/notes/ahmadi.pdf" title="ahmadi">[pdf]</a>

Professor Amir Ali Ahmadi from Princeton spoke about formulating convex optimization problems in terms of finding nonnegative polynomials to provably optimize various problems in controls, dynamical systems, and machine learning, improving upon the time complexity of previous sum-of-squares SDP relaxations. He also describes robust dynamics optimization (RDO), a framework for solving optimization problems over a set of points defined by a dynamical system.

### Francisco Pereira on Decoding Generic Representations of fMRI <a href= "{{ site.baseurl }}/notes/pereira-words.pdf" title="pereira-words">[pdf]</a>

Dr. Francisco Pereira from Siemens talked about his work using sentences and pictures to localize fMRI voxels for representing semantic content in brains.

### Elad Hazan on Simulated Annealing and Interior Point Methods <a href= "{{ site.baseurl }}/notes/hazan-annealing.pdf" title="hazan-annealing">[pdf]</a>

Professor Elad Hazan gave a talk to Princeton's Algorithm-ML reading group on a result demonstrating the existence of a universal barrier function for interior point methods in the membership oracle model of convex optimization. This barrier is related to the heuristic simulated annealing approach often taken in non-convex optimization. 

### Dana Lahat on Joint Independent Subspace Analysis and Blind Source Separation <a href= "{{ site.baseurl }}/notes/joint-ISA.pdf" title="joint-ISA">[pdf]</a>

This talk gives an overview of blind source separation with various statistical independence assumptions, generalizing ICA to learning subspaces of low-rank instead of just rank \\(1\\) subspaces.

### Barbara Engelhardt on Bayesian Structured Sparsity Using Gaussian Fields <a href= "{{ site.baseurl }}/notes/engelhardt_pni.pdf" title="engelhardt-pni-notes">[pdf]</a>

Professor Barbara Engelhardt gave a talk on her work on identifying associations between SNPs and phenotypes with sparse machine learning methods. She also spoke on how these methods can be translated to brain studies.

### Dimitris Bertsimas on Statistics and Machine Learning from a Modern Optimization Lens <a href= "{{ site.baseurl }}/notes/SML_modern_optimization.pdf" title="MIOnotes">[pdf]</a>

Professor Dimitris Bertsimas from MIT gave a talk on using the mixed integer-programming (MIO) framework as a new lens through which to view machine learning, statistics, and optimization problems. 

### Sébastien Bubeck on Optimal Regret Bounds for the General Convex Multi-Armed Bandit Setting <a href= "{{ site.baseurl }}/notes/bubeck_talk.pdf" title="bubecknotes">[pdf]</a>

Dr. Sébastien Bubeck from Microsoft Research gave a talk to the Alg-ML reading group on his \\(2015\\) result on a tight minimax regret bound for the setting of general convex bandit optimization in dimensions greater than one. 

### Han Liu on Nonparametric Graphical Models <a href= "{{ site.baseurl }}/notes/han_liu_pni.pdf" title="hanliunotes">[pdf]</a>

Professor Han Liu from Princeton gave an overview of his recent research and theoretical results on nonparametric graphical models.

### Mehryar Mohri on Deep Boosting <a href= "{{ site.baseurl }}/notes/deep_boosting.pdf" title="mohrinotes">[pdf]</a>
Professor Mehryar Mohri from the Courant Institute speaks about ensemble boosting methods that take advantage of complex hypothesis classes along with the use of standard simple weak learners. He also presents an interpretation of boosting as truly being about model selection.

### Percy Liang on Learning Hidden Computational Processes <a href= "{{ site.baseurl }}/notes/percy_liang_talk.pdf" title="percyliangnotes">[pdf]</a>
Professor Percy Liang from Stanford discussed approaches to solving question-answering tasks on a new hand-built dataset. 

### Young Kun Ko on the Hardness of Sparse PCA  <a href= "{{ site.baseurl }}/notes/braverman_sparse_PCA.pdf" title="sparsepcanotes">[pdf]</a>

This talk summarizing two results on sparse principal components analysis was given to the Braverman Reading Group at Princeton.

### Ben Recht on Perceptron Learning and Stability
Professor Ben Recht from Berkeley discussed a notion of stability applied to stochastic gradient descent to explain why it reaches the same local optima in the nonconvex setting. Notes still to be added.

### Tom Griffiths on Rationality, Heuristics, and the Cost of Computation <a href= "{{ site.baseurl }}/notes/griffiths_rationality_heuristics_computationcost_berkeley.pdf" title="griffithsnotes">[pdf]</a>

Professor Tom Griffiths from Berkeley discussed a notion of rationality which takes into account computation time as a resource. From his perspective, we can explain why some decision-making procedures which appear to be suboptimal are actually optimal from a sparse-resource respective. 

### Anna Choromanska on Optimization for Large-Scale Machine Learning <a href= "{{ site.baseurl }}/notes/choromanska_deeplearning.pdf" title="choromanskanotes">[pdf]</a>

Dr. Anna Choromanska from the Courant Institute talks about new learning algorithms for decision trees and spin-glass interpretations of deep learning. 

### Richard Socher on Dynamic Memory Networks <a href= "{{ site.baseurl }}/notes/socher_last_lec_224d.pdf" title="socherdmnnotes">[pdf]</a>

Dr. Richard Socher's last lecture for the Stanford class 224d (Deep Learning for NLP) was on a recent paper by his startup, Metamind. He spoke about using a novel deep learning architecture to solve question-and-answer problems, and also about how to generalize all of NLP to a question-and-answer framework with this kind of model. 

## Classes

### APC 529: Coding Theory and Random Graphs

<a href= "{{ site.baseurl }}/notes/529_random-graphs.pdf" title="529randomgraphs">[Here]</a> are my notes on the random graphs portion of APC 529, taught by Professor Emmanuel Abbe of Princeton. The topics include an introduction to random graphs, the Erdos-Renyi model, graph properties and phase transition phenomena (for giant component and connectivity), some spectral graph theory, and finally an introduction to recent work on the Stochastic Block Model (SBM). 

### ELE 535: Pattern Recognition and Machine Learning

I will post my scribe notes for this course at the end of January 2016.

### MAT 340: Applied Algebra

I will post my scribe notes for this course very soon.

### COS 511: Theoretical Machine Learning

I took <a href= "http://www.cs.princeton.edu/courses/archive/spring15/cos511/" title= "cos511"> scribe notes</a> for most lectures of this class in its Spring 2015 iteration. Topics in the notes consist of an introduction to statistical learning theory, the Online Convex Optimization (OCO) framework, regularization, Bandit Convex Optimization (BCO), boosting, some game theory all from the point of view of OCO, and finally, an explicit connection between OCO and statistical learning theory in the form of theorems which convert regret analysis into ERM guarantees. There are also two guest lectures, one by Professor Jake Abernethy of University of Michigan, and one by Professor Sanjeev Arora of Princeton.

### COS 510: Programming Languages

Here are <a href= "{{ site.baseurl }}/notes/curry_howard_cos510notes.pdf" title="cos510notes"> my scribe notes</a> on the Curry-Howard Isomorphism. 

### APC 486: Transmission and Compression of Information

Here are <a href= "{{ site.baseurl }}/notes/apc486_kiran_scribe_notes.pdf" title="apc486notes">my scribe notes</a> on probabilistic source models.

## Useful Links

To be populated. I will add some links to other notes, class webpages, and other people's webpages which I have found interesting and helpful. 






