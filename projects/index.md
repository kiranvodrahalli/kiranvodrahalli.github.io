---
layout: page
title: projects
---


<!-- example of the message class
<p class="message">
  My name is Kiran Vodrahalli. 
</p>
-->

# Class Projects

## Solving Word Analogies With Convex Optimization
My final project for COS 511, Theoretical Machine Learning investigated convex loss functions for learning word vectors to solve word analogy problems. Word analogies are of the form king:man :: queen:woman. Given three of the four words, the task is to correctly identify the fourth. Traditionally, this problem is approached in the unsupervised setting and texts are used to learn which words are most relevant. Word vectors, word representations in high-dimensional real space, are often used (particularly in the past few years) as a solution to the analogy problem via dot-product queries, an approach which has recently been validated by <a href= "http://arxiv.org/abs/1502.03520" title= "random_walks_semantic_space"> [Arora et al (2015)]</a>. I formulated a convex loss with which to train word vectors that in principle keeps the spirit of the dot product query intuition, implemented AdaGrad <a href= "http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf" title= "AdaGrad"> [Duchi et al (2011)]</a>, and trained on word pairs. This project is a work in progress, as I am continuing to validate experimental results. Stay tuned for an ArXiv link!

## Comparing Hebbian Semantic Vectors Across Language <a href= "{{ site.baseurl }}/projects/neu330paper.pdf" title= "neu330"> [pdf] </a>
My final project for NEU 330, Connectionist Models, focused on building Hebbian neural network word vector models for parallel corpora, with the purpose of evaluating the word vectors based on how similarly the word vectors for translation pairs behaved in their respective corpora. The principle I held throughout the project was simply that changing language should essentially not effect the representation of a word/concept in a high-dimensional vector space. I both proposed methods of evaluation and made use of previously used metrics to evaluate the 9 models considered. The texts used to form the word vectors were Harry Potter and The Philosopher's Stone and its French counterpart. 

## Estimating Trending Twitter Topics With Count-Min Sketch <a href= "{{ site.baseurl }}/projects/cos521paper.pdf" title= "cos521"> [pdf] </a>
My final project for COS 521 attempted to solve the following problem: Given a time series of Twitter data, can we infer current trending topics on Twitter while appropriately discounting past tweets using a sketch-based approach? We tweaked the Hokusai data structure <a href= "http://www.auai.org/uai2012/papers/231.pdf" title= "Hokusai"> [Matusevych et al 2012]</a> and implemented it, then ran it on experimental Twitter data. 

# Independent Work 

## A Presentation on Expander Graphs <a href="{{ site.baseurl }}/projects/jsem2015paper.pdf" title="jsem"> [pdf] </a>
My junior seminar, taught by Professor Zeev Dvir in the spring of 2015, was on Point-Line Incidence Theorems. My presentation focused on a tangent subject, expander graphs. 

## Noun Compounds in Semantic Quad-Space <a href="{{ site.baseurl}}/projects/iw2014paper.pdf" title= "iw2014"> [pdf] </a>
My independent work in the fall of 2014 was advised by Christiane Fellbaum. I looked at several approaches to analyzing the similarity of noun compounds and build a vector space model of noun compounds, inspired by the papers of Turney <a href= "http://arxiv.org/abs/1309.4035" title="Domain_and_function"> [Turney 2013]</a> and Fyshe <a href= "http://www.aclweb.org/anthology/W13-3510" title="fyshe_paper"> [Fyshe et al 2013] </a>. I extended the dual-space model of Turney to a quad space model and ran it on two large corpora, <a href= "http://corpus.byu.edu/coca/" title="coca"> COCA </a>  and <a href= "http://corpus.byu.edu/glowbe/" title="glowbe"> GloWbE </a>. I then evaluated the results by comparing to a ground truth provided by Mechanical Turk workers.  

<!-- # Coding Projects -->
