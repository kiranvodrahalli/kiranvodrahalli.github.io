---
layout: page
title: Notes from MIT Deep Learning Theory and Non-Convex Optimization Workshop (January 27-30)
---

* TOC
{:toc}

See [website](http://mifods.mit.edu/deep.php).

___

## Bootcamp (01/27/2019)

### Inductive Bias and Generalization in Deep Learning (Nati Srebro)

#### Introduction 

This topic -- this entire question is inductive bias. What I hope to do is rather than walk you through results, or give you new understanding, I'm going to focus on what we do need to understand, setting up the questions. In the first half of the talk, we'll just be talking about learning itself. What is inductive bias? 

Inductive bias is this fuzzy term. When two people say inductive bias, they probably mean different things. 

#### What do we mean by supervised learning ?

The problem of supervised learning is we want to find a good predictor (in expectation over examples we expect to see in reality, the prediction error is low). Where does learning come in? We don't know how to specify reality. We want to learn the rules of reality based on data. We hope the examples are drawn from the reality distribution, and learn a predictor according to this data. We want to have a good learning rule that gives you a good predictor from the data. 

The no free lunch theorem says this is impossible if you don't assume anything about reality. There's some reality for which it's possible to get zero error, but the learning rule will only get you error at least half. Inductive bias tells us that some realities are less likely than others. So we want to bias ourselves towards more likely realities. 

So in this sense of inductive bias, it's something you put into your learning algorithm. Instead of memorizing predictors, let's bias ourselves towards predictors of a particular form (e.g., linear predictors). 

Another way to think about inductive bias is the other way around --- the properties of reality under which we can show the learning rule works. Maybe under some sensible assumptions about reality, our learning rule does work. 

Let's see some examples: We can capture some mappings are more reasonable than others. One is when you have a hypothesis class --- you say reality is captured by some predictor in your hypothesis class. Another is you have some complexity measure of hypothesis (e.g., a smooth predictor). 

A **flat inductive bias** bins the world into plausible and implausible predictors (e.g., a hypothesis class). If we have a flat inductive bias, you can just use empirical risk minimization over the hypothesis class. Now you can get a guarantee on learning. If our assumption holds, then we get a guarantee in expectation, which says that the generalization error for predictor returned by learning rule is at least as good the loss of our predictor plus some error term which scales as the complexity/capacity/size of the hypothesis class divided by the number of samples. This only holds if our assumption about reality holds. For binary predictors, the capacity is the VC-dimension of the hypothesis class. For linear predictors, the capacity is the dimension. Usually with \\(d\\) parameters, the capacity corresponds roughly to the number of parameters. But that's not necessarily true. For infinite dimension predictors of bounded norm, the capacity often depends on the size of the bound. 

Now the name of the game in machine learning is to find good hypothesis classes. What makes one good? It should capture reality well. Doesn't need to capture it exactly -- this is one of the nice aspects of machine learning. As long as we can approximate reality, that's good enough. Also we need the capacity of the class to be small enough to allow generalization. 

#### Non-Flat Complexity Measures and Structural Risk Minimization

A complexity measure is any function from predictors to the reals which tells us how likely we think something is. We're going to use this to compare predictors to figure out which is more likely --- so it's just an ordering on predictors. We prefer predictors of smaller complexity than others. If this is our inductive bias, what should our learning rule be? We use structural risk minimization -- we want to minimize both empirical error and the complexity of a hypothesis. This is a bipartite optimization problem. How do we solve this problem? We need to trade these off, and choose the tradeoff. There are many ways of doing this bipartite optimization problem. You can minimize empirical error plus some regularization term, etc. They're all about finding Pareto frontier for the optimization problem. Whatever tradeoff term you choose, you'll pick it with some kind of validation method (like cross-validation). 

Now what can we say? Roughly we will ensure that our learning rule will compete with any predictor that has small error, but the number of samples you'll need is going to scale corresponding to the complexity of the sublevel set of the predictor. The more samples you have, you can compete with a bigger sublevel set, and you can compete with hypotheses of higher complexity. So talking about a complexity measure and a hierarchy of complexity classes is the same. Note that this hierarchy can be continuous. 

So when we talk about understanding capacity, we should really be talking about complexity measure, but really, it's enough to talk about hypothesis classes (e.g. flat measures of complexity). All we need to know is about the complexity of the hypothesis class of the particular sublevel set. It's almost true that you can take an infimum over \\(h^* \\) in the error bound on the SRM predictor: $$L(SRM_{H})(S) \leq L(h^* ) + \sqrt{\frac{capacity(\mathcal{H}_{complexity(h^* )})}{m}} $$


#### Implicit Inductive Bias
There are other learning rules beyond ERM and SRM, for instance, stochastic gradient descent. We can look at the inductive biases associated with these other learning rules. These might be implicit inductive bias. This doesn't necessarily do ERM or SRM with respect to anything; nevertheless, we can get a generalization bound for this learning rule. We can get a bound 
$$ L(SGD(S)) \leq L(w^* ) + \sqrt{\frac{\|w\|_{2}}{m}} $$

So basically, the inductive bias associated with SGD is the Euclidean norm of the weight. By using SGD, you're saying you believe that reality is captured by a small norm predictor. I want to emphasize that this is NOT ERM. So you can think of inductive bias in two directions. You can first decide on inductive bias, and then design an algorithm (ERM or SRM). The second way to think about it is that you can start with an algorithm (like SGD) and then see oh, it works -- let's understand the underlying inductive bias. This maybe comes after the fact though. 

You could think of other learning algorithms; e.g. nearest neighbor. For nearest neighbor to work, you need \\(P(y|x)\\) to be smooth with respect to some distance measure. Then if you do nearest neighbor with respect to that distance measure, then it will work. 
I don't necessarily have a good reference for how the initial nearest neighbor guarantees were achieved. 

If you do exponential gradient descent / multiplicative weights, then this corresponds to low \\(\ell_1\\) norm e.g. sparsity. There are many other examples. If you look in SGD, you can find where Euclidean norm shows up and replace the inductive bias with other inductive biases (think of Mirror Descent, etc.). 

Now we understand the questions we should ask about neural nets. 

#### Neural Nets and Free Lunches

We'll think of a neural network as a directed acyclic graph. We have activation fixed, graph fixed, etc. The thing to learn is the weights of the units. This defines a hypothesis class --- inductive bias is given by functions which can be captured by this hypothesis class. This is a flat inductive bias. If you're also investigating different architectures, you can think of it as a complexity measure over the architectures. 

Now to understand hypothesis class, we have to understand its capacity, and understand whether it corresponds to reality. We actually know it pretty well --- roughly speaking it's captured by the number of parameters of the network. So now we know how to measure the size, and we can ask "can we capture reality well?" The first answer is yeah! We can approximate any continuous function arbitrarily well with a neural network. It depends on how big your network should be. In order to approximate a function, the number of units has to grow exponentially with dimension. It can only capture reality with huge capacity, which is useless for us. We want to be able to capture reality well with small capacity. There's many explanations of what you can capture with a small neural network. You can talk about all kinds of features (formulas, disjunctions and conjunctions, etc.). But all of this is unnecessary, because there's a much stronger result that tells us we can capture anything we want with a small network. Any time \\(T\\) computable function can be captured with size \\(\mathcal{O}(T)\\). So it seems like we can capture anything we want with neural nets. 

So I tricked you before --- there are free lunches. Why are we doing learning in the first place? Our goal is to be lazy. Instead of programming it directly, instaed we collect a bunch of examples, we get the system to output something efficiently. But we only care about that things that can be done with an efficient system. If there's no efficient system to do something, then learning is going to be useless to me. If the programmer can't do it anyways, it's too hard. So we only care about predictors that can be implemented efficiently. And that actually has small capacity, so it can be learned efficiently! So everything we care about can be learned efficiently. If you look at programs of length at most \\(\mathcal{O}(T)\\). So only \\(2^T \\) programs, so capacity is log of that, so capacity is only \\(T\\). 

So why aren't we using ERM as a learning rule --- find a short program that fits the data well. The problem is that ERM over short and fast programs is very hard!!! It's incomputable!! So we can't do that. It's actually easy to get out of this incomputability --- you can change it to programs that are not allowed for more than time \\(T\\) --- then it becomes computable, but it's NP-complete. Most people think it's bad news. I think it's good news! It means if P = NP, then we have universal learning! Otherwise, there's no polytime algorithm for ERM. That doesn't necessarily mean it's not learnable efficiently though; maybe some other learning rule works. That's unlikely to be true since if cryptography works (existence of one-way functions), then we can say there's no polynomial algorithm for learning this class. So no-free lunch situation is a more refined situation. What's more important than the statistical no-free lunch is the computational no-free lunch, since we only care about classes which are efficiently computable so we get a free statistical lunch, but the learning rule is not computationally efficient, so we don't necessarily get a free computational lunch. 

Now inductive bias is a assumption or property of the data distribution under which a learning **algorithm** which runs efficiently and ensures good generalization error. So just saying the complexity is low is not sufficient, since we care about efficient implementation as well. So inductive bias is about generalization aspect, and you cannot separate this from the computation aspect. Because if you ignore computation, you do have a free lunch. 

Let's go back to neural nets. In terms of expressive power and generalization ability, they're amazing, we can do anything we want. The only issue is if we turn to computation. We don't have any method that always succeeds in training a neural net --- you can prove that training a tiny net with only two units is NP hard. But that shouldnt' really bother us, because maybe something other than ERM works. We know that's not possible, because then we'd be able to learn all functions representable with time \\(T\\). So this thing that looks like a positive result is actually a negative result, because if you can learn everything, you can get a free lunch, and we know that you probably can't have this free lunch (though you get one today and tomorrow). 

We have extremely strong results about the hardness of learning neural networks. If you have a single layer and super-constant number of units, there's no efficient algorithm that will succeed in learning. So not only can you not solve ERM, just because your data is representible by a small network, this is not a sufficient inductive bias to ensure learning. Nevertheless, people are using deep learning, and lo and behold, it actually works! There's some magic property about reality, such that under this property, neural net training actually works. I would claim that we don't have the faintest idea (hopefully this week maybe some people will have a faint idea) what it is. Well, it is NOT that it can be represented by a small neural network (Amit Daniely) --- this inductive bias is not sufficient. So there's some other inductive bias,and I'd really like to know what this inductive bias is. 

If I gave this talk a three or four years ago, I'd probably stop here and say that's the most important question. But some experiments I did several years ago caused me to rethink this. What we did is as follows: We trained a bunch of networks with increasing size. When you get larger and larger error, training error goes down. Test error goes down. What happens after the point that training error has reached the minimum? We expect the test error to go up. But that is not what actually happens! Instead, test error goes down. So we're looking at the wrong capacity measure -- it's not measured by the size of the network, even for the statistical aspects of the problem. This had been observed before for boosting --- the classical explanation of boosting was depending on the number of base predictors. But then people realized the \\(\ell_1\\) margin was the real complexity measure, back in 1998. Actually for neural networks, back in 1996, there is work saying that for neural networks, we knew to look at the size of the weights, that's the real capacity control. 

What we're saying is there's a huge parameter space of functions that we cannot generalize. But reality is actually kind of small. So we only should care about sublevel sets with respect to the norm, and there, we can get generalization. By increasing the size, maybe the norm shrinks. This explains how the training data can fit anything, but generalizes differently depending on what you do. So we can measure the norm of the networks. We expect to see the norm decrease. But actually, the norm increases! This does not explain generalizability. 

So what's going on here? Here we looked at the sum of the squares of all the weights. This is actually not a good norm to look at for neural nets. If we look at a path norm instead, then lo and behold, it actually does go down. We have to talk about which norm? which complexity measure? With different inductive biases, maybe we can actually fit the reality with a very small capacity. So what is the real complexity measure, that's one thing we should ask. 

Now we should ask another question. How is it happening we're minimizing a certain complexity measure? With a thousand units, there are more units than training examples, so there are many ways to minimize. How do you choose? Some have small path-norm and generalize well, but others don't and don't generalize. So how do we minimize path norm? All we did was run gradient descent! So it seems optimization is implicitly biasing us towards minimum path norm. That means if we change our optimization algorithm, we're changing our implicit bias. So Path-SGD gives a minimum that generalizes much better. If you look at SGD and Adam, it turns out that SGD generalizes better, even though it takes a bit slower to get to ERM. 

A simple example is least squares. My claim is that if you do SGD on least squares, you'll get the solution minimizing the Euclidean norm. So here the optimization algorithm is inducing the complexity measure here. All the inductive bias is happening through the optimization algorithm. So much of the bias in deep learning is due to the optimization algorithm selected. What I'm claiming is that machine learning algorithms that don't tell us what the training algorithm details are don't tell us anything about the learning algorithm. So we're optimizing over the space of all possible functions effectively, but the question is "which of those functions will we actually choose, with respect to the optimization algorithm?" You should think about it as a landscape of many zero-error solutions. Depending on where you start, you may end up at different places. Some places are really flat -- then it really matters what preference the geometry of the optimization algorithm is. You'll get to close minimum with respect to the geometry of the optimization algorithm. 

So the ultimate question here again is what is the true inductive bias of deep learning? The approach I'm suggesting is to identify a complexity measure with a few properties: 

* reality is well-explained by something with low complexity 
* to quantify what low-complexity means, have to talk about capacity 
* our learning methods and optimization algorithm bias us towards low-complexity

If you explain all these things, then you have explained the problem. 

The mathematical questions are: 

* What is the capacity of sublevel sets of complexity measure, but it doesn't tell me anything if reality cannot be captured of small-capacity
* For specific algorithms, understand the inductive bias it implies. 

Then, there's a question about reality --- when does reality have small complexity? How can you possibly solve this question about reality? The best you can do is all kinds of empirical investigations. This is what we do in science. Do the models we learn have small complexity? Can they be used to explain generalization? 

I have some answers to these questions. There will be talks by Surya and Jason Lee on Tuesday that will give lots of answers; many others too I am sure, these are just my collaborators. 

There's another question: Under what conditions does local search work? Do we have an inductive bias that explains this? 





### Optimization Landscape: Symmetry, Saddle Points and Beyond (Rong Ge)


#### Introduction

We will start by looking at some of the simpler cases that we know how to analyze. It's very common to run into nonconvex optimization problems. 

There's a difference between theory and practice for nonconvex optimization. Classically the idea is nonconvex optimization is NP hard, so we'll do a convex relaxation. In machine learning, people tried to solve these methods with simple algorithms like SGD. These algorithms seem to work well in practice, so why is that? It's not that these algorithms are magical, they're basic optimization algorithms. It's known that gradient descent will converge to a stationary point, even for nonconvex functions. More recently we were able to show that it can converge to an approximate local minimum, but that's really all it can do. It cannot escape from a local optimal solution. 

#### Optimization Landscape 

The shape of a convex function is very simple. Zero gradient implies global minimum. So convex functions can be optimized efficiently using a large variety of algorithms. In general, the shape of the objective can be very complicated --- there are things like saddle points and so on. Here you can only hope to find a local minimum. Since gradient descent actually optimizes many nonconvex functions in practice, what are the special properties of these nonconvex functions easy to optimize? 

I want to also ask a slightly different question: The question that got me thinking about these problems --- why are the objectives always nonconvex? It's almost always the case that whatever objectives you come up with are always nonconvex? Can I change my objectives to become convex? The answer is not likely: Symmetry within the objective is one reason why your objective has to be nonconvex. 

In many machine learning and other problems, the following is something typical: The problem asks for multiple components, but the components have no particular ordering within themselves. Consider clustering. You can think of the solution as \\(k\\) centers (each point should be close to one of the centers). But you don't care about the ordering. 

Why does this kind of symmetry give you nonconvexity? Given an optimal solution, it's easy to come up with a second solution that has the same objective value. If we have convexity, we know that convex combinations of the solutions will also have a good objective value. But that is not the case for nonconvex. Optimization needs to break the symmetry. 


One special structure that only happens with nonconvex functions is saddle points (flat regions). Now another thing that symmetry implies is saddle points. For instance, consider \\(f(x) = -\|x\|_2^2 + \|x\|_4^4 \\). 

Things similar to this are used in things like ICA. This has 4 symmetric minima. You can get saddle points out of this kind of symmetry by connecting the minima with paths. Along these paths you can often find a saddle point. This doesn't happen for every path connecting locally optimal solutions. The paths which minimize the maximum objective function on the path, then the hig. hest point on the path is going to be a saddle point (or a local maximum, but for us, that will be the same thing in the cases we consider). 

So we need to design optimization algorithms that can escape from saddle points. 

I've often gotten the following question: If symmetry is the only problem you're worried about, why can't you add inequality constraints to break the symmetry? Maybe now there is a unique optimal solution you're looking for. That doesn't always work very well -- suppose you restrict to one quandrant. A point on the border of the quadrant will actually want to converge to a different point, not in the fourth quadrant. So in many cases adding inequality constraints actually makes the problem harder. 

It is indeed possible, if you knwo the invariance of your objective, then you can do something to account for all the invariances (people have done this for low-rank matrix problems that I will add next). But you can't simply add inequality constraints. 

#### Locally-Optimizable Functions 

For locally-optimizable functions, their local minima are symmetric versions of the global min, and the symmetric group is important to look at. 

You also have to assume that there's no high order saddle points --- these have zero gradient and PSD Hessian. Most functions don't have high order saddle points. Once you have these conditions, it's possible to prove that a large variety of algorithms including SGD will always be able to converge to a local minima, and by the first property it's also a global optima. The first constraint is of course very strong. Why do we believe they're global optima? But many interesting problems actually satisfy these properties, including SVD/PCA, matrix completion, max-cut, 2-layer neural net, GLM, dictinoary learning, matrix sensing, and tensor decomposition. 


#### Matrix Completion 

A goal is to recover remaining entries of a matrix given some of the entries. We believe the matrix is low-rank because we can think of decomposing it into two smaller matrices. Each row of the first matrix can be thought of as a r-dimensional feature for the user, and each column of the second matrix can be thought of as an r-dimensional feature for the movie, for instance, in a Netflix recommendation system problem. 

There are 2nr parameters, and we hope to solve the problem with roughly this term. 

The simplest way to try to solve is to write a nonconvex objective down and find the low-rank variables directly. We write \\(M = UV^T\\) and try to recover \\(U, V\\). We assume uniform entries observed from the matrix. The optimization variables are \\(X, Y\\).

The first objective might be to look at the \\(\ell_2\\) loss on observed entries. We call this sort of a norm, because if you observe all the entries of the matrix, this is the Frobenius norm. The function is not convex because there are symmetries in the solutions. There are many equivalent solutions (inject orthonormal matrices in between). There are also many saddle points. 

We have a theorem: when number of observations is at least \\(nr^6\\), all local minima are global. Then you can just run simple stochastic gradient descent from an arbitrary starting point. Prior work was in two categories: One approach is convex relaxation (these have good dependencies on the rank). The second set of results look at nonconvex optimization algorithms, but require carefully chosen starting point, but have better dependence on the rank. Improving \\(nr^6\\) is an open problem for the general nonconvex setting. 

What tools do we use? We look at the optimality conditions. We use second order necessary condition: gradient is zero and hessian is not PSD (then you can follow negative eigendirection and function value will decrease). We want to go from here to show that the point satisfying this will be a global optima. We do this backwards, looking at the contrapositive. We start at x is not a global optima, then we show that there is a direction that either correlates with the gradient or with the negative eigendirection (a direction of improvement). So if we can find a direction of improvement for every point that's not a global min, then we have a proof that every local optima is global optima, since we know how to improve everything else. 

In order to solve matrix completion, we first look at simpler problem: matrix factorization, where every entry is already observed. So we just want to factorize this matrix as two smaller matrices. Consider the symmetric case \\(M = UU^T\\). Objective will be in terms of Frobenius norm. The goal is that all local minima satisfy \\(XX^T = M\\). In matrix factorization case, there's a simple proof: 

* Look at gradient and set it to zero. Then this implies \\(MX = XX^TX\\). 

* If the span of the columns of X is the same as span of columns of M, then it's possible to show that \\(M = XX^T\\) based on the gradient condition. 

* Problem: if the spans are not equal it won't work. But if the Hessian is PSD, we can show the spans are equal. 

To generalize to matrix completion, it's kind of difficult for it to work for matrix completion. Many more cases. So we do something different. 

Intuitively we want X to go to the optimal solution U. But there are many equivalent optimal solutions. So which should I go to? Intuitively, should go to the closest global minimum among all the optimal solutions. Define \\(\Delta = X - UR\\), where \\(R\\) is the closest solution. \\(\Delta\\) is the direction of improvement. It has a nice property that \\(\|\Delta\Delta^T\|_F^2 \leq 2\|M - XX^T\|_F^2\\)

If this is true in the matrix completion norm (with a 3 instead of a 2), then either it's a direction of improvement or the Hessian at this vector is negative. 

For matrix factorization, we immediately have a proof using this improved lemma. It's easier to generalize to the matrix completion case --- because essentially, Frobenius norm and matrix completion norm will be close to each other. You only need to this to be true for matrices of the form \\(\Delta\Delta^T\\) or \\(M - XX^T\\) --- these are low rank. So you just need to preserve the norm of low-rank matrices. So the proof immediately works for close enough matrices. This property is known as the Matrix RIP property. This can be applied to many other problems related to matrices. It applies to asymmetric cases, matrix sensing, and robust PCA. 

There are open problems, mostly about low-rank matrices and low-rank tensors. In between, tensor Tucker decomposition and linear neural networks. 

Can we quickly determine whether an objective is locally optimizable, if the loss has some symmetry. 

#### Optimization landscapes for neural networks 

This is actually much more complicated than all the cases we saw before. It's not the case that the local optima are generated by these symmetry transformations (it plays a role, but not the whole story). We will only consider fully-connected neural networks here. For simplicity we will consider \\(\ell_2\\) loss. Most things I will talk about can be extended to other loss functions. We will talk about the teacher-student setting. Our goal is to prove something about optimization. In order to not get confused with the other problems that Nati talked about, we will assume there's already a good teacher network from whcih we can get samples, and we assume we have enough samples so that generalization isn't a problem. The student network is just trying to mimic the behavior of the teacher network. Our only problem is how to find the solution -- we are in the realizable setting. Here we just try to focus on only the optimization problem. 

#### Linear Networks 

This is just a product of a bunch of matrices. You don't need multiple layers, but it's interesting to look at what we can do if we don't have nonlinearity. 

There are a lots of results on this. All local minima are global, but each work has different technical assumptions, and they're not comparible, and it's also not clear the assumptions are necessary. With 2 layesr, there are no higher order saddle points. With 3 or more, it does have higher order saddles. For critical points, if a product of all layers has rank r, then it is local and global optimal or a saddle point. An open problem for linear networks is: does local search actually find a global minimum? 

#### Two-layer neural networks 

You have an input and it goes to a hidden layer. WLOG the rows are unit norm, and the output is going to be a hidden function on the hidden units. We will make some strong assumptions like the input data comes from a standard gaussian, and the y's are generated by a teacher network. The goal is to recover the true parameters. To avoid the lower bound examples that Nati mentioned, there are more technical assumptions -- you need to assume that the weight matrix in the first layer is full rank. 

First if you do the naive \\(\ell_2\\) loss, you will run into bad local minima. We empirically observed in our paper, and it was formally observed in a later paper. 

So how can we deal with local optima? We're not tied to particular loss function, we could design a new one with no bad local minima. This is not a new idea, it's included in many previous techniques, including regularizaiton, using method-of-moments instead of maximum likelihood. 

There's a provable new objective: can construct an objective for two-layer neural networks such that all local minima are global. This kind of heavily relies on Gaussian distribution; the objective is inspired by tensor decomposition. Later we extended to symmetric input distribution, and then the algorithm is no longer a local search problem anymore. 


#### Spin-glass models / Kac-Rice

The main claim is that maybe all the local minima are all approximately good/ close to the local optima. They ran some experiments on MNIST which suggest this is the case. The proof idea is to use a formula to count the number of local min that satisfy the property directly. There is such a formula (Kac-Rice) formula. You can use this to compute the number of local minima in any region, but this is a complicated looking formula --- how can you even hope to evaluate this? You evaluate it with random matrix theory in the good cases (e.g. Gaussian ensemble). Then you can hope to evaluate it with random matrix theory. They did this in the paper with spin glass model (random polynomials over the sphere). 

Kac-Rice was also used to prove results for overcomplete tensors and also for Tensor-PCA. But usually it's very difficult to evaluate this formula. 


#### Analyzing the dynamics of local search

The idea is that instead of analyzing the global optimization landscape, we will try to analyze the path from a random initialization. Why is this easier than analyzing global optimization landscape? The main observation that many works have is that in certain cases (esp. for overcomplete neural networks), there are global optima everywhere that are very close to where you start. So you can hope that the paths are very short to the global optima, and you can use the properties of the random initialization. A lot of results of this type just came out this past year. They are in different categories: One focused on empirical risk over the training error. Suppose the network is highly overparameterized, then it can achieve zero training error. It was proved for 2-layer, and then generalized by a few groups. When you consider generalization, the problem is much harder. If you look at a special multi-class classification problem you can do it, and a more general result does it for 2-3 layer networks. The result is kind of kernel like (the network under their assumptions can be approximated by low-degree polynomials, and requires special activations to run in polytime). But the common idea is you can analyze the path from random init to a global optimization, and that path will be short. 


I'm going to end on some open problems: 

* Overcomplete Tensors: Suppose you have \\(a_1, \cdots, a_n \in \mathbb{R}^d\\, and \\(d << n << d^2\\) (overcomplete setting). Then we want to maximize the sum of the fourth powers of the inner products over the unit sphere. The conjecture is that all local maxima are close to plus or minus \\(a_i\\). Empirically, gradient ascent always finds one of the \\(a_i\\). Earlier we were able to use Kac-Rice to show that there's no other local maxima if the function is at least \\((1 + \epsilon)\\) times the expected value of the function. Proving you can't have local optima below this threshold is open. 

* Overcomplete Neural Networks. We again have teacher-student network. Say you have a teacher network with only 15 neurons. We find that if a student has 100 neurons, then you won't get stuck in a local optima. But if the student also only has 15, it gets stuck. Suppose the teacher has k neurons, and the student has poly(k, epsilon) neurons, then I conjecture that all local min have an objective value of at most epsilon. 

### Mean Field Description of Two-layers Neural Networks (Andrea Montanari)

We have a two layer network and we are going to minimize square loss with SGD. We'll assume the step-sizes are a small constant. n is the number of samples, and N is the number of hidden units. How can we analyze this? 

Ok so now we have parameters n samples, N units, dimension D, and k steps. People have started to understand some of these regimes, we will describe the picture. First we'll look at the case where you only have a constant number of neurons \\(N = O(1)\\), \\(n \geq D\\), \\(k \geq n\\). I'll call this the small network regime (small number of neurons). You can study a lot of things in this setting. People studied this using spin glass techniques; there was a nice paper last year by Auburne et. al. using statistical mechanics. There's an interesting phase transition. The second regime that was studied is where \\(N > d^c, k \geq n^{c'}\\), where \\(c, c'\\) is some power (imagine like these powers are 6). We call this the overparametrized regime or kernel regime. What happens is what Rong was describing this morning; here SGD needs to take only few steps to get to the optimum, and only the linearization matters (you end up doing something very similar to kernel ridge regression). The initialization of the random rates is very important. 

Finally, we have the **mean-field regime**. We have a large number of neurons \\(N \geq D, D \leq k \leq n\\). Here you only visit each neuron a few number of times (in this talk, just once). Here the dynamics are nonlinear, but you try to take advantage of the large n to simplify the description. 
Here we take \\(k << nD\\). We will focus on this regime for the rest of the talk. 

Now we'll focus on the mean-field regime. We'll also assume no noise for simplicity. A good point to start is the universal approximation. Barron's theorem says the optimal risk is bounded by \\(\frac{1}{N} 2\sigma \int_{\mathbb{R}^d}||w||\mathcal{F}(w)dw \\) where we're taking the Fourier transform. We can relax the function class we consider \\(\hat{f}(x; \rho) = \int \alpha(x, \theta) \rho(d\theta)\\), where we think of \\(\alpha\\) as Fourier coefficient, and where we want the measure \\(\rho\\) to be approximated as a sum of \\(N\\) delta functions, when \\(N\\) is very large.

Now we want to understand what is learned by SGD. So can we study the evolution of the density of the empirical distribution? So the key object we want to study is 
$$ \hat{\rho}_k^{(N)} = \frac{1}{N}\sum_{i = 1}^N \delta_{\theta^k}$$ 
This is the natural way to represent the network, this is invariant under permutation --- you have factored away the permutation by writing it as a sum (and a probability distribution). 

So how does the distribution given above evolve? 
We'd like to write down some kind of ordinary differential equation for this object. We can write down a partial differential equation. We need to have evolution in an infinite dimensional space, this is a PDE. It is 
$$
\delta_t \rho_t = \nabla (\rho_t \nabla \Psi(\theta, \rho_t))
$$
$$
\Psi(\theta, \rho) = V(\theta) + int U(\theta, \hat{\theta}) \rho(d\hat{\theta})
$$
where \\(V(\theta) = \mathbb{E}[y\alpha(x, \theta)]\\) and \\(U(\theta_1, \theta_2) = \mathbb{E}[\alpha(x, \theta_1)\alpha(x, \theta_2)]\\). 

Now this is a bit scary, but we can get some intuition from it. 


### Langevin Diffusions in Non-convex Risk Minimization (Maxim Raginsky)















