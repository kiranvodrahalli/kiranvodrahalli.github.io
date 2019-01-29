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

Now we'll focus on the mean-field regime. We'll also assume no noise for simplicity. A good point to start is the universal approximation. Barron's theorem says the optimal risk is bounded by \\(\frac{1}{N} 2\sigma \int_{\mathbb{R}^d}\|w\|\mathcal{F}(w)dw \\) where we're taking the Fourier transform. We can relax the function class we consider \\(\hat{f}(x; \rho) = \int \sigma(x, \theta) \rho(d\theta)\\), where we think of \\(\sigma\\) as Fourier coefficient, and where we want the measure \\(\rho\\) to be approximated as a sum of \\(N\\) delta functions, when \\(N\\) is very large.

Now we want to understand what is learned by SGD. So can we study the evolution of the density of the empirical distribution? So the key object we want to study is 
$$ \hat{\rho}_k^{(N)} = \frac{1}{N}\sum_{i = 1}^N \delta_{\theta^k}$$ 
This is the natural way to represent the network, this is invariant under permutation --- you have factored away the permutation by writing it as a sum (and a probability distribution). 

So how does the distribution given above evolve? 
We'd like to write down some kind of ordinary differential equation for this object. We can write down a partial differential equation. We need to have evolution in an infinite dimensional space, this is a PDE. It is 
$$
\delta_t \rho_t = \nabla_{\theta} (\rho_t \nabla_{\theta} \Psi(\theta, \rho_t))
$$
$$
\Psi(\theta, \rho) = V(\theta) + int U(\theta, \hat{\theta}) \rho(d\hat{\theta})
$$
where \\(V(\theta) = \mathbb{E}[y\sigma(x, \theta)]\\) and \\(U(\theta_1, \theta_2) = \mathbb{E}[\sigma(x, \theta_1)\sigma(x, \theta_2)]\\). 

Here we're kind of taking the gradient of a measure. There's a sense in which this makes sense (you integrate it against a test function). 

Now this is a bit scary, but we can get some intuition from it. We will give a sort of law of large numbers theorem. 

**Theorem**. Assume that \\(\sigma\\) is bounded and the outputs are also bounded, as well as the gradients of \\(V, U\\) are bounded and Lipschitz. 
Then say you look at \\(W_2\\) distance. Then the distance between empirical distribution and the solution to the PDE is smaller than \\(e^{c(t+1)}err(N, D, \epsilon)\\), and the same is true for the risk. 

So to reiterate, the solution to the PDE and the object we're interested in (empirical distribution) explodes with time, but for constant time it's fine. 

Now what is this error term? 
$$
err(N, D, \epsilon, z) = \sqrt{\max(\frac{1}{N}, \epsilon)}(\sqrt{D + \log(N/\epsilon)}) + z
$$
where z is just a constant that doesn't matter (just for high probability bound). Simplifying, we look at the intereting parameter regime \\(N >> D, \epsilon << 1/D\\), which for \\(\epsilon\\) is better than \\(1/ND\\), which is what we might expect if we think about number of parameters. So the number of samples we use is \\(n = T/\epsilon = O(D) << ND\\). This is surprising somehow. 

#### How to prove it (intuition behind the idea)

There are two steps:

* Approximate SGD by gradient flow on population risk. Here you're mostly using \\(\epsilon\\) small. 

* Approximate gradient flow on population risk by the PDE. 
You can write the gradient flow on population risk as the part of the PDE where \\(\Psi\\) is defined. Nonlinear dynamics is one way to do this. This type of proof has a long history in mathematical physics, because people were interested in systems of particles, and can be traced back to de Brujin. 

Nonlinear dynamics is the following problem: Look at a single particle that evolves according to the gradient of \\(\Psi\\). Then it's possible to show that the nonlinear dynamics is equivalent to the PDE.  
To go from nonlinear dynamics to PDE, you create a simple coupling. 

The crux of the matter is of course looking at concentration on the empirical distribution --- there's considerable benefit from the fact that we're looking at a some of terms here. This is where some of the "magic" happens. 

#### What can you do with this? 

Can we prove approximate global convergence? 
Well we could carefully look at landscape and avoid getting stuck (Rong from the morning). 

This gives a new philosophy. We try to prove something about the behavior of the trajectory. So we'll get approximate convergence. We have a two step strategy: 

* Prove convergence of the risk of \\(\rho_t\\). 

* Connect back by using the general theorem. 

The general upshot of this theorem is that it gives good results for finite dimension. But the dependence on \\(D\\) is not really well understood --- there are bounds, but not good bounds. 

What are the fixed points of the PDE? We will get stuck in fixed points. A fixed point is the derivative with respect to time is 0. The converse is not obvious but is true. This implies that rho is inside (supported on) the set of \\(\theta\\) such that the gradient with respect to \\(\theta\\) of \\(\Psi\\) is 0. 

If you're a physicist this makes sense --- you have a fluid particle, each particle feels zero force. Now what are the global minima? The loss function is quadratic in \\(\rho\\); \\(\rho\\) lies on the probability simplex. Then you can apply Lagrange multipliers, then the global minima such that the 
\\(supp(\rho) \subseteq arg\min_{\theta} \Psi(\theta, \rho)\\). So not all fixed points are global minima. But these equations are not too far apart. These two conditions are not too different, so proofs of convergence will go through. 

Intuitively, consider fixed point \\(\rho^*\\) that is supported everywhere. 


We assume that this is a well-behaved probability distribution. If you can avoid \\(\rho_t\\) becoming degenerate, then you can get global convergence. 




### Langevin Diffusions in Non-convex Risk Minimization (Maxim Raginsky)

We wish to minimize some function \\(f\\) of d variables; f is a finite average over the data. We want to minimize f. We're just going to talk about minimizing f somehow. Gradient descent is simple and it's a procedure of choice. We'll for the most part work in continuous time. We'll write 
$$ 
dX_t = - \nabla f(x_t)dt
$$
$$
X_{t + dt} - X_t = U_t
$$

We know that if \\(t > s\\), \\(f(x_t) \leq f(x_s)\\). With a simple calculation we can see that the derivative with respect to time is non-positive. However, we can still get stuck somewhere. Once it's in a stable equilibrium, you're stuck. The initialization determines your fate forever. 

You can actually integrate this and turn it into a limit. But it does tend to get stuck in local minima. 

How can we avoid that? One weird trick that works is to add a bit of noise to the gradient step update. Say, a Gaussian centered around the usual update. How much variance should we add? We have a spherical covariance matrix, but its size is proportional to dt and inversely proportional to \\(\beta\\), which is inverse temperature (e.g., simulated annealing it's 1/temperature). The point is that you take a random step. What this does is it prevents you from getting stuck, a nonzero chance of escaping. You add this noise between times t and t + dt. Why would you want this noise to be independent of the past noise? This gives you white noise, but why would you want this to be white noise? Here's how I understand it: Essentially, suppose that you're very close to a local minimum, and the gradient is basically zero. What you want is multiple attempts to escape. If they're independent, it's a geometrically distributed random variable with a finite mean. There's large deviation theory for dynamic systems that makes this intuition precise. You can then show that the time of escape from each basin of attraction is finite in expectation, and you have an exponentially distributed random variable if you normalize by the mean, which ties it back to memoryless process. 

So this all leads us to think of d-dimensional Brownian motion: we know increments are both independent and Gaussian. As a result, we end up with a random evolution. We can thus write down a stochastic differential equation. 
$$
dX_t = \nabla f(x_t)dt + \sqrt{\frac{2}{\beta}}dW_t 
$$

This was developed by Ito, who thought in terms of evolution in the space of measures, continuous time Markov processes that are analogues of ODEs. You have to define a vector field for this, and in the case of diffusions, reduces to stochastic differential equations. Since Gaussians are defined by first two moments, you can think of a flow for the mean and a flow for the covariance. 

When you integrate this you get a random process. This particular equation (after integrating) is called the **overdamped Langevin equation**:
$$
X_t = X_0 - \int_0^1 \nabla f(x_t)dt + \sqrt{\frac{2}{\beta}}\int dW_s
$$

Now, Brownian motion is nowhere differentiable. So Langevin introduced a damping term in 1908; the velocity is determined by the momentum, and the momentum is evolving according to an SDE. When you take the friction to infinty, you end up with something like this. But this was all physics. This was a realistic description of diffusion in some potential energy field such that you can meaningfully talk about random energy of a particle. 

Fast forward several years and we end up with papers by Guidas (1985), Geman-Huang (1986), and they said you can use this for global optimization! You can say you'll eventually visit all basins of attraction, so you'll eventually get to the best one. 
There were variants of annealing where you take \\(\beta\\) to infinity under a schedule, also constant temperature variants. 

In the context of Bayesian ML and MCMC, in 1996 people would think of f as proportional to some posterior that you want to sample from. You decompose this using Bayes rule and run your Langevin dynamics. It has a rich history. Let's fast forward and think in terms of machine learning: We have SGD. You can think of SGD as true gradient + gaussian fluctuation (via central limit assumptions). Now the variance is dependent on the position. Max Welling and Ye Whye Teh brought Langevin dynamics back to ML and things took off from there in 2006. 

The idea here is that you can quantify all sorts of things if you're working in continuous time. 

This is called Friedlin-Wentzell theory (late 80s). We will explicitly indicate \\(\beta\\) here, because we want to see what happens when \\(\beta\\) gets cranked up: 

$$
dX_t^{\beta} = -\nabla f(x_t^{\beta}) dt + \sqrt{\frac{2}{\beta}}dW_t
$$
We're interested in the probability that 
$$
P(\sup_{0 < t \leq T} \| X_t^{\beta} - X_t^{\infty}\|_2 > \delta) = P^{\beta}(\delta)$$

Then if you normalize this probability by \\(\beta\\), then as you take \\(\beta\\) to infinity this goes to zero. But you can still exit basins of attraction!


Now the issue is that you want the amount of time it takes to find the optimum to be small. You generically expect exponential time. Can you quantify how long it takes? Large deviation says you'll get there, but it could even be doubly exponential in dimension. 

#### Global optimality in finite time

This is a work I did with various people at COLT 2017. You can show that as time goes to infinity, the law of \\(X_t\\) converges to a distribution \\(\pi(dx) \sim e^{-\beta f(x)} dx\\). 

Under some conditions on f, you can show that if you sample from this \\(\pi\\) and look at the expected value of f at that sample. The difference between this and the value of f at the global min is roughly \\(d/\beta \log (\beta/d)\\). "What's a few logs between friends". We showed the Wasserstein-2 distance between \\(\mu_t\\) and \\(\pi\\) is bounded by \\(c\sqrt{KL(\mu_0 \| \pi)}e^{-t/c_{ls}}\\) where \\(c_{ls}\\) is the log-sobolev constant, which is unfortunately known to be exponential in dimension generically. 

We needed some conditions: 
* The gradient of f is Lipschitz
* dissipitivity: there exists positive m and b such that \\(\langle x, \nabla f(x) \rangle \geq m\|x\|^2 - b\\). 

This was based on some Lyapunov function techniques from probability theory. You can show this process is bounded in \\(L_2\\) and convert it to a Wasserstein bound. On the other hand, this takes a long time. There was a paper by Yuchen Zhang, Percy Liang, and Moses Charikar that shows a discretized version will hit in polynomial time. So there's a discrepancy between these things. 

#### Metastability 

Suppose you initialize your diffusion so that it happens to lie within some local minima. Suppose it's smooth, dissipative, and also Lipschitz-Hessian. The intuition behind the result I'm about to state is simpler than the previous proof. Suppose you initialize your diffusion; it happens to lie in some ball around some non-degenerate local minima. You can linearize your gradient around that local minima. Suppoes it's going to be in a ball of d dimensions of radius \\(\sqrt{b/m}\\) (follows from dissipitivity). Let \\(H\\) be the hessian. Let \\(\tilde{X}_t = X_t - \bar{x}\\). Then 
$$
d\tilde{X}_t = -H\tilde{X}_t dt + \sqrt{2/\beta}dW_t + \rho(\tilde{X}_t)dt 
$$

The first two terms are very well understood: This is Ornstein-Uhlenbeck process. WLOG you can assume eigenvalues of Hessian are lower bounded by m. The asympotic mean is zero, and the variance is controlled by \\(\beta\\). It turns out you can take this process, and use an idea from the theory of ODEs. You run time backwards by doing \\(X_t = e^{-Ht}X_0\\), where we're using definition of matrix exponential. 

Suppose that you want to spend time T around local min with high probability. Define \\(T_{recurrence} \sim (1/m)\log(r/\epsilon)\\) and \\(T_{escape} \sim T_{recurrent} + T\\).
Then if \\(\beta \geq \frac{d}{\epsilon^2}(1 + \log (T/\delta))\\), you can control the length of the metastability phase. You want to explore all the minima but tune the time that you explore them. T will tend to be exponentially large when you go to discrete time, but in continuous time, it's clear: You transition out quickly and transition to another basin quickly, but if you hang out for a certain amount of time, you'll hang around a bit more and you can control this by the parameter \\(\beta\\). 

Since then there have been many works working with the Hamiltonian Langevin dynamics (with momentum), you can use it for sampling. The exponential dependence on time was improved into something like exponential dependence on condition number of the Hessian. Discretization I didn't talk about, because that's a whole can of worms. But I think Ito calculus understanding is worthwhile in the context of deep learning. 

## January 28

### Convergence of Gradient-based Methods for the Linear Quadratic Regulator (Maryam Fazel)

#### Introduction 

I'll be talking about linear quadratic regulator and control. This is joint work with Rong Ge, Sham Kakade, and Mehran Mesbahi. I will look into LQR. Our motivation is to understand policy gradient methods, a class of methods very popular in RL, on a special and simple problem. I'll look at the case where we have access to exact gradients, and then the case where we don't have exact gradients, and then look at some extensions to this framework. 

The LQR is the problem of controlling a linear dynamical system with quadratic costs. We look at the problem in discrete time with infinite horizon. Here, \\(x_t\\) is the state of the system at time \\(t\\), \\(u_t\\) is the control applied at time \\(t\\). We have this dynamical system in LQR, and we want to choose the \\(u\\) given an initial state \\(x_0\\) in order to minimize a quadratic cost function of the inputs and the states (made of positive semidefinite matrices). The goal is you want to drive the dynamics of the system to some desired final state, and you want to find the minimum effort (in \\(\ell_2\\) cost sense), in the infinite horizon case. 

This problem has been well studied -- it's a dynamic program. You want to solve for every \\(u\\) with a cost-to-go function. It can be solved in closed form via defining the Riccarti variable \\(P\\), which is derived by algebraically solving the Riccati equation. It's possible to do this with linear algebra and iterative methods. Then the problem is solved since we can prove that optimal \\(u\\) is a static state feedback \\(Kx_t\\), where \\(K\\) is derived from the Riccati equation. Interestingly here, in infinite horizon, \\(K\\) is fixed with respect to time and this is optimal. This is a cornerstone of optimal control going back to Kalman. Recently people have tried to connect back to reinforcement learning methods which are popular these days in robotics. This is a nice setting to examine those problems because it is simple. 

All the methods assume you have given dynamics and costs. So all the methods rely on solving \\(P\\) first and then finding optimal control. 

We want to see can we solve the LQR problem by algorithms that iterate on the policy (e.g., \\(K\\)). Here we only use the cost of the policy. Suppose we first consider methods that have first-order (gradient) access to the cost function, either exactly or approximately. Does gradient descent, even with exact gradients, converge? If it does converge, under what assumptions? Does it converge to globally optimal \\(K^*\\)? 

If so, what is the rate of convergence in terms of problem parameters? 

What if we didn't have the parameters of the problem (e.g., model-free)? But we have access to function value samples of the cost of the policy? Can we still converge to \\(K^*\\)?

This wouldn't be challenging if it were convex, but it's nonconvex.

Why do we want to study gradient descent on \\(K\\)? Well it allows extensions (additional constraints, etc.) and it can be extended to noisy and inexact gradients. Once we understand gradient descent, we can consider derivative free methods, which are similar to policy gradient. It's a spectrum from knowing the model to not knowing the model. 

#### Existing Literature on Learning and Control 

There's a lot of work from the 1990s (Neurodynamic Programming), online control, adaptive control (usually only asymptotic, not finite-sample). The goals are different, some things are bounding regret, some are bounding error of estimation of the optimal control, and so on. Hardt et. al. 2016 is kind of related (they're solving system identification) but they're doing it by gradient descent on a nonconvex problem. Under their assumptions, the problem becomes quasiconvex. Other work which is related is Ben Recht's group's work. One approach is do full system identification up to some accuracy for all models identified in some uncertainty regime. Elad Hazan's group has done work on this too (learning linera dynamical system, and some work on control when the matrix is symmetric, some extensions to nonsymmetric too). The goal in these papers is a bit different though. Here there's only regret bounds on prediction error, not on learning the system. 

There's also from the control literature: It's about the known model case but they want to do gradient descent on controller \\(K\\) because they wanted to do structured controller design (e.g. projective gradient descent) but this is only empirically validated, so we would like to provide theoretical guarantees with gradient descent. 

#### LQR 

We will not consider state noise, and we will assume a random initial condition. Plausibly we can extend to having nois at every step, but this might be messy. We will call \\(\Sigma_K\\) the state covariance matrix and \\(\Sigma_0\\), which is just the covariance of the first step. 

#### Settings of Optimization Algorithms 

First we consider gradient descent on the cost and update \\(K\\) with fixed step size \\(\eta\\). We then look at natural gradient descent (which is just conditioned by an appropriate covariance matrix, inverse of state covariance is multiplied). 

We also have to define what oracle algorithm has access to. For first problem, it's just standard first order oracle (exact gradient oracle). We'll also consider approximate gradient oracle (noisy gradients, or only function values, which is close to zeroth-order oracle). 

This problem is hard because the cost as a function of \\(K\\) is not convex. 
It's related to the fact that a set of stabilizing controllers is not convex. 
It is also not quasiconvex or star convex. Starting from dimension three and up, it's completely not convex. 

#### First Order Oracle
We start by looking at stationary points of the problem (where is the gradient zero?) If the gradient zero, either \\(K\\) is optimal, or \\(\Sigma_K\\) is rank-deficient. This is helpful, because we can easily avoid the second case if we choose state from initial distribution that has covariance matrix that is full rank. Then it won't be an issue! 

We can also examine this via transformation to a convex linear matrix inequality, but proofs are not simpler. It's simpler to look at stationary points of the nonconvex function. 

Now suppose that \\(\Sigma_0\\) is full rank, as above. Then the function value at \\(K\\) versus \\(K^*\\)

is upper obounded by the norm of the gradient of the cost at \\(K\\) squared --- this is called gradient-dominated. So here, the rate of the norm of the gradient becoming small is the rate of convergence of \\(K\\) to the optimum. There's also some dependence on the minimum singular value (which relates to the condition number). 

Usually this is applied together with smoothness, but here the cost is not globally smooth. It blows up at places which are not stabilizing. However it's not too difficult to deal with this, because if you start from a stabilizing controller, you'll never hit a non-stabilizing controller. Putting this together with gradient domination theorem, you get the result. By assuming that the cost of the initial \\(K\\) is finite, you get the stabilizing condition, which is needed. We can thus prove a theorem that says that \\(K\\) gets \\(\epsilon\\)-close to the optimum. The dependence is on \\(\log(1/\epsilon)\\), e.g. a linear rate of convergence. It is possible to understand these rates a little better, but we do get linear rate when exact gradients are available. 

#### Unknown Model Case 

Usually we do not know the system. You can assume a simulator, which is typically costly to simulate. Or you have the ability to control by changing the control technique. These are partial information about your system. So now we will use this limited information regime. This is kind of "model-free", and mimics what people are doing in practice. In model-free estimation, you do multiple rollouts by perturbing the controller with Gaussian noise. This is also similar to zeroth order derivative free optimization, where you only get function values. 

What are the issues? How do you do this querying in a good way, what's the length of rollouts, what's the noise level you need to add, and what's the overall sample complexity (number of rollouts times length of each rollout). 

The controller we start with is given, and the number of trajectories is \\(m\\). Rollout length is \\(l\\), dimension \\(d\\), parameter \\(t\\). You draw a random matrix uniformly from a Frobenius ball constrained by \\(t\\). Then you sample policies, simulate each policy for \\(l\\) steps, and get empirical estimates of the costs and state covariance matrix.  Then you use certain estimates for the gradient of the cost at \\(K\\) and just average to get empiricial state covariance matrix. Here we're just uniformly sampling, possible you could do more intelligent things here. 

Here if we again assume we start from a stabilizing controller, then if we choose parameters all in poly(\\(\log(1/\epsilon)\\)), then we get \\(\epsilon\\)-close to the optimal and the number of samples is poly(\\(1/\epsilon\\)). So we get optimal dependence on \\(\epsilon\\) but the other parameters are not sharp, and we don't really try to do a sharp analysis. 

#### Proof sketch

* First we fix the rollout length. Then we have to show that the estimates we get on short trajectory is not too far. 

* Then show that with enough samples, the algorithm can do the gradient and covariance estimates. 

* Finally, show they converge with similar rate.

#### Structured Controller Design

Suppose you want to design a controller \\(K\\) when dynamics are known, and you want it to have a specific structure (e.g., specific sparsity pattern).  This is known to be hard in general. There's a special case that is convex under an assumption called quadratic invariance (very restricted assumption). There is also related work earlier that is empirical, that projects onto structured controllers, but is only empirical. We would like to use our tools to say something about structured controller design using gradient projection. This is ongoing work, but it's possible for some special cases to get some results. 


## January 29

### Theory for Representation Learning (Sanjeev Arora)

#### Introduction 

Can we learn good representations for a multiple downstream tasks with no labeled data? Hoe can one representation anticipate what you'll need in the future with low-dimensional representations. Semi-supervised methods assume you have both unlabeled data and labeled data, and performance is only good when you have some labeled data --- it's not quite doing representation learning in the way we want. 

Generative models for data are fully complete paradigm. It's unclear though why training log-likelihood should give good representations, Andrej Risteski and I have a blog post about it. 

#### Representation Learning in Practice 

Let me tell you about some interesting representation learning ideas that work well in practice. I'll mention two papers but there are others. The idea is this: train a convnet on the following task. The input is an image and its rotation by one of its possible rotations (90, 180, 270 degrees). Now the desired output is which of the three rotations were applied. They call this self-supervised learning. The resulting representations are quite good. You didn't need all those supervised classes here! This task seems trivial! Why should you need to learn any rotation to solve this task??! What the heck is this? The reason that trivial solution (rotate mentally and pick an exact match) doesn't get found cannot get implemented by the conv-net, so it is forced to do a semantic analysis of the image and come up with a representation. 


#### Contrastive Learning 

Another example in NLP for learning sentence representations is giving sentences which are adjacent have high inner product and those that are not low inner products. We will call such methods contrastive learning. 

The rest of the talk is based on "A theoretical analysis of contrastive learning", our recent paper. The framework will try to relate what are semantically similar pairs of data points. We assume that the world has a collection of classes out there. \\(\rho_c\\) is a probability associated with this class. Each class has a distribution \\(D_c\\) on datapoints. Similar pairs are picked according to some probability associated with classes and takes samples \\(x, x'\\) according to \\(D_c\\). Negative samples you just pick \\(c\\) according to \\(\rho\\) and then sample from a different class. 

When you're going to learn the representation it will be tested: You generate a binary classification task by picking a random pair of classes. To evaluate the representations, we pick the binary task, then you have to solve it by using a linear classifier on the representations. For this talk we'll assume logistic loss, but the theory extends to usual convex losses. 

What is the unsupervised loss of this representation function? We just use the contrastive training method described from before; there can be other training methods. This is over the full distribution of examples, in practice you see \\(n\\) samples and minimize the empirical loss. Now the important note is the amount of unlabeled data in the world is large and cheap, so we assume that the amount of unlabeled data is large enough. We can assume this because the representation class is some fixed class with fixed sample complexity. So if you have enough unlabeled data you'll learn a good representation. We ignore the computational costs of minimizing unsupervised losses in this work, it's for future work. 

It's very important --- for many years, for instance in the Bayesian world (and other settings) --- people tried to do unsupervised learning i.i.d. But you need some weak supervision empirically (e.g., some task like predicting next words, or predicting adjacency, etc.). 

What would be a dream result for analysis? The dream result is you do this weakly supervised/unsupervised learning and then you'd like to compete with a representation you could learn with imagenet. People are empirically getting closer to that. We'd like our learned \\(f\\) to compete with every representation function in some class, including the case where you were able to train with supervised labels. However, this is not possible -- there are counter examples and impossibility results. So we'll have to make a few more assumptions. 

Before I get to the results, we define the **mean classifier** for two-way classification. Instead of finding classifiers by convex optimization, you could do something similar: Just use the classifiers derived by computing the mean of samples for each class. We will use those. 

The first result is that if the unsupervised loss is low, then average loss on classification tasks is low. This is already useful -- you can just try to minimize unsupervised loss. You can prove this; the key step is just Jensen's inequality applied to contrastive loss. This is already good in many settings, and it's very straightforward. 

It would also be nice to handle the case where the unsupervised loss is not small. The idea is that you can break up this unsupervised loss into two terms: You break it up into two cases in the weakly supervised loss: where the labeled terms are the same and where they're different. Then you look at the concentration in each class. 

Now, we have some progress towards the dream result: We want to compete against the best representation. The assumption we add to make this possible is that we compete against the best concentrated representation (make some subGaussian assumption). It's very hard to test for this though, so it's unclear if it's realistic; but if you visualize, it seems concentrated. Under this assumption you get a result. 

This extends to \\(k\\)-way classification, here you use a similar pair and \\(k-1\\) negative samples. Finally we come up with a new objective (CURL, our version of contrastive learning with blocks -- leverage the latent class ) that is empirically somewhat better and gives better theoretical results. 

#### Experiments 

We look at text embeddings -- train sentence representations and try to solve 2-way and 3-way classification results. Things work well if you even use only around 5 labeled examples per class! This is a verification of the theory. This dataset hadn't been created before, we created to test the theory. 

We did some similar experiments for cifar100, but the gap is somewhat larger. 

#### Conclusions

This is just a first cut theory, I hope I've whetted your appetite. It fits much more closely with learning theory. In practice classes are probably not arbitrary, if you assume some things like that, more empirical and theoretical development, and connections to transfer and meta learning. 


### Gradient Descent Aligns the Layers of Deep Linear Networks (Matus Telgarsky)

#### Introduction

Let me give you a tiny teaser: Why in 2019 are we talking about linear networks, what's the purpose of this? Let's say you download some data from the internet and it's two circles and two classes. If you train a logistic classifier via gradient descent, you get the max-margin classifier. What you should ask yourself is what if you do a deep linear network? This is just an overparameterization of many layers of a linear classifier. It's just a linear function of the input! This is multi-linear polynomial optimization. You actually get the same solution -- gradient descent finds the same solution. 

So the goal today will be to dissect this as much as we can, and I will give you the punchline right now: The punchline is it's not just finding max margin classifier, it's also minimizing Frobenius norm constraint on all the matrices! It balances the Frobenius norms. I find this utterly shocking! If you wanted to induce this sort of property in past framework, you would add a bunch of constraints -- now we just do this! 

Now what aspects of this are true in the nonlinear case (e.g. ReLu case)? You could try doodling this during this talk. 

#### Margin Maximization is Interesting

Now, Ben Recht gave a provactive talk about rethinking generalization -- it really got me out of my comfort zone! The thing I zoomed in on was the explicit regularization comment -- I thought, I've seen this before, it sounds like margin theory. If you do choose to try to resolve some aspects of this paper using margin theory, you're also introducing some new problems. First you have to prove you have a margin maximizing algorithm --- for gradient descent it's still open. We haven't resolved this at all yet. A few more comments about margins --- why is it a useful goal to prove that we have margin maximization? A lot of people try to make the generalization bounds to be small and predictive. We want it to reflect actual empirical generalization. For instance, in a Bartlett paper, the Lipschitz constant tracks generalization. Sanjeev + Nati: It's correlated not predictive (there are lots of other properties that correlate too). If you're not convinced by this, I find the question of finding a normalization to be difficult. Another reasons I found these bounds to be predictive (independent of Lipschitz constant), if you plot the margin distribution, you find some interesting patterns: The margins are closer to zero for random-labeled CIFAR, so the normalized (from the Bartlett et. al. paper) margins are good predictors for the problem. One of the plots that shocked me the most: If you compare MNIST and CIFAR both random, the margin plots are on top of each other. Another question is: are these plots artifcats of the normalization chosen. Why should bound capture anything real if it's a loose upper bound? As I recomputed bounds and debugged proof, you always saw this. These were my loose personal reasons for why I thought it interesting to prove this margin maximization property in the algorithm. One more comment --- on arxiv people are proving settings of convergence for deep nonlinear networks --- they all assume a very wide overparametrization, and this causes the weights to change very little. By contrast, today I will talk about an asymptotic result, where the norm of the predictor goes to infinity. It might seem at odds with these other recent papers. However these results are sort of complementary. 

#### Proving Convex Optimization Guarantees

First let's discuss the regular one-layer linear case that we get max-margin predictor. This is not a textbook theorem, even though it's a convex problem. 

Let's look at the problem: We had toy data -- two classes and they were concentric circles. By symmetry you can say the solution is global optimally and it's zero. What happens as you move the circles apart? Logistic regression is strongly convex. As you keep moving the data apart -- the solution keeps moving towards infinity. In classical learning theory parlance, this is the separable case. Now if you throw a bunch of blue and red points in the middle and move them apart, then you are changing where the global optimum will be, and it shifts the gradient descent path upwards or downwards. For the general version of the problem you have to specify an offset. For logistic regression the solution is a ray with a fixed offset. Here's the two part theorem I can give here (just in convex case). The first part is just the reduction to empirical risk -- it follows a \\(1/T\\) rate, but there is no dependence on distance to the optimum. You can also prove a parameter convergence guarantee. Superficially it does not have a unique optimum. There always exists a unique ray such that gradient descent follows this path. The purpose of this part of the talk is to say that even in the case of linear logistic regression, there were interesting things to uncover in the case where the optimum is not bounded. There's prior and parallel work -- Nati and colleagues analyzed this in the separable case and got a better convergence. This is implicit regularization setting. It's sort of following the optimal path. One reason that constrained optimization is not used as much is gradient descent with appropriate loss is solving some of those constrained things. 

To prove this margin maximization with a slightly slower rate than Nati uses a trick with the Fenchel-Young inequality. 

#### Deep Linear Networks 

The predictor is a multilinear form (simple polynomial) in the weights, but linear in the covariates. Some prior work -- it has saddle properties for the squared loss. Here we are using the logistic loss which has an optimum at infinity. Suriya will talk about another result -- if you assume risk converges and gradients converge you find margin maximizing solution. 

Not only do you maximize margin but you minimize Frobenius norms! On the technical side we want to reduce as many of these assumptions as possible. Let's compare the descent paths of gradient descent for deep linear networks. We know at least one layer is going off to infinity. Spectral norms also go to infinity. There's a theorem by Jason Lee and Simon Du that says they should go to infinity in lockstep (for all the layers). 

Since we are learning a linear predictor, there's no purpose in having a lot of stuff in the weight matrices other than the first rank. We have \\(x \to W_L\cdots W_1 x\\). Each becomes asymptotically rank one, and they're all aligned and become asymptotically equal (we can introduce rotations inbetween matrices, there are infinitely many solutions). There's no norm lost at all. From here you can reverse engineer this minimum Frobenius norm property. To summarize you get \\((u_Lv_L^T)\cdots(u_1v_1^T))\\), and they're all aligned. If you look at first layer, it's the max margin linear predictor if you look at \\(u_1\\). Now if you look at second layer, it learns max margin predictor at that layer. And so on, for every single layer. 

Finally I'll give you real theorem: Assume the data is linearly separable (this is very important). We have an initialization assumption: The initial risk is less than the risk of the zero vector. We're not doing random initialization (you can get to this with a couple random inits, but that's what we require). This means you don't start at a saddle point. This holds for gradient flow or gradient descent with small step size and logistic loss. 
The results: The risk goes to zero, the Frobenius norms go to infinity (see paper for rest). 

#### Questions

What happens in multiclass? The proof which crucially uses rank-1 property breaks down. I don't actually know how to prove this in multi-class case. 


### Optimization Bias in Linear Convolutional Networks (Suriya Gunasekar)

#### Introduction 

The motivating goal for this line of work is learning in deep neural networks -- what is the true inductive bias involved in deep learning? We want to parametrize prediction functions with parameters \\(w\\). Empirical success has arisen from training large networks on large datasets. This is a loss minimization over examples using variations of gradient descent or SGD. 

While neural nets have enjoyed theoretical success, the problem is nonconvex, but we see that SGD is able to find good optima of neural nets. In practice special minimizers have remarkably good performance. This suggests that algorithms are key to learning good models rather than optimization objective itself. 

#### Matrix Completion
Let's look at this setup in a completely different problem -- matrix completion. A common assumption is that groundtruth matrix is somewhat low-rank which fills in the missing entries. A natural objective is to just minimize loss over differences. We can overparametrize and optimize over two matrices \\(U, V\\) and it's easier to restrict to low-rank constraint. 

Today we will instead optimize over full dimensional \\(U, V\\) -- no rank constraints. We will see what happens when we solve this problem using gradient descent. This is a special case of matrix estimation from linear measurements. So is gradient descent on this loss different from gradient descent on the non-overparametrized version, even though the objectives are equivalent? Both algorithms are doing the job of optimizing objective correctly. But what is surprising is that when we do descent on the factorized objective, you get different solutions --- this raises the question which global minimum does gradient descent actually reach? It turns out it's the minimum nuclear norm solution (relaxation of a rank constraint). So we have a bunch of empirical observations on this topic. We can concretize it -- when does gradient descent on factored space converge to nuclear norm solution? We can formalize it more mathematically -- it was proved for special case of commutative measurements. Li, Ma, Zhang 2018 generalized to common random measurement models. 

We wanted to go through this exercise to understand whether this is indeed the phenomenon observed when training neural networks. The number of parameters is often much more than the number of examples in deep nets -- most minimizers won't do well on new datasets, but nevertheless the ones that gradient descent finds do. Different optimization updates might lead to different minimizers. Hyperparameters etc. will also affect what optima you get to. 

This brings up to the point that the algorithms rather than the objective itself are what matters for getting good results. 

#### Optimization algorithms induce inductive bias for learning

This insight could lead to better algorithms and heuristics for faster heuristics etc. 

We'll start out with gradient descent for logistic regression. There's no global minima - gradient descent iterates will diverge in norm, but we care about the direction --- e.g., the normalized iterates. So which classifier/direction does gradient descent converge in -- it's the maximum margin separator, as Matus said earlier. The interesting thing about this result is that it's independent of step size. We have an extension of this work -- using an adaptive step size can still increase the speed at which you reach the final solution. This heuristic has been tried in actual neural networks, that show it could be a valid approach. These results follow one of Matus's much earlier work -- when you do boosting over linearly separable dataset, you converge to maximum margin solution. 

Usually people decrease step size, but we have found with linear networks, increasing step size is actually a better thing to do to reach max-margin. It's something to look at empirically more. 

#### What about neural networks? 

We don't have an answer, but a bit of understanding for linear networks. You can think of it as a multi-layer compositional network, where each layer has a transformation. At this point you take a linear combination to get the final solution. You can think of representing each layer as having different constraints. We're also only considering the case where there's no bottleneck layer, all layers have enough width so there's enough overparametrization. 
Also we look at convolutional networks (linear convolutional networks). 

We can think of different architectures as different parametrizations of the same optimization problem, and the question is what classifier does gradient descent learn depending on the parametrization? 

Linear fully connected networks (also studied in parallel by Matus) -- convergence happens through an alignment process. We started off saying different parametrizations lead to different solutions. Everything converges to the same solution as logistic regression. Matus removed some the assumptions from our work. 

What happens for convolutional networks? Here we observe the solution we get is very different from fully-connected networks. GD promotes sparsity in the frequency components. Gradient descent ends up learning something that looks sparse in the Discrete Fourier domain. We end up maximizing the margin subject to an \\(\ell_1\\) penalty on the frequency components of the predictor. Convolutional layers for larger and larger networks converges to stationary points. 

Why do we get these very different answers? If we have different architectures, how do we figure out what happens? We can think of homogenous linear models as a larger model class. These are all linear functions where the mapping from parameters to the linear predictor is a homogenous function (\\(P(\alpha w) = \alpha^n P(w)\\)). For this function class, gradient does something to minimize the Euclidean norm of the parameters. For nonlinear homogenous functions, we can show a similar result with an extra assumption. 

Now we have bias in the parameter space -- what does this mean in prediction space? By choosing different architectures, we can drive different biases in prediction space. If we have 2-layer network with multiple outputs (matrix factorization setting), this complexity measure is essentially nuclear norm. 

Another followup work whcih Nati is super excited about is if you're learning a map from scalar to scalar with infinite width network, minimizing \\(\ell_2\\) norm corresponds to total variation of function class. Takeaway point is that the geometry of the parameters influences inductive bias, and architecture interacts nicely with the form of the bias. 

Some related questions: does gradient descent globally optimize the objective? Does the direction of iterates converge? In our results we assume the loss goes to zero and the iterates converge, etc. 

So our results: 

* squared loss linear regression implies minimum Euclidean norm 

* squared loss matrix completion minimizes Euclidean norm of factors 

* logistic loss for linear classification minimizes Euclidean norm with margin condition 

#### Optimization bias in non-Euclidean geometry

We can think of finding a step which has the most correlation with negative gradient but which at the same time is close to the current iterate (this is where Euclidean geometry comes in). We could think of steepest descent in general norms, non-Euclidean norm perhaps. If this is \\(\ell_1\\), this is coordinate descent. For mirror descent, we get Bregman divergence. This defines the geometry in which I'm making the updates. We want to see if there's a relation between optimization bias and optimization geometry. 

What's the optimization bias of mirror descent? Here we get a different flavor of results. For square loss we get that it gets the argmin of the potential function associated with mirror descent. With appropriate init we can show it converge to this. For exponentiated GD you can show it goes to max-entropy solution. 

#### Conclusions

* Optimization bias has interesting interactions with optimization geometry 

* Optimization geometry relates to the nature of optimization bias 

* Squared loss is very different compared to exponential bias. 


### Towards a Foundation of Deep Learning: SGD, Overparametrization, and Generalization (Jason Lee)

### Representational Power of narrow ResNet and of Graph Neural Networks (Stefanie Jegelka)

### Is There a Tractable (and Interesting) Theory of Nonconvex Optimization? (Santosh Vempala)

### Panel (Sanjeev Arora, Andrea Montanari, Katya Scheinberg, Nati Srebro, and Antonio Torralba)


## January 30

### Learning Restricted Boltzmann Machines (Ankur Moitra)

### SGD with AdaGrad Adaptive Learning Rate: Strong Convergence without Step-size Tuning (Rachel Ward)

### Understanding adaptive methods for non-convex optimization (Satyen Kale)

### Adversarial Examples from Computational Constraints (Sebastien Bubeck)

### A Critical View of Global and Local Optimality in Deep Networks (Suvrit Sra)







