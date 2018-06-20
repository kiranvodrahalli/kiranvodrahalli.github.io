---
layout: page
title: Theoretical Frameworks and Results in Reinforcement Learning  
---

* TOC
{:toc}

<!-- should i make this onto a separate page? how do i organize these survey notes for my own reasons? don't really want them to be blog posts... hm --> 

<!-- another issue is stuff like MIT sublinear, that shouldn't be a blog post. it should also be on the notes page, but instead of pdf, link to the page. that is easier to figure out. I suppose the same can be true for this RL survey/ overview/wiki/cheatsheet of RL settings and paper notes --> 


Here we conduct a survey of RL research which attempts to beat the \\(\Omega(S^2*A)\\) lower bound by
assuming things about state and action spaces. I am filling this in as I go along my research survey. 

Note: In the offline version on my computer, links are links to files on my computer, not the internet. 

If you are accessing this page on the web, the links link to the papers where they can be found online. 

___
 
### \\(E^3\\) Algorithm and variants

\\(E^3\\) Algorithm 
* |**Known State**: a state visited often enough so that dynamics and rewards are accurately modeled at this state.|

* |**Exploratory MDP**: agent plans to escape in a fictitious MDP where the escape state has maximal reward.| 

* |Algorithm: When the agent is not at a known state, wander randomly to obtain new information. At a known state: explore or exploit (can decide efficiently given dynamics and rewards and a planning oracle). The decision to explore is made by assessing potential reward if the agent escapes known states to get maximal reward elsewhere: computed by planning to escape in a fictitious MDP which provides maximal reward for entering an unknown state.| 

* |Proof: either agent exploits for near optimal reward, or explores quickly, and increases set of known states -- since # states is bounded, alg eventually exploits and gets near optimal reward.|  

Papers 

* [paper title](link)

### R-Max

R-Max Algorithm 

* description 

Papers

* [paper](link)

### Metric Assumption

Papers 

#### [Exploration in Metric State Spaces (Kakade, Kearns, Langford 2003)](http://www.aaai.org/Papers/ICML/2003/ICML03-042.pdf)

##### High level takeaways

* This paper uses the \\(E^3\\) algorithm and assumes a *blackbox for approximate planning*. 

* Setting: \\(T\\)-step planning, average finite reward case. \\(d((s, a), (s', a'))\\) is the metric. Requirements for \\(d\\): must have \\(d((s, a), (s, a)) = 0\\) and symmetry, but don't require the triangle-inequality. 

* Different from E3: time/sample? complexity depends on the *covering number* of the state space, rather than the number of states. E.g., notion of resolution for modeling under the metric. 

* global exploration: trying to learn near optimal \\(T\\)-step policy in amount of time with no dependence on state space size: only on complexity of *state-space representation*. 

* gradient methods for approximate planning (compare/contrast w/recent Berkeley stuff?): old stuff (pre-2003) studied convergence to polices which small amounts of random exploration cannot improve. Notably, this does NOT handle the case of planning a long sequence of steps which might not be encountered by random approaches. Is this a drawback to Universal Planning Networks? 

* common technique: separate planning problem from exploration problem by assuming blackbox for approximate planning (e.g., DP or something else)

##### Assumptions

* assume there is a metric on state-action pairs (Metric MDP)

* "nearby" state-actions can be useful in predicting state-action dynamics

|**Local Modeling Assumption** (formal): We have an algorithmic model \\(A\\) for MDP. For any \\((s, a)\\), if we take \\(m\\) transitions \\((s', a') --> s''\\) and rewards according to the MDP, and also all these states we see are \\(\alpha\\)-close under the metric \\(d\\), then our model \\(A\\) of the MDP outputs a state according to transition probability \\(\hat{P}(\hat{s} \|s, a)\\) and reward \\(\hat{R}\\), such that \\(\hat{P}\\) is within TV-distance \\(\leq \alpha\\) of the true distribution and the reward vector over states is within \\(\ell_{\infty}\\)-norm \\(\leq \alpha\\) of the true reward output for each state (all given curren state, action).|

* Why do we need the local modeling assumption? (note that this is key to get the metric result): We need it so that we can use the discretization of state-action space embued by the cover and still succeed in simulating the MDP. 

* We do not require that a destination state is in the neighborhood of \\((s, a)\\): only that nearby state-actions permit generalization in next-state distributions

|**Approximate planning assumption**: Given a generative model for an MDP and a state \\(s\\), we can return a policy \\(\pi\\) whose average reward is within \\(\beta\\) of the average \\(T\\)-step optimal reward.|

* Notably, the approximate planning assumption abstracts away computational complexity of this procedure (as in factored \\(E^3\\)). 

##### Method 

* given sufficient "nearby" experience, implicit non-parametric model of dynamics in neighborhood of state-action space can be pieced together for planning on a subset of the global space

* introduce notion of metric over state-action space, assume this metric permits construction or inference of "local" models, and then assume that such models permit planning. 

* in order to be effective, the covering number needs to be much smaller under the metric chosen. 

|**Known State-Action**: satisfies local modeling assumption --- any pair (s, a) s.t. alg has obtained at least m \\(\alpha\\)-close experiences (transition + reward). |

|**Known MDP**: model for part of global MDP we can approximate. It operates with input \\((s, a)\\), and flag bit "exploit". If it is not a known state-action, output fail and stop. Otherwise, give the input and its \\(m\\) prior experiences in the \\(\epsilon\\)-neighborhood of the input to the generative model of the MDP, which outputs a state and reward. Recall that the model is close in TV distance for the conditional (on \\((s, a)\\)) transition distribution and close in \\(\ell_{\infty}\\) for the reward function over states. Let the outputs be \\(\hat{s}, \hat{r}\\). If "exploit" is true, then if \\(\hat{s}\\) is not part of any known state-action pair, output special "absorbing" state \\(z\\) and reward \\(0\\) and stop. Otherwise, if \\(\hat{s}\\) is part of a known state-action, then output the correct reward and the true state \\(\hat{s}\\) and stop. If "exploit" is false, then if \\(\hat{s}\\) is part of a known state-action, output \\(\hat{s}\\) and reward \\(0\\). If it is not part of a known state-action, then output special absorbing state \\(z\\) and reward \\(1\\). We can think of \\(z\\) as the "exploration state", which represents an escape from the MDP, similarly to the vanilla \\(E^3\\) algorithm. |

|Metric-\\(E^3\\) Algorithm: decides if state-actions are known, so MDP model only needs a list of prior experiences. The algorithm is as follows: Use random moves until encountering a state with at least one known action: \\((s, a)\\). Use the planning oracle to plan on the approximate exploit-MDP ("exploit" \\(= 1\\)) and the approximate explore-MDP ("exploit" \\(= 0\\)). Call the resulting policies \\(\pi_1, \pi_2\\). If the value of the explore-policy over the approximate explore-MDP is \\(> \epsilon\\), then execute the explore-policy for the next \\(T\\) steps (we are dealing with \\(T\\)-step policies) and restart. Otherwise, finish and output the exploit-policy.|


##### Methods that don't work

* state aggregation: Markov dynamics must be preserved, often are not (Markovian assumption invalidated in the aggregate-state-space) (seems similar to what we assume w/FSA)

##### Examples

* sufficent sensory info and advance knowledge of effects of actions so that local modeling works even with \\(m = 1\\) (e.g., just one step suffices to get \\(\alpha\\)-close in the metric)

* line-of-sight metric for particular maze: \\(d((s, a), (s', a')) = 0 \\) if in line of sight and \\(\infty\\) otherwise. (doesn't need to satisfy triangle!) --- this admits a small covering of the space. Can add the Euclidean metric to it so that modeling error can grow with distance. 

* can also use metrics which can be defined with actions

* state-space aggregation doesn't work: corners where three augmented states exist are unstable near the corner. 

##### Theoretical Results 

* \\(\pi^*\\) is optimal policy in the MDP

* \\(T\\) is time horizon

* Covering number \\(N(\alpha)\\)

* \\(m, \alpha\\) are defined in local modeling assumption: no. of samples for local modeling, and precision (radius) for approximate MDP

* \\(\beta\\) is approximate planning precision

* theorems do not require assumption that MDP mixes

| **Sample Complexity of Metric-**\\(E^3\\): After \\(\frac{T\cdot m \cdot N(\alpha)}{\epsilon - \alpha(T + 1)}\log(1/\delta) + m\cdot N(\alpha)\\) actions, Metric-\\(E^3\\) halts and outputs policy such that the expected average \\(T\\)-step value is at least \\(OPT - \epsilon - 2\beta - 2\alpha(T + 1)\\)|

* The proof is as follows: As per **bounded exploration lemma**, we encounter a known state after at most \\(m\cdot N(\alpha)\\) exploration attempts, each of which happens when the expected value over the approximate explore-MDP is \\(> \epsilon\\). By the **simulation lemma**, the true value is at least \\(\epsilon - \alpha(T + 1)\\). Therefore, since you get reward \\(1\\) when you successfully explore, since the expected reward is at least \\(\epsilon - \alpha(T + 1)\\), so is the probability of successful exploration. Then, there are a maximum of \\(\frac{1}{\epsilon - \alpha(T + 1)}\\) actions each time we have a successful exploration, and thus at most a total of \\(\frac{m\cdot N(\alpha)}{\epsilon - \alpha(T + 1)}\\) actions per episode. Then there are \\(T\\) episodes, so multiply by that to get an upper bound on the total number of possible actions taken. Apply Chernoff-Hoeffding to the coin flip scenario to get the first term in the bound. This is valid because, although the samples may not be independent, any upper bound that holds for independent samples also holds for samples obtained in an online manner by the agent (see Strehl's thesis, section 1.5). The policy output satisfies the value bound by examining the halting condition and looking at the planning and simulation errors, using the **Explore-or-Exploit** lemma. 

| **Simulation Lemma**: For an \\(\alpha\\)-approximation of the MDP, then the expected average \\(T\\)-step value of the policy in the approximated MDP is within \\(\alpha(T + 1)\\) of the expected average \\(T\\)-step value of the policy in the true MDP for any policy. This lets us use the approximate MDPs while paying some additive error. Proved with triangle inequality and linearity of expectation, followed by induction. |

| **Explore or Exploit Lemma**: The optimal exploit-policy policy performs at least OPT \\(- \epsilon\\) well, where OPT is the value of the optimal policy, or the optimal explore-policy has probability of at least \\(\epsilon\\) of leaving the known states. |

| **Bounded Exploration Time Lemma**: Metric-\\(E^3\\) encounters at most \\(m\cdot N(\alpha)\\) unknown state-actions. Proof idea: For \\(m = 1\\), construct an invariant for all \\(t\\): \\(C\\) is a minimal set of disjoint balls with radius \\(\alpha\\) over state-actions seen up to time \\(t\\). Thus for all \\(t\\), \\(\|C\| < N(\alpha)\\) by definition of covering number. For general \\(m\\), construct \\(m\\) analagous sets \\(C_1, \cdots, C_m\\) instead, and consider \\(\sum_{i = 1}^m \|C_i\| < m\cdot N(\alpha)\\) |

### Factored MDPs

### PAC RL

### Regret Bounds 

### Approximate MDPs

### Regret Bounds

### Contextual Decision Processes

---


