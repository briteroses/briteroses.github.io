---
layout: post
title: DPO Mixture Analysis
date: 2024-12-25
description: Mathematical analysis of DPO mixture distributions
tags: math dpo machine-learning
categories: research
related_posts: false
---

In progress!

First, we show that the output distribution has a specific mixture form:

**Lemma 1.** Assume the preference-tuning objective function used is [TODO]. With no additional assumptions on $$ \mathcal{A}$$  and $$ \mathcal{U} $$, the output distribution of $$ \theta $$ can be written as:

$$
p_\theta(y|x)=\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)+\alpha_\mathcal{U}(y;x)p_\mathcal{U}(y|x)
$$

$$
\alpha_\mathcal{A}(y;x)=\frac{1}{Z(x)}\alpha(x)e^{r(y;x)/\beta}, \;\; \alpha_\mathcal{U}(y;x)=\frac{1}{Z(x)}(1-\alpha(x))e^{r(y;x)/\beta}
$$

where `$$Z(x)=\sum_{y \in \supp p_\theta(\cdot | x)} p_{ref}(y|x)e^{r(y;x)/\beta}$$` is the partition function of `$$p_\theta(y;x)$$`.

We can then prove a stronger result about how the mixture weights shift:

**Theorem 2.** Write the "total probability mass" of `$$\mathcal{A}$$` in `$$p_\theta(\cdot | x)$$` as `$$\TPM(\mathcal{A};x) = \sum_{y \in \supp p_\theta(\cdot | x)} \alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)$$` (and define this respectively for `$$\mathcal{U}$$`). If the reward function `$$r(y;x)$$` satisfies the following additional assumptions:

* `$$r(y;x)$$` and `$$p_\mathcal{A}(y|x)$$` are positively correlated
* `$$r(y;x)$$` and `$$p_\mathcal{U}(y|x)$$` are negatively correlated  
* `$$r$$` is finite

then the total probability masses of `$$\mathcal{A}$$` and `$$\mathcal{U}$$` in `$$p_\theta(\cdot | x)$$` satisfy:

$$
1 > \TPM(\mathcal{A};x) \geq \alpha(x), \;\; 0 < \TPM(\mathcal{U};x) \leq (1-\alpha(x)).
$$

The total probability masses of `$$\mathcal{A}$$` and `$$\mathcal{U}$$` can be viewed as a measure of the aligned (resp. unaligned) distribution's influence on the tuned language model's output distribution. Intuitively, we can interpret Theorem 2 as the following idea: if we select aligned and unaligned distributions that indeed correspond to a level of "alignment" or utility measured by our rewards, then the preference-tuned model draws outputs more heavily from the aligned distribution, while the influence of the unaligned distribution is diminished but not expunged. Crucially, the preference-tuned model still has nonzero probability of producing any harmful or otherwise undesirable output that was possible in the base model.

## Proofs

Our proofs of Lemma 1 and Theorem 2 will illuminate further details for how the probability of eliciting a given output from our model is amplified or diminished as a function of the reward.

First, the closed-form solution for the KL-constrained RL objective is well-known, but we reproduce it here. First, rewrite the objective as:

$$
\begin{align*}
\mathcal{L}(\theta) &= \sum_y p_\theta(y|x)r(y;x) - \beta p_\theta(y|x) \log\left(\frac{p_\theta(y|x)}{p_{ref}(y|x)}\right) \\
&= \beta \sum_y p_\theta(y|x) \Big[ \log e^{r(y;x)/\beta} - \log\left(\frac{p_\theta(y|x)}{p_{ref}(y|x)}\right) \Big] \\
&= -\beta \sum_y p_\theta(y|x) \log\left(\frac{p_\theta(y|x)}{\frac{1}{Z(x)}p_{ref}(y|x)e^{r(y;x)/\beta}}\right) + \beta\log{Z(x)}
\end{align*}
$$

where `$$Z(x) = \sum_y p_{ref}(y|x)e^{r(y;x)/\beta}$$` is the partition function. Notice that `$$\frac{1}{Z(x)}p_{ref}(y|x)e^{r(y;x)/\beta}$$` is a valid probability distribution, so by Gibbs' inequality, the objective `$$\mathcal{L}(\theta)$$` is maximized when:

$$
p_\theta(y|x) = \frac{1}{Z(x)}p_{ref}(y|x)e^{r(y;x)/\beta}.
$$

Substituting the mixture formulation for `$$p_{ref}(y|x)$$` and performing a little more algebra gives us:

$$
p_\theta(y|x)=\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)+\alpha_\mathcal{U}(y;x)p_\mathcal{U}(y|x)
$$

$$
\alpha_\mathcal{A}(y;x)=\frac{1}{Z(x)}\alpha(x)e^{r(y;x)/\beta}, \;\; \alpha_\mathcal{U}(y;x)=\frac{1}{Z(x)}(1-\alpha(x))e^{r(y;x)/\beta}
$$

as Lemma 1 states.

For Theorem 2, we assumed `$$r(y;x)$$` was positively correlated with `$$p_\mathcal{A}(y|x)$$` and negatively correlated with `$$p_\mathcal{U}(y|x)$$`. Since `$$Z(x)$$` and `$$\alpha(x)$$` are constants with respect to `$$y$$`, `$$\alpha_\mathcal{A}(y;x)$$` and `$$\alpha_\mathcal{U}(y;x)$$` are clearly monotonic functions of `$$r(y;x)$$`, and as a result, the correlation properties are preserved: `$$\alpha_\mathcal{A}(y;x)$$` is positively correlated with `$$p_\mathcal{A}(y|x)$$`, and `$$\alpha_\mathcal{U}(y;x)$$` is negatively correlated with `$$p_\mathcal{U}(y|x)$$`.

Notice that finiteness of `$$r$$` implies `$$e^{r(y;x)/\beta} > 0$$` everywhere, so the output distribution of `$$\theta$$` must have nonzero probability for any output in `$$\supp p_{ref}$$`. Therefore, `$$\supp p_\theta(\cdot | x) = \supp p_{ref}(\cdot | x)$$`. Consider the expectations of `$$\alpha_\mathcal{A}(y;x)$$` and `$$\alpha_\mathcal{U}(y;x)$$` over a uniform distribution over (WLOG) `$$\supp p_{\theta}(\cdot | x)$$`, and use the correlation properties:

$$
\begin{align*}
E_{y \in \unif(\supp p_{\theta}(\cdot | x))}[\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)] &> E_y[\alpha_\mathcal{A}(y;x)]E_y[p_\mathcal{A}(y|x)] \\
&= \frac{1}{Z(x)}\alpha(x)E_y[e^{r(y;x)/\beta}]E_y[p_\mathcal{A}(y|x)].
\end{align*}
$$

$$
\begin{align*}
E_y[\alpha_\mathcal{U}(y;x)p_\mathcal{U}(y|x)] &< E_y[\alpha_\mathcal{U}(y;x)]E_y[p_\mathcal{U}(y|x)] \\
&= \frac{1}{Z(x)}(1-\alpha(x))E_y[e^{r(y;x)/\beta}]E_y[p_\mathcal{U}(y|x)].
\end{align*}
$$

(For the sake of brevity, we mostly omit writing the distribution over which the expectation is taken; for the rest of the proof, `$$E_y[\cdot]$$` is shorthand for `$$E_{y \in \unif(\supp p_{\theta}(\cdot | x))}[\cdot]$$`.) The `$$E_y[e^{r(y;x)/\beta}]$$` and partition function terms vanish upon dividing the inequalities:

$$
\frac{E_y[\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)]}{E_y[\alpha_\mathcal{U}(y;x)p_\mathcal{U}(y|x)]} > \frac{E_y[\alpha(x)p_\mathcal{A}(y|x)]}{E_y[(1-\alpha(x))p_\mathcal{U}(y|x)]}
$$

Reciprocating the inequality, adding 1 to both sides, and reciprocating again yields:

$$
\frac{E_y[\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)]}{E_y[p_\theta(y|x)]} > \frac{E_y[\alpha(x)p_\mathcal{A}(y|x)]}{E_y[p_{ref}(y|x)]}
$$

Finally, we multiply all expectations by `$$\card(\supp p_\theta(\cdot | x))$$`. Using the property that

$$
\card(\supp p) \cdot E_{y \in \unif(\supp p)}[p(x)] = 1
$$

for any discrete probability distribution `$$p$$`, along with our earlier observation that `$$\supp p_\theta(\cdot | x) = \supp p_{ref}(\cdot | x)$$`, we see that the expectations of `$$p_\theta(y|x)$$` and `$$p_{ref}(y|x)$$` vanish. On the other hand, multiplying the expectation of `$$\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)$$` by `$$\card(\supp p_\theta(\cdot | x))$$` gives the total probability mass. We recover:

$$
\TPM(\mathcal{A};x) > \alpha(x) \;\;\;\;\;\;\;\; \text{and likewise, } \TPM(\mathcal{U};x) < (1-\alpha(x)).
$$