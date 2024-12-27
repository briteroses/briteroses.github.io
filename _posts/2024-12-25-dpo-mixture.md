---
layout: post
title: Some DPO napkin math
date: 2024-12-26
description: TL;DR if we formulate prompt-conditioned output distributions from an LLM as a mixture of an aligned component + unaligned component, then a DPO tune "tilts" the output towards the aligned component, diminishing---but not expunging!---the unaligned component.
tags: dpo preference-tuning alignment math
categories: research
related_posts: false
---

$$
\DeclareMathOperator\supp{supp}
\DeclareMathOperator\TPM{TPM}
\DeclareMathOperator\unif{unif}
\DeclareMathOperator\card{card}
$$

As a mathy Christmas treat, I wanted to share some napkin math I did earlier this year while studying the preference tuning literature, especially the (now seminal) [DPO paper](https://arxiv.org/abs/2305.18290). For starters, DPO and its cousin techniques (like [KTO](https://arxiv.org/abs/2402.01306) and others) hinge on optimizing a KL-constrained RL objective, which I reproduce below:

$$
\mathcal{L}(\theta) = E_{y \in p_\theta(y|x)}[r(y;x)] - \beta\,\mathbb{D}_{KL}(p_\theta(y|x) || p_{\theta_{ref}}(y|x)).
$$

Here, $$ \theta $$ is our current model parameters optimized during the preference tune; $$ \theta_{ref} $$ is our (frozen) reference model parameters (usually, the SFTed model or base model at the initialization of the preference tune); $$ x $$ is a (given) prompt and $$ y $$ is its sampled response from the model; $$ r $$ is the reward function; and $$ \beta $$ is the hyperparameter controlling the weighting of the KL regularization term.

The solution to the KL-constrained RL loss has a well-known form that upweights or downweights a sampled response relative to its reward. However, at least to me, this solution isn't very intuitive for understanding how more overarching model behaviors are altered under a preference tune. As a motivating example, DPO tuning is often used as a line of defense for model safety: say, promoting refusals for a wide range of explicitly [harmful intents](https://arxiv.org/abs/2402.04249), or [reducing toxicity](https://arxiv.org/abs/2401.01967) for all kinds of mundane prompts. If we care about obtaining better refusals or more nontoxic behaviors "across the board," then we want to understand how entire probability masses corresponding to general output specifications (for example, all possible responses with toxic content) are shifted under DPO. These desired behavioral shifts can be further confuzzled in practice, especially since the reward is not known a priori and issues such as data-policy mismatch or likelihood displacement can produce pretty unintuitive results from a DPO tune. ([This blog post](https://tianjianl.github.io/blog/2024/dpo/) provides a great overview of DPO failure modes, if you want to read further.)

To capture a more macroscopic view of DPO, I find it useful to formulate the prompt-conditioned output distribution of our reference model $$ p_{\theta_{\text{ref}}}(y \vert x) $$ as a mixture of two component distributions, one "aligned" and the other "unaligned." We denote $$ \mathcal{A}_{x}(y \vert x) $$ for the aligned distribution and $$ \mathcal{U}_{x}(y \vert x) $$, with mixture weight $$ \alpha_{x} $$:

$$
p_{\theta_{ref}}(y | x) = \alpha_x p_{\mathcal{A}_x}(y | x) + (1-\alpha_x) p_{\mathcal{U}_x}(y | x).
$$

For a (somewhat handwavy, sorry!) illustration based on our toxicity-reduction scenario, we might assign all fully nontoxic responses to the aligned component, all extremely toxic responses to the unaligned component, and a range of ambiguous responses to both components, with intra-component weightings depending on the level of toxicity. In other words, we can potentially describe more general specifications of model behavior under the umbrella of these mixture components. If we establish certain properties of the post-DPO output distribution $$ p_\theta(y \vert x) $$ through the lens of $$ \mathcal{A} $$ and $$ \mathcal{U} $$, we gain insight into how DPO may induce---or fail to induce---a broader behavior such as toxicity reduction or safety refusals.

First, we show that the post-DPO output distribution has a specific form---no longer a mixture per-se, but a per-sample tilting of the original mixture.

**Lemma 1.** Let $$ \theta $$ be the model parameters that maximize the previous KL-constrained RL objective. With no additional assumptions on $$ \mathcal{A} $$ and $$ \mathcal{U} $$, the output distribution of $$ \theta $$ can be written as:

$$
p_\theta(y|x)=\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)+\alpha_\mathcal{U}(y;x)p_\mathcal{U}(y|x)
$$

$$
\alpha_\mathcal{A}(y;x)=\frac{1}{Z(x)}\alpha(x)e^{r(y;x)/\beta}, \;\; \alpha_\mathcal{U}(y;x)=\frac{1}{Z(x)}(1-\alpha(x))e^{r(y;x)/\beta}
$$

where $$ Z(x)=\sum_{y \in \supp p_\theta(\cdot \vert x)} p_{ref}(y \vert x)e^{r(y;x)/\beta} $$ is the partition function of $$ p_\theta(y \vert x) $$.

We can then prove a stronger result about shifts in the probability masses corresponding to each mixture component:

**Theorem 2.** Write the "total probability mass" of $$ \mathcal{A} $$ in $$ p_\theta(\cdot \vert x) $$ as

$$
\TPM(\mathcal{A};x) = \sum_{y \in \supp p_\theta(\cdot \vert x)} \alpha_\mathcal{A}(y;x)p_\mathcal{A}(y \vert x)
$$

(and define this respectively for $$ \mathcal{U} $$). If the reward function $$ r(y;x) $$ satisfies the following additional assumptions:

* $$ r(y;x) $$ and $$ p_\mathcal{A}(y \vert x) $$ are positively correlated
* $$ r(y;x) $$ and $$ p_\mathcal{U}(y \vert x) $$ are negatively correlated  
* $$ r $$ is finite

then the total probability masses of $$ \mathcal{A} $$ and $$ \mathcal{U} $$ in $$ p_\theta(\cdot \vert x) $$ satisfy:

$$
1 > \TPM(\mathcal{A};x) \geq \alpha(x), \;\; 0 < \TPM(\mathcal{U};x) \leq (1-\alpha(x)).
$$

The total probability masses of $$ \mathcal{A} $$ and $$ \mathcal{U} $$ can be viewed as a measure of the aligned (resp. unaligned) distribution's influence on the tuned language model's output distribution.

## Proofs

First, the closed-form solution for the KL-constrained RL objective is well-known, but we reproduce it here. First, rewrite the objective as:

$$
\begin{align*}
\mathcal{L}(\theta) &= \sum_y p_\theta(y|x)r(y;x) - \beta p_\theta(y|x) \log\left(\frac{p_\theta(y|x)}{p_{ref}(y|x)}\right) \\
&= \beta \sum_y p_\theta(y|x) \Big[ \log e^{r(y;x)/\beta} - \log\left(\frac{p_\theta(y|x)}{p_{ref}(y|x)}\right) \Big] \\
&= -\beta \sum_y p_\theta(y|x) \log\left(\frac{p_\theta(y|x)}{\frac{1}{Z(x)}p_{ref}(y|x)e^{r(y;x)/\beta}}\right) + \beta\log{Z(x)}
\end{align*}
$$

where $$ Z(x) = \sum_y p_{ref}(y \vert x)e^{r(y;x)/\beta} $$ is the partition function. Notice that $$ \frac{1}{Z(x)}p_{ref}(y \vert x)e^{r(y;x)/\beta} $$ is a valid probability distribution, so by Gibbs' inequality, the objective $$ \mathcal{L}(\theta) $$ is maximized when:

$$
p_\theta(y|x) = \frac{1}{Z(x)}p_{ref}(y|x)e^{r(y;x)/\beta}.
$$

Substituting the mixture formulation for $$ p_{ref}(y \vert x) $$ and performing a little more algebra gives us:

$$
p_\theta(y|x)=\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)+\alpha_\mathcal{U}(y;x)p_\mathcal{U}(y|x)
$$

$$
\alpha_\mathcal{A}(y;x)=\frac{1}{Z(x)}\alpha(x)e^{r(y;x)/\beta}, \;\; \alpha_\mathcal{U}(y;x)=\frac{1}{Z(x)}(1-\alpha(x))e^{r(y;x)/\beta}
$$

as Lemma 1 states.

For Theorem 2, we assumed $$ r(y;x) $$ was positively correlated with $$ p_\mathcal{A}(y \vert x) $$ and negatively correlated with $$ p_\mathcal{U}(y \vert x) $$. Since $$ Z(x) $$ and $$ \alpha(x) $$ are constants with respect to $$ y $$, $$ \alpha_\mathcal{A}(y;x) $$ and $$ \alpha_\mathcal{U}(y;x) $$ are clearly monotonic functions of $$ r(y;x) $$, and as a result, the correlation properties are preserved: $$ \alpha_\mathcal{A}(y;x) $$ is positively correlated with $$ p_\mathcal{A}(y \vert x) $$, and $$ \alpha_\mathcal{U}(y;x) $$ is negatively correlated with $$ p_\mathcal{U}(y \vert x) $$.

Notice that finiteness of $$ r $$ implies $$ e^{r(y;x)/\beta} > 0 $$ everywhere, so the output distribution of $$ \theta $$ must have nonzero probability for any output in $$ \supp p_{ref} $$. Therefore, $$ \supp p_\theta(\cdot | x) = \supp p_{ref}(\cdot \vert x) $$. Consider the expectations of $$ \alpha_\mathcal{A}(y;x) $$ and $$ \alpha_\mathcal{U}(y;x) $$ over a uniform distribution over (WLOG) $$ \supp p_{\theta}(\cdot \vert x) $$, and use the correlation properties:

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

(For the sake of brevity, we mostly omit writing the distribution over which the expectation is taken; for the rest of the proof, $$ E_y[\cdot] $$ is shorthand for $$ E_{y \in \unif(\supp p_{\theta}(\cdot \vert x))}[\cdot] $$.) The $$ E_y[e^{r(y;x)/\beta}] $$ and partition function terms vanish upon dividing the inequalities:

$$
\frac{E_y[\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)]}{E_y[\alpha_\mathcal{U}(y;x)p_\mathcal{U}(y|x)]} > \frac{E_y[\alpha(x)p_\mathcal{A}(y|x)]}{E_y[(1-\alpha(x))p_\mathcal{U}(y|x)]}
$$

Reciprocating the inequality, adding 1 to both sides, and reciprocating again yields:

$$
\frac{E_y[\alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x)]}{E_y[p_\theta(y|x)]} > \frac{E_y[\alpha(x)p_\mathcal{A}(y|x)]}{E_y[p_{ref}(y|x)]}
$$

Finally, we multiply all expectations by $$ \card(\supp p_\theta(\cdot \vert x)) $$. Using the property that

$$
\card(\supp p) \cdot E_{y \in \unif(\supp p)}[p(x)] = 1
$$

for any discrete probability distribution $$ p $$, along with our earlier observation that $$ \supp p_\theta(\cdot \vert x) = \supp p_{ref}(\cdot \vert x) $$, we see that the expectations of  $$ p_\theta(y \vert x) $$ and $$ p_{ref}(y \vert x) $$ vanish. On the other hand, multiplying the expectation of $$ \alpha_\mathcal{A}(y;x)p_\mathcal{A}(y|x) $$ by $$ \card(\supp p_\theta(\cdot \vert x)) $$ gives the total probability mass. We recover:

$$
\TPM(\mathcal{A};x) > \alpha(x) \;\;\;\;\;\;\;\; \text{and likewise, } \TPM(\mathcal{U};x) < (1-\alpha(x)).
$$

## Takeaways

Intuitively, we can interpret Theorem 2 as the following idea: if we select aligned and unaligned distributions that indeed correspond to a level of "alignment" or utility measured by our rewards, then the preference-tuned model draws outputs more heavily from the aligned distribution, while the influence of the unaligned distribution is diminished but not expunged. Crucially, the preference-tuned model still has nonzero probability of producing any harmful or otherwise undesirable output that was possible in the base model.

Looking at our math, we do need to be careful about our construction of aligned and unaligned distributions: the reward correlation conditions in Theorem 2 are fairly reasonable, but can certainly be violated in practice. Fortunately, the reward finiteness condition should always hold; in DPO, finiteness is clear from the form of the implicit reward used.

There's a link here between DPO-based safety finetunes and adversarial attacks (jailbreaks) which may be fairly obvious, but that I still want to highlight. Because we can never expunge harmful generations, we can re-amplify the probability of obtaining harmful trajectories from our post-DPO model with various attack methods, from high-temperature decoding to discrete-optimized suffixes to randomized prompt fuzzing with a high attack budget. Additionally, our analysis only extends to prompt-conditioned output distributions, and we can't say anything definitive about model behavior under prompts out-of-distribution of those in our DPO data. Under the [mismatched generalization]() hypothesis for jailbreaks, safety finetunes lose effectiveness on out-of-distribution prompts; this proposition illuminates what may be happening in the blind spots of our analysis.