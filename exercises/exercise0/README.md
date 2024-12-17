## Task Summary
We are given a set of data points $X_k = \{ x_1, x_2, ..., x_n \}$, where all points $x_{j}$ are sampled i.i.d. from one of 3 distributions: 1) Normal Distribution with $\sigma = \sqrt2$, 2) Laplace distribution with $b = 1$, or 3) Student's t-distribution with v = 4. 

We are given m datasets X. Any dataset X has a 35% chance of being drawn from the normal distribution, a 25% chance of being drawn from the Laplace distribution and a 40% chance of being from the Student's t-distribution. We let $H_i$ denote the probability of X being drawn from $p_i$. We want to calculate $p(H_i|X)$ for $i = 1,2,3$. 

## Solution Approach
Using Bayes' Theorem we have  

$$p(H_i|X) = \frac{p(X|H_i)p(H_i)}{p(X)} = \frac{p(X|H_i)p(H_i)}{\sum_i p(X|H_i)p(H_i)}$$  

$$p(X|H_i) = \prod_{x \in X}p(x|H_i)$$ as samples are i.i.d.

For numerical reasons we use log probs, so we need to adjust the formula  

$$\log p(H_i|X) = log p(X|H_i) + log p(H_i) - log(\sum p(X|H_i)p(H_i))$$

$$\log p(X|H_i) = \sum_{x \in X} \log p(x|H_i)$$

We calculate $log(\sum p(X|H_i)p(H_i))$ using the logsumExp trick. 
