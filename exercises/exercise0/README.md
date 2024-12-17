# Task Description
## Task

The task is to implement Bayesian inference in a simple setting. In particular, the setting is as follows. You are given a set of data points $X={x_1,…,x_n}$ which are sampled i.i.d. from one of the following three distributions: 

* Normal distribution:
  
  &ensp;&ensp; $$p_1(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}(\frac{x}{\sigma})^2}, &ensp;&ensp;\sigma = \sqrt{2}$$
* Laplace distribution:
 
  &ensp;&ensp; $$p_2(x) = \frac{1}{2b}e^{-\frac{1}{b}\|x|}, &ensp;&ensp; b = 1$$
* Student's t-distribution:
   
  &ensp;&ensp; $$p_3(x) = \frac{\Gamma ((v+1)/2)}{\sqrt{v \pi} \Gamma (v/2)}(1+\frac{x^2}{v})^{-(v+1)/2}, &ensp;&ensp; v = 4$$

In 35% of the cases, the dataset is drawn from the normal distribution, in 25% of the cases from the Laplace distribution and in 40% of the cases from the Student's t-distribution. Let $H_i$ denote the event that the data was sampled from $p_i$ for $i=1,2,3$. Your task is to implement a Bayes-optimal predictor that, given the dataset $X$, outputs the posterior probabilities $P(H_i|X)$ for $i=1,2,3$. 

## Evaluation
The posterior inference implementation is evaluated on 50 random datasets (with different number of samples per dataset), sampled from the data generating process which is described above. In particular, we compute the Hellinger distance  

$$H(P,Q) = \sqrt{\frac{1}{2} \sum_{i=1}^3 (\sqrt{P_i} - \sqrt{q_i})^2}$$

between your posterior Q and the correct posterior P. The score of your submission is the average of 1−H(P,Q) across the 50 datasets. If your implementation is correct, you should get a score close to 1.0. You pass this task with a score > 0.98. 
