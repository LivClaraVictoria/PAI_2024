## Task Description

The task is to implement Bayesian inference in a simple setting. In particular, the setting is as follows. You are given a set of data points $X={x_1,…,x_n}$ which are sampled i.i.d. from one of the following three distributions: 

* Normal distribution:  
  &ensp;&ensp; $$p_1(x) = \frac{1}{\sqrt(2\pi)\sigma}e^{-\frac{1}{2}(\frac{x}{\sigma})^2}, &ensp;&ensp;\sigma = \sqrt(2)$$
* Laplace distribution:  
  &ensp;&ensp; $$p_2(x) = \frac{1}{2b}e^{-\frac{1}{b}\|x|}, &ensp;&ensp; b = 1$$
* Student's t-distribution:  
  &ensp;&ensp; $$p_3(x) = \frac{\Gamma ((v+1)/2)}{\sqrt(v \pi) \Gamma (v/2)}(1+\frac{x^2}{v})^{-(v+1)/2}, &ensp;&ensp; v = 4$$
