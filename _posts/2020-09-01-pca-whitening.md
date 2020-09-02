---
layout: post
title: pca-whitening
date: 2020-09-01 23:44 -0700
---

Dealing with image data, the raw input is redundant, since adjacent pixel values are highly correlated. The purpose of **whitening** is to i) make features less correlated with each other. ii) give all of the features the same variance.

Recall the last post about [PCA]({% post_url 2020-09-01-pca %}), the eigenvectors are orthogonal vectors. If we get the eigenvectors of data set, and we project data onto the new space, naturally, the first purpose of whitening will be satisfied. So, **whitening** can be two-step:

1. PCA on the data set $X$ and get the new $X^\prime$
2. Normalize the covariance to make them equal to 1.

Let's dive into the code. First, we generate some data points.

```python
import numpy as np
import matplotlib.pyplot as plt

# generate sample data points.
x = np.arange(50)
delta = np.random.uniform(-10,10, size=(50,))
y = .4 * x +3 + delta
data = np.asarray([[x[i],y[i]] for i in range(len(x))])

plt.scatter(x, y)
plt.xlim(-10, 60)
plt.ylim(-10, 40)
plt.grid()
```

![](/images/2020-09-01-pca-whitening/raw_data.png#center)
*raw image data points*

We first perform PCA on data matrix:
```python
cov_mat = np.cov(data.T)
U, S, V = np.linalg.svd(cov_mat)

u1 = U[:,0] # [-0.94212686, -0.3352566 ]
u2 = U[:,1] # [-0.3352566 ,  0.94212686]
```
We obtain two eigenvectors: `u1`, `u2`, which are column vectors of `U`, and display the two eigenvector in orignal space. We change the director of `u1` here to make it fit the plot.

```python
plt.scatter(x, y)
plt.xlim(-10, 60)
plt.ylim(-10, 40)
plt.grid()
plt.arrow(0,0,-u1[0]*50,-u1[1]*50, head_width=2, color='red')
plt.arrow(0, 0, u2[0]*20, u2[1]*20, head_width=2, color='green')
```
![](/images/2020-09-01-pca-whitening/eigen_vectors.png#center)
*two eigenvectors in original space*

Then we project data matrix onto the new space under two orthogonal vectors `u1` and `u2`.

$$
  X^\prime = XU
$$

```python
p_data = np.matmul(data, U)
```
![](/images/2020-09-01-pca-whitening/project_data.png#center)
*projected data*

After PCA, we need normalize the variance of each dim.

$$
    X^{\prime\prime} = \frac{X^\prime}{\sqrt{\lambda_i+\epsilon}}
$$

where $\lambda_i$ is the i-th eigenvalue of covariance, which is also the variance of i-th feature. So we divide $X^\prime$ by $\sqrt{\lambda_i}$ to make sure unit variance of each feature. $\epsilon$ is to avoid zero-divided error.

```python
update_lam = S + np.asarray([1e-5, 1e-5])
sqrt_update_lam = np.sqrt(update_lam)
w_data = p_data / sqrt_update_lam
```
Finally we plot whitening data.
![](/images/2020-09-01-pca-whitening/whitening_data.png#center)
*whitening data*

If we want to do ZCA, we just need mutiply $U^\top$ to map back to original space.

$$
  X^{\prime\prime\prime} = X^{\prime\prime}U^\top
$$

```python
zca_data = np.matmul(w_data, U.T)
```
![](/images20-p9a-1hitening/zca_whitening.png#center)
*zca whitening data*

**About why whitening is beneficial?** Quote from [here](http://mccormickml.com/2014/06/03/deep-learning-tutorial-pca-and-whitening/)

> "This is a common trick to simplify optimization process to find weights. If the input signal has correlating inputs (some linear dependency) then the [cost] function will tend to have "river-like" minima regions rather than minima points in weights space. As to input whitening - similar thing - if you don't do it - error function will tend to have non-symmetrical minima "caves" and since some training algorithms have equal speed of update for all weights - the minimization may tend to skip good places in narrow dimensions of the minima while trying to please the wider ones. So it does not directly relate to deep learning. If your optimization process converges well - you can skip this pre-processing."
