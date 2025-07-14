---
title: Poisson's Equation
comments: true
---
# Poisson's Equation
!!! info "Related Works"
    **[Stanford course Math 220B](https://web.stanford.edu/class/math220b/lecturenotes.html)**

## 1. Problem Description
We want to solve the **Laplace's equation**

$$
\Delta u = 0
$$

and it's inhomogeneous versions, **Poisson's equation**

$$
- \Delta u = f
$$

We say a function *u* satisfying Laplace's equation is a **harmonic function**.

## 2. The Fundamental Solution

Consider Laplace's equation in $\mathbb{R}^n$,

$$
\Delta u = 0 \quad x \in \mathbb{R}^n
$$

Clearly, there are a lot of functions *u* which satisfy this equation. In particular, any constant function is harmonic. In addition, any function of the form $u(x)=a_1x_1 + ... + a_nx_n$ for constants $a_i$ is also a solution. Of course, we can list a number of others. Here, however, we are intrested in finding a particular solution of Laplace's equation which will allow us to solve Poisson's equation.

Give the symmertic nature of Laplace's equation, we look for a *radial* solution. That is, we look for a harmonic funtion $u$ on $\mathbb{R}^n$ such that $u(x)=v(|x|)$. In addition, to being a natural choice due to the symmetry of Laplace's equation, radial solutions are natural to look for because they reduce a PDE to an ODE, which is generally easier to solve. Therefore, we look for a radial solution.

If $u(x)=v(|x|)$, then

$$
u_{x_i} = \frac{x_i}{|x|}v'(|x|) \quad |x| \neq 0
$$

which implies

$$
u_{x_ix_i} = \frac{1}{|x|}v'(|x|) - \frac{x_i^2}{|x|^3}v'(|x|) + \frac{x_i^2}{|x|^2}v''(|x|) \quad |x| \neq 0
$$

Therefore,

$$
\Delta u = \frac{n-1}{|x|}v'(|x|) + v''(|x|)
$$

Letting $r = |x|$, we see that $u(x)=v(|x|)$ is a radial solution of Laplace's equation implies $v$ satisfies

$$
\frac{n-1}{r}v'(r)+v''(r) = 0
$$

And then, from some calculations, we see that for any constants $c1$, $c2$, the function

$$
u(x) \equiv
\begin{cases}
c_1 \ln |x| + c_2, & \text{if } n = 2 \\
\frac{c_1}{(2 - n)|x|^{n - 2}} + c_2, & \text{if } n \geq 3
\end{cases}
\tag{3.1}
$$

for $x \in \mathbb{R}^n$, $|x| \neq 0$ is a solution of Laplace's equation in $\mathbb{R}^n - {0}$. We notice that the function $u$ defined in (3.1) satisfies $\Delta u(x)=0$ for $x \neq 0$, but at $x=0$, $\Delta u(0)$ is undefined. We claim that we can choose constants $c_1$ and $c_2$ appropriately so that

$$
- \Delta_x u = \delta_0
$$

in the sense of distributions. Recall that $\delta_0$ is the distribution which is defined as follows. For all $\phi \in \mathbb{D}$,

$$
(\delta_0, \phi) = \phi(0)
$$

Below, we will prove this claim. For now, though, assume we can prove this. That is, assume we can find constants $c_1$,$c_2$ such that $u$ defined in (3.1) satisfied

$$
- \Delta_x u = \delta_0
\tag{3.2}
$$

Let $\Phi$ denote the solution of (3.2). Then, define

$$
v(x) = \int_{\mathbb{R}^n} \Phi(x - y) f(y) \, dy
$$

*Formally*, we compute the Laplacian of *v* as follows,

$$
\begin{aligned}
- \Delta_x v &= - \int_{\mathbb{R}^n} \Delta_x \Phi(x - y) f(y) \, dy \\
             &= - \int_{\mathbb{R}^n} \Delta_y \Phi(x - y) f(y) \, dy \\
             &= \int_{\mathbb{R}^n} \delta_x f(y) \, dy = f(x)
\end{aligned}
$$

That is, *v* is a solution of Poisson's equation! Of course, this set of equalities above is entirely formal. We have not prove anything yet. However, we have motivated a solution formula for Poisson's equation from a solution to (3.2). We now return to using the radial solution (3.1) to find a solution of (3.2).

Define the function $\Phi$ as follows. For $|x| \neq 0$, let

$$
\Phi(x) = 
\begin{cases}
-\dfrac{1}{2\pi} \ln |x|, \quad \text{if } n = 2\\
\dfrac{1}{n(n-2)\alpha(n)} \cdot \dfrac{1}{|x|^{n-2}}, \quad \text{if } n \geq 3
\end{cases}
\tag{3.3}
$$

where $\alpha (n)$ is the volume of the unit ball in $\mathbb{R}^n$. We see that $\Phi$ satisfies Laplace's equation on $\mathbb{R}^n-{0}$. As we will show in the following claim, $\Phi$ satisfies $-\Delta_x \Phi = \delta_0$. For this reason, we call $\Phi$ the **fundamental solution** of Laplace's equation.