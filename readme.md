第n层第i个神经元记为 $x_{i}^{n}$

$$\newcommand{\part}[2]{\frac{\partial #1}{\partial #2}}
\begin{align*}
    x_{i}^{n} &= f(u_{i}^{n}) \\
    u_i^n &= \sum_{j}w_{i,j}^{n-1}x^{n-1}_{j} \\
    L&=\frac12\sum_i(x_i^N-t_i)^2\\
    最后一层&
    \part{L}{}
\end{align*}$$

