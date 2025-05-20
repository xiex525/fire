$$
{线性回归的基础带hat的都是估计量，不带的是我们假设的隐藏真值;y_{i}，x_{i}是观测值}\\
\vec{y}_{i}=\vec{\alpha}+\vec{\beta}{x_{i}}+\stackrel{\rightharpoonup}{\epsilon} \quad \vec{y}_{N \times 1} \quad \vec{\alpha}_{N \times 1} \quad \vec{\beta}_{N \times K} \quad \vec{\varepsilon}_{N \times 1}\\
{ }\\
\begin{array}{l}
\vec{y}=\vec{\alpha}+\vec{\beta} x_{i} \\
\vec{y_{i}}=\vec{\alpha}+\vec{\beta} x_{i}+\vec{\varepsilon_{i}} \\
\vec{y}_{i}=\hat{\vec{\alpha}}+\hat{\vec{\beta}} x_{i}+\hat{\vec{\varepsilon_{i}}}
\end{array}
$$

$$
\begin{array}{l}
 { 最小二乗:(以下所有的推导均在\vec{\epsilon}服从正态分布的假设下导出) } \\
\underset{a}\arg\min\sum_{i=1}^{T}\left\|\vec{y}_{i}-\vec{\alpha}-\vec{\beta} x_{i}\right\|^{2} \\
\frac{\partial}{\partial \vec{a}} \sum_{i=1}^{T}\left(\vec{y}_{i}-\vec{\alpha}-\vec{\beta} x_{i}\right)^{\prime}\left(\vec{y}_{i}-\vec{\alpha}-\vec{\beta} x_{i}\right) \\
=\frac{\partial}{\partial \vec{a}} \sum_{i=1}^{T} \operatorname{Tr}\left(\vec{y}_{i}-\vec{\alpha}-\vec{\beta} x_{i}\right)^{\prime}\left(\overrightarrow{y_{i}}-\vec{\alpha}-\vec{\beta} x_{i}\right) \\
=\frac{\partial}{\partial \vec{a}} \sum_{i=1}^{T} \operatorname{Tr}\left(\vec{y}_{i}^{\prime} y_{i}-\vec{y}_{i}^{\prime} \vec{a}-\vec{y}_{i}^{\prime} \vec{\beta}^{\prime} x_{i}-\vec{\alpha}^{\prime} \vec{y}_{i}+\vec{\alpha}^{\prime} \vec{\alpha}+\vec{\alpha}^{\prime} \vec{\beta} x_{i} - x_{i}^{\prime} \vec{\beta}^{\prime} \vec{y}_{i}+x_{i}^{\prime} \vec{\beta}^{\prime} \vec{\alpha}+\left(\vec{\beta} x_{i}\right)^{\prime}\left(\vec{\beta} x_{i}\right)\right) \\

{ }\\
{ 对\alpha求导等于0 } \\
\Rightarrow\sum_{i=1}^{T}-2 \vec{y}_{i}+2 \vec{a}+2 \vec{\beta}x_{i}=0 \\

{ 对\beta求导等于0}\\
\frac{\partial}{\partial \vec{\beta}} \sum_{i=1}^{T} \operatorname{Tr}\left(\vec{y_{i}}-\vec{\alpha}-\vec{\beta} x_{i}\right)^{\prime}\left(\vec{y_{i}}-\vec{\alpha}-\vec{\beta} x_{i}\right)=\sum_{i=1}^{T}-\vec{y_{i}} x_{i}^{\prime}+\vec{\alpha} x_{i}^{\prime}-x_{i} \vec{y}_{i}^{\prime} + x_{i} \vec{\alpha}^{\prime}+2 \vec{\beta}{x}_{i}{x}_{i}^{\prime}=0 \\
{ }\\
{求\alpha和\beta的值}\\
\left\{\begin{array}{ll}
\sum_{i=1}^{T}-2 \overrightarrow{y_{i}}+2 \vec{\alpha}+2 \vec{\beta} x_{i}=0  \\
\sum_{i=1}^{T}-2 \overrightarrow{y_{i}} x_{i}^{\prime}+2 \vec{\alpha} x_{i}^{\prime}+2 \vec{\beta} {x}_{i} x_{i}^{\prime}=0 \\
\end{array}\right. \\
\Rightarrow\left\{\begin{array}{lll}
2 T \bar{y}+2 T \vec{\alpha}+2 T \vec{\beta} \vec{x}=0 \Rightarrow \hat{\vec{\alpha}}=\bar{y}-\vec{\beta} {\bar{x}}\\
\sum_{i=1}^{T}-2 \vec{y_{i}} x_{i}^{\prime}+(2 {\vec{y}}-2 \vec{\beta} \bar{x}) x_{i}^{\prime}+2 \vec{\beta} \vec{x}_{i} \vec{x}_{i}^{\prime}=0\\
\end{array}\right. \\
{ }\\
{\hat{\vec{\beta}}的方差}\\

\hat{\vec{\beta}}=\sum_{i=1}^{T}\left(\overrightarrow{y_{i}}-\vec{y}\right) {x}_{i}^{\prime}\cdot \left(\sum_{i=1}^{T}\left({x_{i}}-\bar{x}\right) \vec{x}_{i}^{\prime}\right)^{-1}
{}
\hat{\vec{\beta}}\\
=\vec{\beta}+\sum_{i=1}^{T}\left(\vec{\varepsilon}_{i}-\bar{\varepsilon}_{i}\right) x_{i}^{\prime}\left[\sum_{i=1}^{T}\left(x_{i}-\bar{x}\right) x_{i}^{\prime}\right]^{-1}\\
=\vec{\beta}+\sum_{i=1}^{T}\left(\vec{\varepsilon}_{i}-\varepsilon_{i}\right)\left(x_{i}-\bar{x}\right)^{\prime}\left[\sum_{i=1}^{T}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{\prime}\right]^{-1} \\


{原因：x_{i}以及\bar{x}都是常数(这个不管一元多元都是这样的)，用到的假设是\epsilon的方差与x无关，即\operatorname{Var}(\epsilon|x)=\sigma^2,\Sigma=\sigma^2I}\\
\operatorname{Var}(\hat{\beta})=\sigma^{2}\left[\sum_{i=1}^{T}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{\prime}\right]^{-1}[\Sigma_{i=1}^{T}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{\prime}]\left[\sum_{i=1}^{T}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{\prime}\right]^{-1}=\Omega^{-1} \Sigma \quad \quad \quad \quad \Omega=\frac{1}{T} \sum_{i=1}^{T}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{\prime} \\

{\hat{\alpha}的方差}\\
\operatorname{Var}(\hat{\alpha})=\frac{\sum}{T}+\bar{x}^{\prime} \operatorname{Var}(\hat{\beta})\bar{x}=\frac{1}{T}\left(1+\bar{x}^{\prime} \Omega^{-1} \bar{x}\right) \Sigma \\
\left[\hat{\vec{a}}=\bar{y}-\hat{\vec{\beta}} \bar{x}=\alpha+\vec{\beta} \bar{x}+\frac{1}{T} \sum_{i=1}^{T} \vec{\varepsilon}_{i}-\left(\vec{\beta}+\sum_{i=1}^{T}\left(\varepsilon_{i}-\bar{\varepsilon}_{i}\right) x_{i}^{\prime}\left[\sum_{i=1}^{T}\left(x_{i}-\bar{x}\right) x_{i}^{\prime}\right]^{-1}\right) \bar{x} =\frac{1}{T} \sum_{i=1}^{T} \vec{\varepsilon}_{i}+\sum_{i=1}^{T}\left(\vec{\varepsilon}_{i}-\bar{\varepsilon}_{i}\right) x_{i}^{\prime}\left[\sum_{i=1}^{T}\left(x_{i}-\bar{x}\right) x_{i}^{\prime}\right]^{-1} \bar{x}\right] \\
\Rightarrow \hat{\vec{\alpha}} \sim N_{N}\left(\vec{\alpha}, \frac{1}{T}\left(1+\bar{x} \Omega^{-1} \bar{x}\right)^{-1} \Sigma\right)\\


{ }\\
{ 假设\vec{\alpha}=0} \\

(T-K-1) \hat{\Sigma}\sim W_{N}(T-K-1, \Sigma)\quad \quad({因为\Sigma的估计量为} \hat{\Sigma}=\frac{1}{T-k-1}(\hat{\varepsilon}-\bar{\hat{\varepsilon}})(\hat{\varepsilon}-\bar{\hat{\varepsilon}})^{\prime}\quad最小二乘的假设里E(\hat{\epsilon})=0,因此\bar{\hat{\varepsilon}}=0) \\
\sqrt{T / (1+\bar{x} \Omega^{-1} \bar{x}}) \cdot \hat{\vec{\alpha}} \sim N_{N}\left(0,  \Sigma\right)\\
{ }\\
{ 构造 Hotelling  T^{2}  statistics}\\
\left(统计量的构造方法如下：x \sim N_{p}(0, \Sigma),\quad w \sim w_{p}(n, \Sigma)\quad \Rightarrow \frac{n-p+1}{p n} n x^{\prime} w^{-1} x \sim F(p, n-p+1)\right) \\
{ 用\sqrt{T / (1+\bar{x} \Omega^{-1} \bar{x}}) \cdot \hat{\vec{\alpha}} \sim N_{N}\left(0,  \Sigma\right)代替x}\\
{ 用(T-K-1) \hat{\Sigma}\sim W_{N}(T-K-1, \Sigma)代替\omega}\\
{ 即p=N，n=T-K-1}\\
\Rightarrow \frac{T(T-K-N)}{N(T-K-1)}\cdot
\hat{\alpha}^{\prime} \sum^{-1} \hat{\alpha}\left(\frac{1}{1+\bar{x}^{\prime} \Omega^{-1} \bar{x}}\right) \sim F_{N, T-K-N}
\end{array}
$$



