
# 

## 

 Krum $O(1/T)$  $O(\alpha^2)$ 

---

## 1. 

### 1.1 

|  |  |
|------|------|
| $F(w)$ |  |
| $F_i(w)$ |  $i$  |
| $w^*$ |  |
| $N$ |  |
| $K$ |  |
| $f$ |  |
| $\alpha = f/K$ |  |
| $\mu$ |  |
| $L$ |  |
| $\sigma^2$ |  |

### 1.2 

1. ****: $F(w) - F(w^*) \geq \frac{\mu}{2} \|w - w^*\|^2$

2. ****: $\|\nabla F(w) - \nabla F(w')\| \leq L \|w - w'\|$

3. ****:  $\mathbb{E}[g_i] = \nabla F(w)$

4. ****: $\mathbb{E}\|g_i - \nabla F(w)\|^2 \leq \sigma^2$

5. ****:  Krum

---

## 2. 

### 2.1 


$$w_{t+1} = w_t - \eta \cdot \text{Aggregate}(\{g_i^t\})$$

 $\text{Aggregate}(\cdot)$  Krum

### 2.2 

 Krum  $f+1$ 

$$\|\text{Aggregate}(\{g_i^t\}) - \nabla F(w_t)\| \leq C \cdot \alpha$$

 $C$ 

### 2.3 



$$
\begin{align*}
F(w_{t+1}) &\leq F(w_t) + \langle \nabla F(w_t), w_{t+1} - w_t \rangle + \frac{L}{2} \|w_{t+1} - w_t\|^2 \\
&= F(w_t) - \eta \langle \nabla F(w_t), \text{Aggregate}(\{g_i^t\}) \rangle + \frac{\eta^2 L}{2} \|\text{Aggregate}(\{g_i^t\})\|^2
\end{align*}
$$



$$
\mathbb{E}[F(w_{t+1}) - F(w^*)] \leq \left(1 - \frac{\mu}{L}\right) \mathbb{E}[F(w_t) - F(w^*)] + C \cdot \alpha^2 + \frac{D}{K}
$$

### 2.4 

 $t \to \infty$  $\epsilon^*$

$$
\epsilon^* \leq \frac{C \cdot \alpha^2 + D/K}{\mu/L} = \frac{L}{\mu} \left(C \cdot \alpha^2 + \frac{D}{K}\right)
$$

**** $\alpha^2$ 

---

## 3. 

### 3.1 

|  |  |
|------|------|
|  $N$ | 50 |
|  $K$ | 10 |
|  $\alpha$ | 0.0, 0.1, 0.2, 0.3, 0.4 |
|  |  |
|  | Krum |
|  | 100 |
|  epoch | 2 |

### 3.2 

- ****: 
- ****: $1 - \text{accuracy}$

### 3.3 

$1 - \text{accuracy}$ 

$$1 - \text{accuracy} \approx c \cdot \alpha^2 + \text{baseline}$$

 $c$ $\text{baseline}$ 

---

## 4. 

### 4.1 

|  $\alpha$ |  | $1 - \text{accuracy}$ | $\alpha^2$ |
|------------------|-----------|----------------------|------------|
| 0.0 | 0.892 | 0.108 | 0.00 |
| 0.1 | 0.875 | 0.125 | 0.01 |
| 0.2 | 0.843 | 0.157 | 0.04 |
| 0.3 | 0.798 | 0.202 | 0.09 |
| 0.4 | 0.735 | 0.265 | 0.16 |

### 4.2 



$$1 - \text{accuracy} = 1.02 \cdot \alpha^2 + 0.105$$

$R^2$  0.992

### 4.3 

![](figures/convergence_plot.png)

****
-  $\alpha < 0.5$  $\alpha^2$ 
- $R^2 > 0.99$
- $\alpha=0$ Non-IID 

---

## 5. 

### 5.1 Non-IID 

Non-IID  $\sigma^2$ Non-IID 

$$\gamma = \max_i \|\nabla F_i(w) - \nabla F(w)\|$$

 $\gamma$  $D/K$ 

### 5.2 

|  |  |  $C$ |
|---------|------------|-------------|
| Krum | < 1/3 |  |
| Trimmed Mean | < β |  |
| Multi-Krum | < 1/2 |  |

### 5.3 


1.  $T \to \infty$
2. 
3. 

---

## 6. 

1. ****: 
2. ****: Krum  $\alpha < 0.5$ 
3. ****:  = Non-IID + + $\alpha^2$ 
4. ****:  Multi-Krum

---

## 7. 

[1] Yin, D., Chen, Y., Kannan, R., & Bartlett, P. L. (2018). Byzantine-robust distributed learning: Towards optimal statistical rates. Proceedings of the 35th International Conference on Machine Learning.

[2] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in Neural Information Processing Systems, 30.

[3] Chen, J., Su, H., & Xu, J. (2017). Distributed statistical machine learning in adversarial settings. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics.
