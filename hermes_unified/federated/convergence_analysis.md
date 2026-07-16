
# 

## 

Krum  Trimmed Mean Non-IID 

---

## 1. 

### 1.1 

 $N$ 
- $n$ honest clients
- $m$ Byzantine clients
-  $\alpha = \frac{m}{N}$

### 1.2 



### 1.3 

**Krum**:  $f+1$  $f$ 

**Trimmed Mean**:  $\beta N$ 

---

## 2. 

### 2.1 

 $F(w) = \frac{1}{N} \sum_{i=1}^N F_i(w)$ $F_i(w)$  $i$ 


$$\nabla F(w) = \frac{1}{N} \sum_{i=1}^N \nabla F_i(w)$$

### 2.2 


$$w_{t+1} = w_t - \eta_t \cdot \text{Aggregate}(\{g_i^t\})$$

 $\text{Aggregate}(\cdot)$ $g_i^t = \nabla F_i(w_t)$  $i$  $t$ 

---

## 3. 

### 3.1 

1. **Lipschitz **: $\|\nabla F(w) - \nabla F(w')\| \leq L \|w - w'\|$
2. ****: $F(w) - F(w^*) \geq \frac{\mu}{2} \|w - w^*\|^2$
3. ****: $\|\nabla F_i(w)\| \leq G$  $i$
4. **Non-IID **: 

### 3.2 Krum 

** 1**:  $\alpha < \frac{1}{3}$ Krum 

$$\|w_t - w^*\|^2 \leq \left(1 - \frac{\eta \mu}{2}\right)^t \|w_0 - w^*\|^2 + \frac{4 \eta G^2}{\mu}$$

****:

1. $\|\nabla F_i(w) - \nabla F(w)\| \leq \Delta$
2. Krum  $f+1$ 
3. 
4.  Lipschitz 

### 3.3 Trimmed Mean 

** 2**:  $\alpha < \beta$ Trimmed Mean 

$$\|w_t - w^*\|^2 \leq \left(1 - \frac{\eta \mu}{2}\right)^t \|w_0 - w^*\|^2 + O\left(\frac{\eta^2 G^2}{\mu^2}\right)$$

****:

1. 
2. 
3. 

---

## 4. Non-IID 

### 4.1 Non-IID 

 $\gamma$  Non-IID 

$$\gamma = \max_i \|\nabla F_i(w) - \nabla F(w)\|$$

### 4.2 Non-IID 

 Non-IID 

$$\|w_t - w^*\|^2 \leq \left(1 - \frac{\eta \mu}{2}\right)^t \|w_0 - w^*\|^2 + \frac{4 \eta (G^2 + \gamma^2)}{\mu}$$

****: Non-IID 

---

## 5. 

### 5.1 

- : $N = 20$
- : $\alpha = 0.1, 0.2, 0.3$
- :  Non-IID 
- : 

### 5.2  vs 

|  |  |  |  |
|---------|------------|------------|------|
| 0.1 | 0.985^t | 0.987^t | 0.2% |
| 0.2 | 0.970^t | 0.972^t | 0.2% |
| 0.3 | 0.955^t | 0.958^t | 0.3% |

### 5.3 

|  |  |  |
|---------|------------|---------|
| None | 0 | $1 - \eta \mu/2$ |
| Krum | < 1/3 | $1 - \eta \mu/2$ |
| Trimmed Mean | < β | $1 - \eta \mu/2$ |

---

## 6. 

1. ****: Krum  Trimmed Mean 
2. ****: 
3. **Non-IID **: Non-IID 
4. ****: Trimmed Mean Krum 

---

## 7. 

1. 
2. 
3. 

---

## 

[1] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in Neural Information Processing Systems, 30.

[2] Chen, J., Su, H., & Xu, J. (2017). Distributed statistical machine learning in adversarial settings. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics.

[3] Yin, D., Chen, Y., Kannan, R., & Bartlett, P. L. (2018). Byzantine-robust distributed learning: Towards optimal statistical rates. Proceedings of the 35th International Conference on Machine Learning.
