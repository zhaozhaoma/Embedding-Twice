# Embedding-Twice
##### Sep, 8th, 2025
---

将文档向量表示为 $[z_1; z_2]$，查询向量表示为 $[q; q]$，然后进行 L2 归一化和点积：

$$
\text{im}_{\text{concat}} = \left\langle \frac{[q; q]}{\|[q; q]\|}, \frac{[z_1; z_2]}{\|[z_1; z_2]\|} \right\rangle
$$

---
单位向量归一化

假设 $q$, $z_1$, $z_2$ 近似单位向量：

$$
\|[q; q]\| = \sqrt{\|q\|^2 + \|q\|^2} = \sqrt{2}
$$

$$
\|[z_1; z_2]\| = \sqrt{\|z_1\|^2 + \|z_2\|^2} = \sqrt{2}
$$

---

相似度展开公式变为：

$$
\text{sim}_{\text{concat}} = \left\langle \frac{[q; q]}{\sqrt{2}}, \frac{[z_1; z_2]}{\sqrt{2}} \right\rangle = \frac{1}{2} \langle q, z_1 \rangle + \frac{1}{2} \langle q, z_2 \rangle
$$

也就是，拼接后归一化点积相当于对第一遍和第二遍的相似度得分进行等权平均。

---


从信息论角度：理论上，concat操作不会减少信息,但作用到效果上，需要存在一个最优的判决函数，其性能在拼接后不会下降。但现实中：最优的融合函数都是有损的，因此：可能导致检索性能指标下降。

---
