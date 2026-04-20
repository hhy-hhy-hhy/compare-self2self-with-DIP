# Self2Self vs DIP: 单图像自监督去噪对比实验

本仓库实现了两种单图像自监督去噪方法：**Deep Image Prior (DIP)** 和 **Self2Self**，并在 Set12 数据集上进行了全面的对比评估。

## 特性

- 实现 DIP 和 Self2Self 两种自监督去噪算法
- 支持多个噪声水平（σ = 15, 25, 35, 50）
- 提供 PSNR 和 SSIM 自动计算
- 输出带噪图与去噪图的可视化对比
- 与 BM3D 和 Neighbor2Neighbor 进行性能对比

## 环境要求

- Python 3.7+
- PyTorch (for DIP)
- TensorFlow 1.15 (for Self2Self)
- numpy, opencv-python, scikit-image, matplotlib


# 克隆仓库
git clone https://github.com/hhy_hhy_hhy/compare-self2self-with-DIP.git
cd compare-self2self-with-DIP

# DIP 去噪
python dip_denoise.py --image ./Set12/01.png --sigma 25

# Self2Self 训练与测试
python self2self_train.py --image ./Set12/01.png --sigma 25
## 引用

本实现基于以下原始论文：

- **Deep Image Prior**: Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). *Deep Image Prior*. CVPR 2018.  
  [Paper](https://arxiv.org/abs/1711.10925) | [Official Code](https://github.com/DmitryUlyanov/deep-image-prior)

- **Self2Self**: Quan, Y., Chen, M., Pang, Y., & Ji, H. (2020). *Self2Self with Dropout: Learning Self-Supervised Denoising from Single Image*. CVPR 2020.  
  [Paper](https://arxiv.org/abs/1901.08034) | [Official Code](https://github.com/scut-mingqinchen/Self2Self)

- **Neighbor2Neighbor** (if used): Huang, T., Li, S., Jia, X., Lu, H., & Liu, J. (2022). *Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images*. IEEE TIP 2022.

- **BM3D** (if used): Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). *Image denoising by sparse 3D transform-domain collaborative filtering*. IEEE TIP 2007.
