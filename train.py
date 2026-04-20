import tensorflow as tf
import network.Punet
import numpy as np
import util
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ========== 参数设置 ==========
LEARNING_RATE = 1e-4
N_PREDICTION = 50
N_SAVE = 500
N_STEP = 5000


def train(file_path, dropout_rate, sigma=25):
    print(f"训练: {file_path}, sigma={sigma}")
    
    tf.reset_default_graph()
    gt = util.load_np_image(file_path)
    
    # 保存路径
    img_name = os.path.basename(file_path).split('.')[0]
    model_path = f"./results/{img_name}/sigma{sigma}/"
    os.makedirs(model_path, exist_ok=True)
    
    # 添加噪声
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    
    # 构建网络
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy=False)
    
    loss = model['training_error']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    best_psnr = 0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for step in range(N_STEP):
            feed_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _, loss_value, o_image = sess.run([optimizer, avg_op, loss, our_image], feed_dict=feed_dict)
            
            if (step + 1) % N_SAVE == 0:
                # 集成推理
                pred_sum = np.float32(np.zeros(o_image.shape))
                for j in range(N_PREDICTION):
                    feed_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_img = sess.run([slice_avg, our_image], feed_dict=feed_dict)
                    pred_sum += o_img
                
                denoised = np.squeeze(np.clip(pred_sum / N_PREDICTION, 0, 1))
                denoised_uint8 = np.uint8(denoised * 255)
                
                # 计算 PSNR 和 SSIM
                gt_np = np.squeeze(gt)
                psnr = peak_signal_noise_ratio(gt_np, denoised, data_range=1)
                ssim = structural_similarity(gt_np, denoised, data_range=1)
                
                print(f"Step {step+1}, Loss: {loss_value:.6f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
                
                if psnr > best_psnr:
                    best_psnr = psnr
                    cv2.imwrite(f"{model_path}/best_denoised.png", denoised_uint8)
                
                cv2.imwrite(f"{model_path}/denoised_step{step+1}.png", denoised_uint8)
                saver.save(sess, f"{model_path}/model.ckpt-{step+1}")
    
    return best_psnr


if __name__ == '__main__':
    # 作业要求的噪声水平
    sigma_list = [25, 35, 50]
    
    # 直接指定单张图片路径（改成你的图片路径）
    image_path = 'D:/Set12/01.png'
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在 {image_path}")
        exit(1)
    
    print(f"处理单张图片: {image_path}")
    
    # 对每个噪声水平训练
    for sigma in sigma_list:
        psnr = train(image_path, dropout_rate=0.3, sigma=sigma)
        print(f"σ={sigma}: 最佳 PSNR={psnr:.2f}")
    
    print("全部完成！")