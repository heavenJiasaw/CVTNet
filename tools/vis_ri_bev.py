import numpy as np
import matplotlib.pyplot as plt


def visualize_ri_bev(npy_path):
    """
    Visualize range images and BEV images from the generated npy file
    Args:
        npy_path: path to the .npy file
    """
    # Load the data
    ri_bev = np.load(npy_path)
    n_images, h, w = ri_bev.shape
    n_range = (n_images + 1) // 2  # number of range thresholds

    # Create figure
    plt.figure(figsize=(15, 8))

    # Plot range images (top row)
    for i in range(n_range):
        plt.subplot(2, n_range, i + 1)
        plt.imshow(ri_bev[i], cmap='viridis')
        if i == 0:
            plt.title('Full Range Image')
        else:
            plt.title(f'Range Image {i}')
        plt.axis('off')

    # Plot BEV images (bottom row)
    for i in range(n_range):
        plt.subplot(2, n_range, n_range + i + 1)
        plt.imshow(ri_bev[n_range + i], cmap='viridis')
        if i == 0:
            plt.title('Full BEV')
        else:
            plt.title(f'BEV {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage
    npy_file = "/data/jiehao/dataset/3D/NCLT/velodyne_data/2012-01-08_vel/ri_bev/028126.npy"  # 替换为您的.npy文件路径
    visualize_ri_bev(npy_file)