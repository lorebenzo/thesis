from matplotlib import pyplot as plt

def plot_two_images(image1, image2, title1, title2, cmap1='hot', cmap2='hot'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image1, cmap=cmap1)
    ax[0].set_title(title1)
    ax[0].axis('off')
    ax[1].imshow(image2, cmap=cmap2)
    ax[1].set_title(title2)
    ax[1].axis('off')
    plt.show()
    
    
def plot_three_images(image1, image2, image3, title1, title2, title3, cmap1='hot', cmap2='hot', cmap3='hot'):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image1, cmap=cmap1)
    ax[0].set_title(title1)
    ax[0].axis('off')
    ax[1].imshow(image2, cmap=cmap2)
    ax[1].set_title(title2)
    ax[1].axis('off')
    ax[2].imshow(image3, cmap=cmap3)
    ax[2].set_title(title3)
    ax[2].axis('off')
    plt.show()
