import matplotlib.pyplot as plt
import cv2

def plot_results(query_img, results, person, extra_text=""):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Image 1")
    plt.axis("off")

    img2, verified, dist = results[0]
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    symbol = '✓' if verified else '✗'
    plt.title(f"Image 2\n{symbol} {dist:.2f}")
    plt.axis("off")

    plt.suptitle(f"{person}: {'Match' if verified else 'No Match'}\n{extra_text}", fontsize=10)
    plt.tight_layout()
    plt.show()
