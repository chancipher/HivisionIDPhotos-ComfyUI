import cv2
import numpy as np
import os

from whitening import make_whitening_png

# Test with an example image
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    test_image_path = os.path.join(base_dir, "test_image.png")
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found at {test_image_path}")
    
    image = cv2.imread(test_image_path)
    if image is None:
        raise ValueError("Failed to load test image")

    # Test LUT whitening
    whitened_image = make_whitening_png(image, strength=100)

    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Whitened Image", whitened_image)

    # Save results for further inspection
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "original.png"), image)
    cv2.imwrite(os.path.join(output_dir, "whitened.png"), whitened_image)
    print(f"Results saved in {output_dir}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()