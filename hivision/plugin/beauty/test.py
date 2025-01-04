import cv2
import numpy as np
import os
from whitening import make_whitening_png

# Test with multiple strength values
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    test_image_path = os.path.join(base_dir, "demo3.png")
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found at {test_image_path}")
    
    image = cv2.imread(test_image_path)
    if image is None:
        raise ValueError("Failed to load test image")

    # Output directory for the results
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Test with different strengths
    strengths = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
    for strength in strengths:
        # Apply whitening with the current strength
        whitened_image = make_whitening_png(image, strength=strength)

        # Save the original and whitened images
        cv2.imwrite(os.path.join(output_dir, f"original.png"), image)
        cv2.imwrite(os.path.join(output_dir, f"whitened_{strength}.png"), whitened_image)
        print(f"Test with strength {strength} completed and saved.")

    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()