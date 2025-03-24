import cv2
import numpy as np
import argparse
import os


def main(args):
    # Load the image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not load image from {args.image_path}")
        return
    # Initialize the BING objectness saliency detector
    saliency = cv2.saliency.ObjectnessBING_create()
    
    # Load the BING model
    saliency.setTrainingPath(args.model_path)
    # Compute the saliency map
    (success, saliencyMap) = saliency.computeSaliency(img)
    numDetections = saliencyMap.shape[0]
    
    # Create an empty heatmap
    heatmap = np.zeros(img.shape[:2], dtype=np.float32)
    
    # Add each detection to the heatmap
    for i in range(0, min(numDetections, args.max_detections)):
        # extract the bounding box coordinates
        (startX, startY, endX, endY) = saliencyMap[i].flatten()
        startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
        
        # Higher weight for earlier detections (more important regions)
        weight = 1.0 - (i / args.max_detections)
        heatmap[startY:endY, startX:endX] += weight
    
    # Normalize heatmap to 0-255 range
    if np.max(heatmap) > 0:
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    
    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend with original image
    alpha = 0.6  # Transparency parameter
    overlay = cv2.addWeighted(img, 1-alpha, colored_heatmap, alpha, 0)
    
    # Display results
    cv2.imshow("Original", img)
    cv2.imshow("Heatmap", colored_heatmap)
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    image_path = r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train\n02749479\n02749479_2417.JPEG"
    model_path = r"C:\Users\joren\Documents\_Uni\Master\Thesis\models"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=image_path, help="Path to the input image")
    parser.add_argument("--model_path", type=str, default=model_path, help="Path to the BING models")
    parser.add_argument("--max_detections", type=int, default=500, help="Maximum number of objectness detections")
    args = parser.parse_args()
    main(args)