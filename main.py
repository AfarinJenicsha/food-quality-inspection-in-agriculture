
import cv2
import numpy as np

def classify_strawberry(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image")
        return None

    # Resize for consistency (optional)
    image = cv2.resize(image, (600, 400))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for fresh (red) regions
    lower_red1 = np.array([0, 70, 50])      # Lower red range (0-10)
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])    # Upper red range (160-180)
    upper_red2 = np.array([180, 255, 255])

    # Create masks for fresh (red) regions
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    fresh_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # Defect mask (non-red regions)
    defect_mask = cv2.bitwise_not(fresh_mask)

    # Remove small noise (optional)
    kernel = np.ones((5, 5), np.uint8)
    defect_mask_cleaned = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)

    # Calculate percentage of fresh vs. defective area
    total_pixels = image.shape[0] * image.shape[1]
    fresh_pixels = cv2.countNonZero(fresh_mask)
    defect_pixels = total_pixels - fresh_pixels
    fresh_percentage = (fresh_pixels / total_pixels) * 100

    # Detect large rotten patches (brown/black)
    lower_brown = np.array([10, 50, 20])    # Brown/dark defects
    upper_brown = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_pixels = cv2.countNonZero(brown_mask)

    # Classification logic
    if fresh_percentage > 90 and brown_pixels < 0.05 * total_pixels:
        classification = "FRESH"
        color = (0, 255, 0)  # Green
    elif fresh_percentage > 50:
        classification = "PARTIALLY SPOILED"
        color = (0, 165, 255)  # Orange
    else:
        classification = "ROTTEN"
        color = (0, 0, 255)  # Red

    # Draw results on image
    cv2.putText(image, f"Status: {classification}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(image, f"Fresh: {fresh_percentage:.1f}%", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Highlight defects
    contours, _ = cv2.findContours(defect_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display results
    cv2.imshow("Strawberry Quality Inspection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return classification

# Example usage
image_paths = ["strawberry1.jpg", "strawberry2.jpg", "strawberry3.jpg"]
for path in image_paths:
    print(f"Image: {path} → {classify_strawberry(path)}")
