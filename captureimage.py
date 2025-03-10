import cv2

def capture_image(filename="./Captures/captured_image0.jpg"):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Camera not found")
        return None
    
    ret, frame = cam.read()
    if ret:
        cv2.imwrite(filename, frame)
        print("Image captured successfully!")
    
    cam.release()
    cv2.destroyAllWindows()
    return filename
