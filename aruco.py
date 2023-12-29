import numpy as np
import cv2
from cv2 import aruco
import math
from PIL import Image
import torchvision.transforms as transforms
import torch # Use the input size corresponding to the model
model = None
count = 110
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"


def event_identification(img):        # NOTE: You can tweak this function in case you need to give more inputs 
    event_list = []
    im5 = img[879:963,197:281] #A
    im4 = img[683:762,675:754] #B
    im2 = img[475:558,683:763] #C
    im3 = img[470:552,184:266] #D
    im1 = img[133:216,205:285] #E

    event_list.append(im5)
    event_list.append(im4)
    event_list.append(im2)
    event_list.append(im3)
    event_list.append(im1)

    cv2.imshow("im1",im1)
    cv2.imshow("im2",im2)
    cv2.imshow("im3",im3)
    cv2.imshow("im4",im4)
    cv2.imshow("im5",im5)
    cv2.waitKey(1)

    return event_list

def classify_event(image):
    global model
    cv2.imwrite("a.jpg",image)
    
    test_image = Image.open("a.jpg")
    manual_transforms = transforms.Compose([
    transforms.Resize((150, 150), antialias=True),
    # transforms.Resize((244, 244), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
    
    test_image = manual_transforms(test_image)
    

    with torch.no_grad():
        output = model(test_image.unsqueeze(0))
    target_image_pred = output
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    class_names = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"]
    # os.remove("a.png")
    return class_names[target_image_pred_label.item()]

def classification(event_list):
    detected_list = []
    for img_index in range(0,5):
        img = event_list[img_index]
        detected_event = classify_event(img)
        # print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    return detected_list

def detect_ArUco_details(image):
    global count
    ArUco_details_dict = {} 
    ArUco_corners = {}
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)
    desired_ids = [5, 7, 6, 4]
    marker_dict = {4: [], 5: [], 6:[], 7:[]}
    marker_corners = []
    marker_id_order = [] 
    if ids is not None:
        for i in range(len(ids)):
            if ids[i][0] in desired_ids:
                marker_corners.append(corners[i][0])
                marker_id = int(ids[i][0])
                marker_id_order.append(marker_id)
                corners_list = corners[i][0]

                center_x = int(np.mean(corners_list[:, 0]))
                center_y = int(np.mean(corners_list[:, 1]))

                dx = corners_list[1][0] - corners_list[0][0]
                dy = corners_list[1][1] - corners_list[0][1]
                marker_orientation = int((180.0 / math.pi) * math.atan2(dy, dx))
                ArUco_details_dict[marker_id] = [[center_x, center_y], marker_orientation]
                ArUco_corners[marker_id] = corners_list.tolist()

        if len(marker_corners) >= 4:
            marker_centers = np.mean(np.array(marker_corners), axis=1).astype(int)
            # sorted_indices = np.argsort(marker_centers[:, 0])
            # sorted_marker_centers = marker_centers[sorted_indices]
            combined_lists = list(zip(marker_id_order, marker_centers))

            # Sort the combined lists based on the values of the first list
            sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])

            # Unpack the sorted pairs back into separate lists
            sorted_list1, final_pts = zip(*sorted_combined_lists)

            paper = image
            pts1 = np.float32(final_pts)
            x = (int(count%15)+1)*100
            # pts2 = np.float32([ [0, x],[x, x],[x, 0], [0, 0]])
            pts2 = np.float32([[1000, 0], [0,0], [1000, 1000], [0,1000]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(paper, M, (1000, 1000))

            event_list = event_identification(dst)
            detected = classification(event_list)
            print(detected)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 255, 255)  # White color in BGR
            font_thickness = 2

            # Define the text position (bottom-left corner)
            text_position = (190,869)
            cv2.putText(dst, detected[0], text_position, font, font_scale, font_color, font_thickness)
            text_position = (663,671)
            cv2.putText(dst, detected[1], text_position, font, font_scale, font_color, font_thickness)
            text_position = (671,463)
            cv2.putText(dst, detected[2], text_position, font, font_scale, font_color, font_thickness)
            text_position = (172,458)
            cv2.putText(dst, detected[3], text_position, font, font_scale, font_color, font_thickness)
            text_position = (193,121)
            cv2.putText(dst, detected[4], text_position, font, font_scale, font_color, font_thickness)
            cv2.imshow("img", dst)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Press 's' to save the current frame
                cv2.imwrite(f'saved_image{count}.jpg', dst)
                count = count+1
                print("Image saved!")
                
            else:
                pass


        else:
            print("Not enough desired markers found in the image.")

    return ArUco_details_dict, ArUco_corners

def mark_ArUco_image(image, ArUco_details_dict, ArUco_corners):
    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.circle(image, center, 5, (0, 0, 255), -1)
        corner = ArUco_corners[int(ids)]
        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2)
        display_offset = int(math.sqrt((tl_tr_center_x - center[0])**2 + (tl_tr_center_y - center[1])**2))
        cv2.putText(image, str(ids), (center[0] + int(display_offset / 2), center[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return image

def set_resolution(cap, width, height):
    print("resolution set entered")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print("resolution set exit")

if __name__ == "__main__":
    model_path = '/home/adi/GG_1267/Task_1A_git/Task_4a/modelx.pth'  #model uploaded on google drive
    model = torch.load(model_path,map_location='cpu')
    model.eval()
    marker = 'aruco'
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Attempting to set resolution")
    set_resolution(cap, 1920, 1080)
    print("Resolution set")

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Assuming these functions are correctly implemented
        ArUco_details_dict, ArUco_corners = detect_ArUco_details(img)
        img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners)

        cv2.imshow("Marked Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
