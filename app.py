# import cv2
# import numpy as np
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# from collections import deque
# import tempfile
# import os

# def plt_show(image, title=""):
#     if len(image.shape) == 3:
#         st.image(image, caption=title, use_column_width=True)

# def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
#     dim = None
#     (h, w) = image.shape[:2]

#     if width is None and height is None:
#         return image

#     if width is None:
#         r = height / float(h)
#         dim = (int(w * r), height)
#     else:
#         r = width / float(w)
#         dim = (width, int(h * r))

#     resized = cv2.resize(image, dim, interpolation=inter)
#     return resized

# def process_image(image):
#     st.image(image, caption="Original Image", use_column_width=True)
    
#     resized_image = image_resize(image, width=275, height=180)
#     plt_show(resized_image, title="Resized Image")

#     gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite('grayImg.jpg', gray_image)
    
#     plt_show(gray_image, title="Grayscale Image")

#     return resized_image, gray_image

# def detect_contours(image):
#     ret, thresh = cv2.threshold(image, 127, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def draw_and_display_contours(image, contours):
#     image_with_contours = image.copy()
#     cv2.drawContours(image_with_contours, contours, -1, (0, 250, 0), 1)
#     plt_show(image_with_contours, title="Contours on Image")

# def additional_processing_and_display(image):
#     blur = cv2.blur(image, (5, 5))
#     plt_show(blur, title="Blurred Image")

#     gblur = cv2.GaussianBlur(image, (5, 5), 0)
#     plt_show(gblur, title="Gaussian Blurred Image")

#     median = cv2.medianBlur(image, 5)
#     plt_show(median, title="Median Blurred Image")

#     kernel = np.ones((5, 5), np.uint8)
#     erosion = cv2.erode(median, kernel, iterations=1)
#     dilation = cv2.dilate(erosion, kernel, iterations=5)
#     closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
#     edges = cv2.Canny(dilation, 9, 220)

#     plt_show(erosion, title="Erosion Image")
#     plt_show(closing, title="Closing Image")
#     plt_show(edges, title="Edges Image")

# def road_damage_assessment(uploaded_video):
#     best_model = YOLO('model/best.pt')

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     text_position = (40, 80)
#     font_color = (255, 255, 255)
#     background_color = (0, 0, 255)

#     damage_deque = deque(maxlen=20)

#     # Save the uploaded video to a temporary file
#     temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
#     with open(temp_video_path, "wb") as temp_video:
#         temp_video.write(uploaded_video.read())

#     cap = cv2.VideoCapture(temp_video_path)

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('road_damage_assessment.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             results = best_model.predict(source=frame, imgsz=640, conf=0.25)
#             processed_frame = results[0].plot(boxes=False)
            
#             percentage_damage = 0 
            
#             if results[0].masks is not None:
#                 total_area = 0
#                 masks = results[0].masks.data.cpu().numpy()
#                 image_area = frame.shape[0] * frame.shape[1]
#                 for mask in masks:
#                     binary_mask = (mask > 0).astype(np.uint8) * 255
#                     contour, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#                     total_area += cv2.contourArea(contour[0])
                
#                 percentage_damage = (total_area / image_area) * 100

#             damage_deque.append(percentage_damage)
#             smoothed_percentage_damage = sum(damage_deque) / len(damage_deque)
                
#             cv2.line(processed_frame, (text_position[0], text_position[1] - 10),
#                      (text_position[0] + 350, text_position[1] - 10), background_color, 40)
            
#             cv2.putText(processed_frame, f'Road Damage: {smoothed_percentage_damage:.2f}%', text_position, font, font_scale, font_color, 2, cv2.LINE_AA)         
        
#             out.write(processed_frame)

#             cv2.imshow('Road Damage Assessment', processed_frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break

#     # Release resources and delete the temporary file
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     os.remove(temp_video_path)

# def main():
#     st.title("Image and Road Damage Assessment")

#     # Image Section
#     st.markdown("## Image Section")
#     uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         original_image = np.array(image)

#         st.image(original_image, caption="Uploaded Image", use_column_width=True)

#         resized_image, gray_image = process_image(original_image)

#         if gray_image is not None:
#             contours = detect_contours(gray_image)

#             if contours:
#                 draw_and_display_contours(resized_image, contours)
#                 st.success("Pothole Detected!")
#             else:
#                 st.warning("No Pothole Detected!")

#             additional_processing_and_display(gray_image)

#     # Video Section
#     st.markdown("---")  # Separation between image and video sections
#     st.markdown("## Video Section")
#     uploaded_video = st.file_uploader("Choose a video...", type="mp4")
    
#     if uploaded_video is not None:
#         road_damage_assessment(uploaded_video)

# if __name__ == "__main__":
#     main()







# import cv2
# import numpy as np
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# from collections import deque
# import io
# import tempfile
# import os
# from demand import load_data, show_demand_analysis

# def plt_show(image, title=""):
#     if len(image.shape) == 3:
#         st.image(image, caption=title, use_column_width=True)

# def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
#     dim = None
#     (h, w) = image.shape[:2]

#     if width is None and height is None:
#         return image

#     if width is None:
#         r = height / float(h)
#         dim = (int(w * r), height)
#     else:
#         r = width / float(w)
#         dim = (width, int(h * r))

#     resized = cv2.resize(image, dim, interpolation=inter)
#     return resized

# def process_image(image):
#     st.image(image, caption="Original Image", use_column_width=True)
    
#     resized_image = image_resize(image, width=275, height=180)
#     plt_show(resized_image, title="Resized Image")

#     gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite('grayImg.jpg', gray_image)
    
#     plt_show(gray_image, title="Grayscale Image")

#     return resized_image, gray_image

# def detect_contours(image):
#     ret, thresh = cv2.threshold(image, 127, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def draw_and_display_contours(image, contours):
#     image_with_contours = image.copy()
#     cv2.drawContours(image_with_contours, contours, -1, (0, 250, 0), 1)
#     plt_show(image_with_contours, title="Contours on Image")

# def additional_processing_and_display(image):
#     blur = cv2.blur(image, (5, 5))
#     plt_show(blur, title="Blurred Image")

#     gblur = cv2.GaussianBlur(image, (5, 5), 0)
#     plt_show(gblur, title="Gaussian Blurred Image")

#     median = cv2.medianBlur(image, 5)
#     plt_show(median, title="Median Blurred Image")

#     kernel = np.ones((5, 5), np.uint8)
#     erosion = cv2.erode(median, kernel, iterations=1)
#     dilation = cv2.dilate(erosion, kernel, iterations=5)
#     closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
#     edges = cv2.Canny(dilation, 9, 220)

#     plt_show(erosion, title="Erosion Image")
#     plt_show(closing, title="Closing Image")
#     plt_show(edges, title="Edges Image")

# def road_damage_assessment(uploaded_video):
#     import torch

#     # Set the device to CPU
#     device = torch.device('cpu')

#     # Load the YOLO model
#     best_model = YOLO('model/best.pt')
#     best_model.to(device)

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     text_position = (40, 80)
#     font_color = (255, 255, 255)
#     background_color = (0, 0, 255)

#     damage_deque = deque(maxlen=20)

#     # Save the uploaded video to a temporary file
#     temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
#     with open(temp_video_path, "wb") as temp_video:
#         temp_video.write(uploaded_video.read())

#     cap = cv2.VideoCapture(temp_video_path)

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('road_damage_assessment.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             results = best_model.predict(source=frame, imgsz=640, conf=0.25)
#             processed_frame = results[0].plot(boxes=False)
            
#             percentage_damage = 0 
            
#             if results[0].masks is not None:
#                 total_area = 0
#                 masks = results[0].masks.data.cpu().numpy()
#                 image_area = frame.shape[0] * frame.shape[1]
#                 for mask in masks:
#                     binary_mask = (mask > 0).astype(np.uint8) * 255
#                     contour, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#                     total_area += cv2.contourArea(contour[0])
                
#                 percentage_damage = (total_area / image_area) * 100

#             damage_deque.append(percentage_damage)
#             smoothed_percentage_damage = sum(damage_deque) / len(damage_deque)
                
#             cv2.line(processed_frame, (text_position[0], text_position[1] - 10),
#                      (text_position[0] + 350, text_position[1] - 10), background_color, 40)
            
#             cv2.putText(processed_frame, f'Road Damage: {smoothed_percentage_damage:.2f}%', text_position, font, font_scale, font_color, 2, cv2.LINE_AA)         
        
#             out.write(processed_frame)

#             cv2.imshow('Road Damage Assessment', processed_frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break

#     # Release resources and delete the temporary file
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     os.remove(temp_video_path)

# def process_uploaded_video(uploaded_video):
#     temp_video_file = tempfile.NamedTemporaryFile(delete=False)
#     temp_video_file.write(uploaded_video.read())
#     temp_video_file_path = temp_video_file.name
#     temp_video_file.close()

#     road_damage_assessment(temp_video_file_path)

#     os.remove(temp_video_file_path)

# def main():
#     st.set_page_config(page_title="Urban Mobility Solution", page_icon=":car:")
#     st.title("Image and Road Damage Assessment")

#     # Image Section
#     st.markdown("## Image Section")
#     uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         original_image = np.array(image)

#         st.image(original_image, caption="Uploaded Image", use_column_width=True)

#         resized_image, gray_image = process_image(original_image)

#         if gray_image is not None:
#             contours = detect_contours(gray_image)

#             if contours:
#                 draw_and_display_contours(resized_image, contours)
#                 st.success("Pothole Detected!")
#             else:
#                 st.warning("No Pothole Detected!")

#             additional_processing_and_display(gray_image)

#     # Video Section
#     st.markdown("---")  # Separation between image and video sections
#     st.markdown("## Video Section")
#     uploaded_video = st.file_uploader("Choose a video...", type="mp4")
    
#     st.markdown("---") 
#     if uploaded_video is not None:
#         road_damage_assessment(uploaded_video)

#     st.sidebar.markdown("<span style='font-size:28px'>Urban Mobility Solution</span>", unsafe_allow_html=True)
#     st.sidebar.markdown("---")  # Separation between image and demand sections
#     st.sidebar.markdown("## Demand Prediction")

#     # Sidebar menu for demand prediction navigation
#     demand_uploaded_file = st.sidebar.file_uploader("Upload CSV file for Demand Prediction", type=["csv"])

#     # Load data if file is uploaded
#     if demand_uploaded_file:
#         df, demand_model = load_data(demand_uploaded_file)
#         show_demand_analysis(df)


# if __name__ == "__main__":
#     main()




import cv2
import numpy as np
import streamlit as st
from PIL import Image
from collections import deque
import os
from demand import load_data, show_demand_analysis

def plt_show(image, title=""):
    if len(image.shape) == 3:
        st.image(image, caption=title, use_column_width=True)

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def process_image(image):
    st.image(image, caption="Original Image", use_column_width=True)
    
    resized_image = image_resize(image, width=275, height=180)
    plt_show(resized_image, title="Resized Image")

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('grayImg.jpg', gray_image)
    
    plt_show(gray_image, title="Grayscale Image")

    return resized_image, gray_image

def detect_contours(image):
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_and_display_contours(image, contours):
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 250, 0), 1)
    plt_show(image_with_contours, title="Contours on Image")

def additional_processing_and_display(image):
    blur = cv2.blur(image, (5, 5))
    plt_show(blur, title="Blurred Image")

    gblur = cv2.GaussianBlur(image, (5, 5), 0)
    plt_show(gblur, title="Gaussian Blurred Image")

    median = cv2.medianBlur(image, 5)
    plt_show(median, title="Median Blurred Image")

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(median, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=5)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(dilation, 9, 220)

    plt_show(erosion, title="Erosion Image")
    plt_show(closing, title="Closing Image")
    plt_show(edges, title="Edges Image")

def main1():
    st.set_page_config(page_title="RouteMaster", page_icon=":car:")
    st.title("🎭 RouteMaster: Optimized Mobility for Smarter Cities")

    # Image Section
    st.markdown("## Upload the Road Image in Order to detect pothole is present on the road or not ")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_image = np.array(image)

        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        resized_image, gray_image = process_image(original_image)

        if gray_image is not None:
            contours = detect_contours(gray_image)

            if contours:
                draw_and_display_contours(resized_image, contours)
                st.success("Pothole Detected!")
            else:
                st.warning("No Pothole Detected!")

            additional_processing_and_display(gray_image)

    st.write('')
    st.markdown("---")  # Separation between image and demand sections
    st.markdown("## 📊 Demand Prediction")

    # Sidebar menu for demand prediction navigation
    demand_uploaded_file = st.file_uploader("Upload CSV file for Demand Prediction", type=["csv"])

    # Load data if file is uploaded
    if demand_uploaded_file:
        df, demand_model = load_data(demand_uploaded_file)
        show_demand_analysis(df)
        st.write("")
        st.write("")




import re
import pandas as pd
import streamlit as st
from geopy.distance import geodesic
import requests
import random
import math


def display_route(location_route, x, locations, loc_df, distance_matrix):
    num_locations = len(locations)
    route = [0]
    current_place = 0

    location_route_with_coordinates = []
    for loc in location_route:
        if isinstance(loc, str):
            location = loc_df[loc_df['Place_Name'] == loc]['Coordinates'].values[0]
            if location:
                location_route_with_coordinates.append(location)
            else:
                location_route_with_coordinates.append(None)
        else:
            location_route_with_coordinates.append(loc)

    st.write('\n')

    rows = []
    distance_total = 0
    initial_loc = ''  # starting point
    location_route_names = []  # list of final route place names in order

    for i, loc in enumerate(location_route_with_coordinates[:-1]):
        next_loc = location_route_with_coordinates[i + 1]

        # Calculate the geodesic distance between two locations
        distance = geodesic(loc, next_loc).kilometers
        distance_km_text = f"{distance:.2f} km"
        distance_mi_text = f"{distance*0.621371:.2f} mi"

        a = loc_df[loc_df['Coordinates'] == loc]['Place_Name'].reset_index(drop=True)[0]
        b = loc_df[loc_df['Coordinates'] == next_loc]['Place_Name'].reset_index(drop=True)[0]
        
        if i == 0:
            location_route_names.append(a.replace(' ', '+') + '/')
            initial_loc = (a.replace(' ', '+')) + '/'
        else:
            location_route_names.append(a.replace(' ', '+') + '/')

        distance_total += distance
        rows.append((a, b, distance_km_text, distance_mi_text))

    distance_total = int(round(distance_total*0.621371, 0))
    st.write('\n')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal Geodesic Distance", '{} mi'.format(distance_total))
        
    df = pd.DataFrame(rows, columns=["From", "To", "Distance (km)", "Distance (mi)"]).reset_index(drop=True)
    
    st.dataframe(df)  # display route with distance
    location_route_names.append(initial_loc)
    return location_route_names    
    
def tsp_solver(data_model, iterations=1000, temperature=10000, cooling_rate=0.95):
    def distance(point1, point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    num_locations = data_model['num_locations']
    locations = [(float(lat), float(lng)) for lat, lng in data_model['locations']]

    # Randomly generate a starting solution
    current_solution = list(range(num_locations))
    random.shuffle(current_solution)

    # Compute the distance of the starting solution
    current_distance = 0
    for i in range(num_locations):
        current_distance += distance(locations[current_solution[i-1]], locations[current_solution[i]])

    # Initialize the best solution as the starting solution
    best_solution = current_solution
    best_distance = current_distance

    # Simulated Annealing algorithm
    for i in range(iterations):
        # Compute the temperature for this iteration
        current_temperature = temperature * (cooling_rate ** i)

        # Generate a new solution by swapping two random locations
        new_solution = current_solution.copy()
        j, k = random.sample(range(num_locations), 2)
        new_solution[j], new_solution[k] = new_solution[k], new_solution[j]

        # Compute the distance of the new solution
        new_distance = 0
        for i in range(num_locations):
            new_distance += distance(locations[new_solution[i-1]], locations[new_solution[i]])

        # Decide whether to accept the new solution
        delta = new_distance - current_distance
        if delta < 0 or random.random() < math.exp(-delta / current_temperature):
            current_solution = new_solution
            current_distance = new_distance

        # Update the best solution if the current solution is better
        if current_distance < best_distance:
            best_solution = current_solution
            best_distance = current_distance

    # Convert the solution to the required format
    x = {}
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                if (i, j) in x:
                    continue
                if (j, i) in x:
                    continue
                if (i == 0 and j == num_locations - 1) or (i == num_locations - 1 and j == 0):
                    x[i, j] = 1
                    x[j, i] = 1
                elif i < j:
                    x[i, j] = 1
                    x[j, i] = 0
                else:
                    x[i, j] = 0
                    x[j, i] = 1

    # Create the optimal route
    optimal_route = []
    start_index = best_solution.index(0)
    for i in range(num_locations):
        optimal_route.append(best_solution[(start_index+i)%num_locations])
    optimal_route.append(0)
    
    # Return the optimal route
    location_route = [locations[i] for i in optimal_route]
    return location_route, x

# Caching the distance matrix calculation for better performance
@st.cache_data
def compute_distance_matrix(locations):    
    # using geopy geodesic for lesser compute time
    num_locations = len(locations)
    distance_matrix = [[0] * num_locations for i in range(num_locations)]
    for i in range(num_locations):
        for j in range(i, num_locations):
            distance = geodesic(locations[i], locations[j]).km
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix

def create_data_model(locations):
    data = {}
    num_locations = len(locations)
    data['locations']=locations
    data['num_locations'] = num_locations
    distance_matrix = compute_distance_matrix(locations)
    data['distance_matrix'] = distance_matrix
    return data

def geocode_address(address):
    url = f'https://photon.komoot.io/api/?q={address}'
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        if results['features']:
            first_result = results['features'][0]
            latitude = first_result['geometry']['coordinates'][1]
            longitude = first_result['geometry']['coordinates'][0]
            return address, latitude, longitude
        else:
            print(f'Geocode was not successful. No results found for address: {address}')
    else:
        print('Failed to get a response from the geocoding API.')
        
def main2():
    st.write("")
    st.write("")
    st.markdown("---") 
    st.title(" 🚀 Interactive Travel Route Planner")

    default_locations = [['Nagpur'],['Pune'],['Delhi']]
    existing_locations = '\n'.join([x[0] for x in default_locations])
    selected_value = st.text_area("Enter Locations:", value=existing_locations)

    if st.button("Calculate Optimal Route"):
        lines = selected_value.split('\n')
        values = [geocode_address(line) for line in lines if line.strip()]    
        location_names=[x[0] for x in values if x is not None] # address names
        locations=[(x[1],x[2]) for x in values if x is not None] # coordinates        
        loc_df = pd.DataFrame({'Coordinates': locations, 'Place_Name': location_names})    
        
        if locations:
                data_model = create_data_model(locations)
                solution, x = tsp_solver(data_model)

                if solution:
                    distance_matrix = compute_distance_matrix(locations)
                    location_route_names = display_route(solution, x, locations, loc_df, distance_matrix)
                    gmap_search = 'https://www.google.com/maps/dir/+'
                    gmap_places = gmap_search + ''.join(location_route_names)
                    st.write('\n')
                    st.write('[ 🗺️ Google Maps Link with Optimal Route added]({})'.format(gmap_places))
                else:
                    st.error("No solution found.")

    
if __name__ == "__main__":
    main1()
    main2()