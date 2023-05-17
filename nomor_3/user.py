import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

st.title("Vibrio Extractor")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"])

# Load the YOLOv8 model
model_petri = YOLO('./model/best_petri.pt')
model_vibrio = YOLO('./model/best_vibrio.pt')

if uploaded_file is not None:
    loading_placeholder = st.empty()

    with st.spinner("Loading..."):
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # petri
        results_petri = model_petri.predict(image)
        tensor_petri = results_petri[0].masks.data
        numpy_array_petri = tensor_petri.cpu().numpy()
        pixel_petri = np.count_nonzero(numpy_array_petri)

        if tensor_petri.shape[0] > 1:
            combined_mask_petri = np.zeros(
                (tensor_petri.shape[1], tensor_petri.shape[2]), dtype=np.float32)
            for x in range(tensor_petri.shape[0]):
                numpy_array_petri = tensor_petri[x].cpu().numpy()
                combined_mask_petri = cv2.bitwise_or(
                    combined_mask_petri, numpy_array_petri)
            bgr_image_petri = combined_mask_petri
        else:
            # Convert the image to BGR format for OpenCV
            print(numpy_array_petri.shape)
            bgr_image_petri = cv2.cvtColor(
                numpy_array_petri.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        # vibrio
        results_vibrio = model_vibrio.predict(image)
        res_plotted = results_vibrio[0].plot()
        tensor = results_vibrio[0].masks.data
        combined_mask = np.zeros(
            (tensor.shape[1], tensor.shape[2]), dtype=np.float32)

        size_cls = {}
        cls_res = list(results_vibrio[0].boxes.cls.cpu().numpy())
        for x in range(tensor.shape[0]):
            numpy_array = tensor[x].cpu().numpy()
            # count pixel on class
            if cls_res[x] in size_cls:
                size_cls[cls_res[x]] = size_cls[cls_res[x]] + \
                    np.count_nonzero(numpy_array)
            else:
                size_cls[cls_res[x]] = np.count_nonzero(numpy_array)
            combined_mask = cv2.bitwise_or(combined_mask, numpy_array)

        pixel_vibrio = np.count_nonzero(combined_mask)

        cls_result = Counter(list(results_vibrio[0].boxes.cls.cpu().numpy()))
        cls_string = ''
        size_string = ''
        size_cls_string = ''
        for x in cls_result:
            cls_string += results_vibrio[0].names[x] + \
                '=' + str(cls_result[x]) + ' '
            size_string += results_vibrio[0].names[x] + ' = ' + \
                str(size_cls[x]) + ' piksel  '
            # calculate area based on pixel ratio
            size_cls_string += results_vibrio[0].names[x] + ' = ' + \
                str(round(7854/pixel_petri * size_cls[x], 2)) + ' mm2  '

        # View images
        num_columns = 2
        column_width = int(12 / num_columns)
        columns = st.beta_columns(num_columns)
        with columns[2 % num_columns]:
            st.image(image, use_column_width=True, caption='input image')
        with columns[1 % num_columns]:
            st.image(bgr_image_petri, use_column_width=True,
                     caption='mask cup ' + str(pixel_petri) + ' piksel = 7854 mm2')

        columns = st.beta_columns(num_columns)
        with columns[2 % num_columns]:
            st.image(combined_mask, use_column_width=True,
                     caption='mask vibrio ' + size_string + ' ==> ' + size_cls_string)
        with columns[1 % num_columns]:
            st.image(res_plotted, use_column_width=True,
                     caption='result vibrio ' + cls_string)

    # Clear the loading spinner
    loading_placeholder.empty()
else:
    st.warning("Please upload an image file.")
