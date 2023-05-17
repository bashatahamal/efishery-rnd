import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tensorflow as tf
import glob
import os
import shutil

st.title("Document processor")

# Load the YOLOv8 model
model_table = YOLO('./model/best_paper.pt')


def switch_to_landscape(image):
    image = Image.fromarray(image)
    width, height = image.size

    if height > width:
        image = image.transpose(Image.ROTATE_90)
    return image


def show_intersection(original_image, mask_image):
    intersection_image = cv2.add(mask_image.astype(
        np.uint8), original_image.astype(np.uint8))

    return intersection_image


def load_image_for_table_segmentation(image):
    results = model_table.predict(image)

    numpy_array = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
    numpy_array = np.where(numpy_array == 0, 1, np.where(
        numpy_array == 1, 0, numpy_array))

    binary_array_uint8 = (numpy_array * 255).astype(np.uint8)
    # Load the binary image into PIL
    binary_image = Image.fromarray(binary_array_uint8, mode='L')
    numpy_image = np.array(binary_image)

    # Convert color space if necessary
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR), results


# Upload image
uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", 'JPG', "jpeg", "png"])

if uploaded_file is not None:
    loading_placeholder = st.empty()

    with st.spinner("Loading input..."):
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # orientation correction
        image = switch_to_landscape(image)

        st.image(image, use_column_width=True,
                 caption='corrected orientation input image')
    # Clear the loading spinner
    loading_placeholder.empty()

    loading_placeholder = st.empty()
    with st.spinner("Loading segmentation..."):
        mask_table_image, results = load_image_for_table_segmentation(
            image
        )
        st.image(mask_table_image, use_column_width=True,
                 caption='table mask image from paper')
        mask_original_size = cv2.resize(
            mask_table_image, (results[0].orig_shape[1], results[0].orig_shape[0]))

        # original size
        table_img = show_intersection(results[0].orig_img, mask_original_size)
        st.image(table_img, use_column_width=True,
                 caption='table image only')

        image = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)

        # Thresholding the image
        (thresh, img_bin) = cv2.threshold(
            image, 150, 255, cv2.THRESH_BINARY_INV)
        st.image(img_bin, use_column_width=True,
                 caption='black and white table')

        # Defining a kernel length
        kernel_length = np.array(table_img).shape[1]//int(table_img.shape[0]/9)

        # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
        verticle_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, kernel_length))
        # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
        hori_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_length, 1))
        # A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Morphological operation to detect vertical lines from an image
        img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
        verticle_lines_img = cv2.dilate(
            img_temp1, verticle_kernel, iterations=3)
        st.image(verticle_lines_img, use_column_width=True,
                 caption='vertical line image')

        # Morphological operation to detect horizontal lines from an image
        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
        st.image(horizontal_lines_img, use_column_width=True,
                 caption='horizontal line image')
        # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
        alpha = 0.5
        beta = 1.0 - alpha
        # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
        img_final_bin = cv2.addWeighted(
            verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(
            img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        st.image(img_final_bin, use_column_width=True,
                 caption='horizontal and vertical line image')

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(
            img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Filter the contours to detect squares
        squares = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) == 4:
                squares.append(approx)

        # Draw bounding boxes around the detected squares
        for square in squares:
            # print(square)
            cv2.drawContours(table_img, [square], 0, (0, 0, 255), 5)
        st.image(table_img, use_column_width=True,
                 caption='square contour image')

        # Crop the square regions from the original image
        cropped_images = []
        for square in squares:
            x, y, w, h = cv2.boundingRect(square)
            if h >= 3/4 * w and h <= 3 * w:  # if square enough
                cropped = image[y:y+h, x:x+w]
                cropped_images.append(cropped)

        # Create a list of small images
        images = cropped_images
        # Define the grid layout
        grid_cols = 8  # Number of columns in the grid
        num_images = len(images)
        grid_rows = (num_images + grid_cols - 1) // grid_cols

        # Display the images in a grid
        for i in range(grid_rows):
            st.write('---' * grid_cols)  # Add a horizontal line separator
            cols = st.beta_columns(grid_cols)
            for j in range(grid_cols):
                index = i * grid_cols + j
                if index < num_images:
                    cols[j].image(images[index], use_column_width=True,
                                  caption=f"Image {index+1}")

        # Add a final horizontal line separator
        st.write('---' * grid_cols)

    loading_placeholder.empty()

    loading_placeholder = st.empty()
    recognition_results = []
    with st.spinner("Loading OCR..."):
        cropped_dir_path = './cropped/'
        try:
            shutil.rmtree(cropped_dir_path)
        except:
            pass
        try:
            os.makedirs(cropped_dir_path)
        except:
            pass
        for i, cropped in enumerate(cropped_images):
            cv2.imwrite(cropped_dir_path+str(i+1) + '.png', cropped)

        for image_path in glob.glob(cropped_dir_path + '*.png'):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Normalize the pixel values to be between 0 and 1
            normalized_image = gray / 255.0
            resized_image = cv2.resize(normalized_image, (28, 28))
            x = tf.reshape(resized_image, shape=[-1, 28, 28, 1])

            mnist_model = load_model('./model/MNIST_keras_CNN.h5')
            predictions = mnist_model.predict(x)
            # Get the predicted class labels
            predicted_classes = np.argmax(predictions, axis=1)
            # Get the detailed prediction results
            prediction_results = []
            for i, pred in enumerate(predictions):
                result = {
                    'class': int(predicted_classes[i]),
                    'probability': float(pred[predicted_classes[i]])
                }
                prediction_results.append(result)
            # print(prediction_results)
            recognition_results.append(prediction_results)

        st.write(recognition_results)

else:
    st.warning("Please upload an image file.")
