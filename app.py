import streamlit as st
import torch
from load_mvp import get_instance_segmentation_model, predict_new_image
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
device='cpu'
model = get_instance_segmentation_model(2)
model.load_state_dict(torch.load('./models/model_mask-trained_cpu.pkl'))
model.to(device)
model.eval()



st.title('App')

st.subheader('Cats detector')
image_file = st.file_uploader("Upload a file", type=("jpg"))
if st.button('Plot image'):
    if image_file is None:
        st.write("load a image")
    else:
        file = base64.b64encode(image_file.read())
        jpg_original = base64.b64decode(file)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        original_image = cv2.imdecode(jpg_as_np, flags=1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.show()
        st.pyplot()
if st.button('Execute model'):
    if image_file is None:
        st.write("load a image")
    else:
        file = base64.b64encode(image_file.read())
        jpg_original = base64.b64decode(file)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        original_image = cv2.imdecode(jpg_as_np, flags=1)
        predict_new_image(original_image, model)
        st.pyplot()

