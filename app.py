from fastai2.vision.all import *

import streamlit as st

IMAGE_TYPES = ["png", "jpg"]

inf_learner = load_learner('/home/ubuntu/wolf_parade/canid_export.pkl')

class_map = inf_learner.dls.vocab

output_file_path = '/home/ubuntu/wolf_parade/test.jpg'

def main():
    """Run to execute main application"""
    st.title("Wolf Parade: Image Classifier for Various Wild Canids")

    st.write("Detects canids of the following types: \n\n" + '\n\n'.join(list(class_map)))

    image = st.file_uploader("Upload a picture of a wild canid for classification.", IMAGE_TYPES)

    if image:
        st.image(image, use_column_width=True)
        with open(output_file_path, 'wb') as output:
            output.write(image.getbuffer())

        predictions = inf_learner.predict(output_file_path)

        predicted_animal = predictions[0]

        predicted_probability = predictions[2].max() * 100

        prediction_map = zip(
            class_map, predictions[2]
        )

        prediction_string = f"The model thinks this is a {predicted_animal} with a probability of {predicted_probability} percent."

        st.text(prediction_string)

        st.text("Other details:")

        for class_prob in prediction_map:
            class_name = class_prob[0]
            class_pct = class_prob[1] * 100
            class_string = f"Chance of being a {class_name}: {class_pct} percent."
            st.write(class_string)


main()
