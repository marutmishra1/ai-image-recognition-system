from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


# ==================================================
# Candidate categories
# ==================================================

LABELS = [

    # Documents
    "a resume or CV document",
    "a student profile",
    "an educational document",
    "a certificate",
    "an identification document",
    "a business document",
    "a handwritten document",
    "a printed text document",
    "a page containing mostly text",

    # Science / Biotechnology
    "a biotechnology laboratory",
    "a biotechnology experiment",
    "a scientific laboratory",
    "a scientist working in a laboratory",
    "a science classroom",
    "a biology experiment",
    "a medical laboratory",

    # People
    "a photograph of a person",
    "a student",
    "a teacher",
    "a scientist",
    "a professional",

    # Objects
    "a laptop",
    "a mobile phone",
    "a car",
    "a bicycle",
    "a dog",
    "a cat",
    "a bird",
    "an apple",
    "a bottle",
    "a chair",
    "a television",
    "a book",
    "a textbook",

    # Fictional / visual
    "a superhero-style character",
    "a comic-book-style character",
    "a fictional character",
    "a fantasy character",
    "an animated character",
    "an action hero character",

    # Scenes
    "a city street",
    "a classroom",
    "an office",
    "a laboratory",
    "a hospital",
    "a park",
    "a beach",
    "a landscape"
]


# ==================================================
# Human-readable names
# ==================================================

LABEL_NAMES = {

    "a resume or CV document":
        "Resume / CV",

    "a student profile":
        "Student Profile",

    "an educational document":
        "Educational Document",

    "a certificate":
        "Certificate",

    "an identification document":
        "Identification Document",

    "a business document":
        "Business Document",

    "a handwritten document":
        "Handwritten Document",

    "a printed text document":
        "Printed Text Document",

    "a page containing mostly text":
        "Text Document",

    "a biotechnology laboratory":
        "Biotechnology Laboratory",

    "a biotechnology experiment":
        "Biotechnology Experiment",

    "a scientific laboratory":
        "Scientific Laboratory",

    "a scientist working in a laboratory":
        "Scientist / Laboratory",

    "a science classroom":
        "Science Classroom",

    "a biology experiment":
        "Biology Experiment",

    "a medical laboratory":
        "Medical Laboratory",

    "a photograph of a person":
        "Person",

    "a student":
        "Student",

    "a teacher":
        "Teacher",

    "a scientist":
        "Scientist",

    "a professional":
        "Professional",

    "a laptop":
        "Laptop",

    "a mobile phone":
        "Mobile Phone",

    "a car":
        "Car",

    "a bicycle":
        "Bicycle",

    "a dog":
        "Dog",

    "a cat":
        "Cat",

    "a bird":
        "Bird",

    "an apple":
        "Apple",

    "a bottle":
        "Bottle",

    "a chair":
        "Chair",

    "a television":
        "Television",

    "a book":
        "Book",

    "a textbook":
        "Textbook",

    "a superhero-style character":
        "Superhero-Style Character",

    "a comic-book-style character":
        "Comic-Book-Style Character",

    "a fictional character":
        "Fictional Character",

    "a fantasy character":
        "Fantasy Character",

    "an animated character":
        "Animated Character",

    "an action hero character":
        "Action Hero Character",

    "a city street":
        "City Street",

    "a classroom":
        "Classroom",

    "an office":
        "Office",

    "a laboratory":
        "Laboratory",

    "a hospital":
        "Hospital",

    "a park":
        "Park",

    "a beach":
        "Beach",

    "a landscape":
        "Landscape"
}


# ==================================================
# Load model once
# ==================================================

@lru_cache(maxsize=1)
def get_classifier():

    print(
        "Loading AI image recognition model..."
    )

    classifier = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )

    print(
        "AI image recognition model loaded."
    )

    return classifier


# ==================================================
# Image preprocessing
# ==================================================

def preprocess_image(filepath):

    image = cv2.imread(
        filepath,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            "Could not read the uploaded image."
        )


    # ----------------------------------------------
    # Resize large images
    # ----------------------------------------------
    #
    # Very large phone/camera images can be several
    # thousand pixels wide. CLIP does not need that
    # much resolution for this task.
    #

    max_dimension = 1024

    height, width = image.shape[:2]

    largest_dimension = max(
        height,
        width
    )


    if largest_dimension > max_dimension:

        scale = (
            max_dimension /
            largest_dimension
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        image = cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )


    # ----------------------------------------------
    # BGR → RGB
    # ----------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # NumPy → PIL
    return Image.fromarray(
        image
    )


# ==================================================
# Prediction
# ==================================================

def predict_image(filepath):

    classifier = get_classifier()

    image = preprocess_image(
        filepath
    )


    predictions = classifier(
        image,
        candidate_labels=LABELS
    )


    results = []


    # Keep the strongest 10 matches
    for prediction in predictions[:10]:

        raw_label = prediction["label"]

        score = float(
            prediction["score"]
        )


        readable_label = LABEL_NAMES.get(
            raw_label,
            raw_label.title()
        )


        results.append({
            "label": readable_label,
            "confidence": round(
                score * 100,
                2
            )
        })


    if not results:

        return {
            "prediction":
                "No recognizable category found",
            "confidence": 0,
            "top_predictions": []
        }


    strongest = results[0]


    return {

        "prediction":
            strongest["label"],

        "confidence":
            strongest["confidence"],

        "top_predictions":
            results
    }