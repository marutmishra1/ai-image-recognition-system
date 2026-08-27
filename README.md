# AI Image Recognition System

An AI-powered image recognition web application built with Python, Flask,
Hugging Face Transformers, PyTorch, CLIP, OpenCV, Pillow, and NumPy.

The application allows users to upload an image and uses zero-shot image
classification to determine which predefined visual categories best match
the uploaded image.

---

## Overview

This project demonstrates how computer vision, AI models, image processing,
and Flask APIs can be combined into an interactive image-recognition
application.

Instead of using a traditional classifier trained only on a fixed dataset,
the application uses CLIP-based zero-shot image classification. The uploaded
image is compared against a collection of candidate text descriptions, and
the system ranks the categories according to their visual similarity.

This makes it possible to recognize a broad range of categories without
training a separate model for every category.

---

## Technologies

- Python
- Flask
- Hugging Face Transformers
- CLIP
- PyTorch
- OpenCV
- Pillow
- NumPy
- HTML
- CSS
- JavaScript

---

## Key Features

- Image upload through a web interface
- JPG, JPEG, PNG, and WEBP support
- AI-powered zero-shot image classification
- CLIP-based visual recognition
- Custom candidate categories
- Document recognition
- Education-related recognition
- Biotechnology and laboratory categories
- Person and professional categories
- Common object recognition
- Scene recognition
- Fictional and comic-book-style visual categories
- Top prediction
- Confidence score
- Top 10 matching categories
- Browser image preview
- REST prediction endpoint
- Health-check endpoint
- Temporary image storage
- Automatic cleanup of uploaded files
- Large-image preprocessing and resizing
- Cached model loading for faster subsequent predictions

---

## Recognition Categories

### Documents and Education

- Resume / CV
- Student Profile
- Educational Document
- Certificate
- Identification Document
- Business Document
- Handwritten Document
- Printed Text Document
- Text Document
- Book
- Textbook
- Presentation Slide
- Computer Screen

### Science and Biotechnology

- Biotechnology Laboratory
- Biotechnology Experiment
- Scientific Laboratory
- Scientist / Laboratory
- Science Classroom
- Biology Experiment
- Medical Laboratory
- Laboratory

### People

- Person
- Student
- Teacher
- Scientist
- Professional

### Common Objects

- Laptop
- Mobile Phone
- Car
- Bicycle
- Dog
- Cat
- Bird
- Apple
- Bottle
- Chair
- Television

### Fictional and Visual Categories

- Superhero-Style Character
- Comic-Book-Style Character
- Fictional Character
- Fantasy Character
- Animated Character
- Action Hero Character

### Scenes

- City Street
- Classroom
- Office
- Laboratory
- Hospital
- Park
- Beach
- Landscape

---

## Application Architecture

```text
                  User
                   |
                   v
             Image Upload
                   |
                   v
             Flask Backend
                   |
                   v
            File Validation
                   |
                   v
              OpenCV
                   |
                   v
           Image Preprocessing
                   |
                   v
        CLIP Zero-Shot Classification
                   |
                   v
       Compare Image With Candidate Labels
                   |
                   v
          Rank Matching Categories
                   |
                   v
        Confidence + Top 10 Results
                   |
                   v
            Web Interface