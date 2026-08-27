from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from werkzeug.utils import secure_filename

from pathlib import Path

import uuid

from image_model import predict_image


app = Flask(__name__)


# --------------------------------------------------
# Project directories
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = (
    BASE_DIR / "uploads"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Allowed image types
# --------------------------------------------------

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# --------------------------------------------------
# Maximum upload size
# --------------------------------------------------

app.config[
    "MAX_CONTENT_LENGTH"
] = 10 * 1024 * 1024


# --------------------------------------------------
# File validation
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# Image prediction
# --------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return jsonify({
            "error":
                "No image was uploaded."
        }), 400


    image = request.files["image"]


    if image.filename == "":

        return jsonify({
            "error":
                "Please select an image."
        }), 400


    if not allowed_file(
        image.filename
    ):

        return jsonify({
            "error":
                "Only JPG, JPEG, PNG, "
                "and WEBP images are allowed."
        }), 400


    # --------------------------------------------------
    # Create unique temporary filename
    # --------------------------------------------------

    extension = (
        image.filename
        .rsplit(".", 1)[1]
        .lower()
    )


    filename = (
        f"{uuid.uuid4().hex}"
        f".{extension}"
    )


    filepath = (
        UPLOAD_FOLDER
        /
        secure_filename(filename)
    )


    image.save(filepath)


    try:

        result = predict_image(
            str(filepath)
        )

        return jsonify(result)


    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return jsonify({
            "error":
                f"Prediction failed: {error}"
        }), 500


    finally:

        # Remove temporary uploaded image
        if filepath.exists():

            filepath.unlink()


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "healthy",
        "service":
            "AI Image Recognition System"
    })


# --------------------------------------------------
# Application entry point
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )