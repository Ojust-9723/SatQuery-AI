from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
import os
import shutil
import re
import traceback

from PIL import Image, ImageOps


# =========================================================
# LISA IMPORT
# =========================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "satellite-omni-lisa"
        )
    )
)

from inference import (
    load_model,
    generate_text,
    generate_segmentation
)


# =========================================================
# LISA MODEL
# =========================================================

LISA_MODEL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "satellite-omni-lisa"
    )
)

print("LISA model directory:", LISA_MODEL_DIR)

LISA = load_model(
    model_dir=LISA_MODEL_DIR,
    device="cpu"
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="SatQuery AI Backend",
    version="1.0"
)


# =========================================================
# UPLOAD FOLDER
# =========================================================

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# AGENT TASK CLASSIFICATION
# =========================================================

def classify_task(query):

    q = query.lower().strip()

    # -----------------------------------------------------
    # Segmentation
    # -----------------------------------------------------

    if any(
        word in q
        for word in [
            "segment",
            "segmentation",
            "mask",
            "outline"
        ]
    ):
        return "segmentation"

    # -----------------------------------------------------
    # Change Detection
    # -----------------------------------------------------

    if any(
        word in q
        for word in [
            "change",
            "changed",
            "difference",
            "before and after",
            "increase",
            "decrease",
            "new construction",
            "development"
        ]
    ):
        return "change_detection"

    # -----------------------------------------------------
    # Land / Water
    # -----------------------------------------------------

    if any(
        word in q
        for word in [
            "water",
            "river",
            "lake",
            "sea",
            "ocean",
            "coast",
            "land cover",
            "land coverage"
        ]
    ):
        return "land_water"

    # -----------------------------------------------------
    # Buildings
    # -----------------------------------------------------

    if any(
        word in q
        for word in [
            "building",
            "buildings",
            "structure",
            "structures",
            "object",
            "objects"
        ]
    ):
        return "building_detection"

    # -----------------------------------------------------
    # Scene Description
    # -----------------------------------------------------

    if any(
        word in q
        for word in [
            "describe",
            "description",
            "landscape",
            "scene",
            "environment",
            "surroundings",
            "what is in",
            "what do you see",
            "what can you see"
        ]
    ):
        return "scene_description"

    # -----------------------------------------------------
    # Default
    # -----------------------------------------------------

    return "vqa"


# =========================================================
# AGENT ANALYSIS FOCUS
# =========================================================

def get_analysis_focus(query, task):

    q = query.lower().strip()

    if task == "land_water":

        if any(
            word in q
            for word in [
                "water",
                "land"
            ]
        ):
            return (
                "Analyze both water and land features, "
                "including their locations and relationship"
            )

        return "Analyze water and land features"

    if task == "building_detection":

        return (
            "Identify and describe buildings or structures"
        )

    if task == "scene_description":

        return (
            "Describe the overall landscape, "
            "environment, and major visible features"
        )

    if task == "segmentation":

        return (
            "Identify and outline the requested "
            "objects or regions"
        )

    if task == "change_detection":

        return (
            "Identify and describe changes "
            "between the provided images"
        )

    return (
        "Answer the user's question using "
        "the satellite image"
    )


# =========================================================
# OPTIONAL GEO BOUNDS SUPPORT
# =========================================================

def extract_bounds(filename):

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        filename
    )

    if len(numbers) < 4:
        return None

    coords = list(
        map(
            float,
            numbers[-4:]
        )
    )

    return {
        "min_lon": coords[0],
        "min_lat": coords[1],
        "max_lon": coords[2],
        "max_lat": coords[3]
    }


def check_overlap(bounds1, bounds2):

    if not bounds1 or not bounds2:
        return False

    return (
        bounds1["min_lon"] < bounds2["max_lon"]
        and
        bounds1["max_lon"] > bounds2["min_lon"]
        and
        bounds1["min_lat"] < bounds2["max_lat"]
        and
        bounds1["max_lat"] > bounds2["min_lat"]
    )


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def prepare_image_for_lisa(
    original_path,
    processed_path,
    max_size=1024
):
    """
    Opens the uploaded image, converts it to RGB,
    and resizes it to a reasonable size for CPU inference.
    """

    print("Opening uploaded image...")

    image = Image.open(
        original_path
    )

    print(
        "Original image size:",
        image.size
    )

    print(
        "Original image format:",
        image.format
    )

    # Handle images with EXIF orientation
    image = ImageOps.exif_transpose(
        image
    )

    # Convert to RGB
    image = image.convert(
        "RGB"
    )

    # Resize while preserving aspect ratio
    image.thumbnail(
        (max_size, max_size),
        Image.Resampling.LANCZOS
    )

    print(
        "LISA input image size:",
        image.size
    )

    # Save standardized image
    image.save(
        processed_path,
        format="JPEG",
        quality=92
    )

    return image


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "SatQuery AI backend is running!"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "online",
        "model": "LISA",
        "device": "cpu"
    }


# =========================================================
# IMAGE UPLOAD + AI ANALYSIS
# =========================================================

@app.post("/upload")
async def upload_image(

    file: UploadFile = File(...),

    query: str = Form(...)
):

    original_path = None
    processed_path = None

    try:

        # =================================================
        # STEP 1 — VALIDATE QUERY
        # =================================================

        if not query.strip():

            raise HTTPException(
                status_code=400,
                detail="Please enter a question about the image."
            )

        # =================================================
        # STEP 2 — AGENT INTERPRETS QUERY
        # =================================================

        print("\n==============================")
        print("NEW SATQUERY REQUEST")
        print("==============================")

        print(
            "Filename:",
            file.filename
        )

        print(
            "Query:",
            query
        )

        task = classify_task(
            query
        )

        print(
            "Task:",
            task
        )

        # =================================================
        # STEP 3 — SAVE ORIGINAL IMAGE
        # =================================================

        safe_filename = os.path.basename(
            file.filename
        )

        original_path = os.path.join(
            UPLOAD_FOLDER,
            safe_filename
        )

        with open(
            original_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            "Image saved:",
            original_path
        )

        # =================================================
        # STEP 4 — PREPARE IMAGE FOR LISA
        # =================================================

        processed_filename = (
            "processed_" +
            os.path.splitext(
                safe_filename
            )[0] +
            ".jpg"
        )

        processed_path = os.path.join(
            UPLOAD_FOLDER,
            processed_filename
        )

        image = prepare_image_for_lisa(
            original_path,
            processed_path,
            max_size=1024
        )

        # =================================================
        # STEP 5 — DETERMINE ANALYSIS FOCUS
        # =================================================

        focus = get_analysis_focus(
            query,
            task
        )

        print(
            "Analysis focus:",
            focus
        )

        # =================================================
        # STEP 6 — LISA AI ANSWER
        # =================================================

        print("\nRunning LISA answer generation...")

        answer = generate_text(
            LISA,
            image,
            (
                f"{focus}. "
                f"User question: {query}"
            ),
            max_new_tokens=128
        )

        print(
            "LISA answer completed."
        )

        # =================================================
        # STEP 7 — VISUAL EVIDENCE
        # =================================================

        print(
            "Running LISA evidence generation..."
        )

        evidence = generate_text(
            LISA,
            image,
            (
                "Analyze the satellite image specifically "
                "to provide visual evidence for this question: "
                f"{query}. "
                "Explain what visible features in the image "
                "support the answer. "
                "Mention relevant regions or sides of the "
                "image when possible. "
                "Do not guess the city, country, sensor, "
                "or exact geographic location. "
                "Do not repeat the question. "
                "Give only the evidence."
            ),
            max_new_tokens=96
        )

        print(
            "LISA evidence completed."
        )

        # =================================================
        # STEP 8 — RETURN RESULT
        # =================================================

        print(
            "SatQuery AI analysis completed successfully."
        )

        return {

            "filename": file.filename,

            "query": query,

            "task": task,

            "answer": answer,

            "evidence": evidence,

            "agent_trace": [

                "Image received",

                "Query interpreted",

                f"Task identified: {task}",

                f"Analysis focus: {focus}",

                "Image normalized for AI inference",

                "LISA satellite VLM selected",

                "AI answer generated",

                "Visual evidence generated",

                "AI analysis completed"
            ]
        }

    except HTTPException:

        raise

    except Exception as error:

        print("\n==============================")
        print("SATQUERY AI ERROR")
        print("==============================")

        print(
            "Error:",
            str(error)
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "SatQuery AI could not analyze this image. "
                f"Error: {str(error)}"
            )
        )

    finally:

        # =================================================
        # CLEAN UP PROCESSED FILE
        # =================================================

        if processed_path and os.path.exists(
            processed_path
        ):

            try:

                os.remove(
                    processed_path
                )

            except Exception:

                pass


# =========================================================
# SIMPLE TEST ENDPOINT
# =========================================================

@app.get("/ask")
def ask(query: str):

    return {
        "answer": f"You asked: {query}"
    }