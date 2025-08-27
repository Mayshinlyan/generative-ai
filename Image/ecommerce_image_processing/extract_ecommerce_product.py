from IPython.display import display, Markdown
from PIL import Image as PIL_Image

from google import genai
from google.genai.types import (
    Image,
    RecontextImageSource,
    ProductImage,
    RecontextImageConfig
)

import matplotlib.image as img
import matplotlib.pyplot as plt
import logging, os

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================
#  Initializing the Project and Storage Client
# ============================================

def initialize_client():

    """Initializes a GenAI client with project and location settings."""

    logger.info(f"Initalizing GenAI client")

    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id"))
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

    # vertexai.init(project=PROJECT_ID, location=LOCATION)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    if not client._api_client.vertexai:
        logger.info(f"Using Gemini Developer API.")
    elif client._api_client.project:
        logger.info(
            f"Using Vertex AI with project: {client._api_client.project} in location: {client._api_client.location}"
        )
    elif client._api_client.api_key:
        logger.info(
            f"Using Vertex AI in express mode with API key: {client._api_client.api_key[:5]}...{client._api_client.api_key[-5:]}"
        )

    return client

client = initialize_client()


import google.auth
from google.cloud import storage

credentials, project_id = google.auth.default()
storage_client = storage.Client(credentials=credentials)


# ============================================
#  Helper function to display images
# ============================================

from google.cloud import storage
import io

storage_client = storage.Client()

def display_image(uri: str) -> PIL_Image.Image:
    """Display an image from a GCS URI."""
    if uri.startswith("gs://"):
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError("The gcs_uri must be in the format 'gs://bucket_name/blob_name'")
        bucket_name, blob_name = parts
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        image_data = blob.download_as_bytes()
        pil_image = PIL_Image.open(io.BytesIO(image_data))

    else:
        pil_image = PIL_Image.open(uri)

    return pil_image



def display_ref_and_gen_images(ref_image, gen_image) -> None:
    fig, axis = plt.subplots(1, 2, figsize=(12, 6))
    axis[0].imshow(ref_image)
    axis[0].set_title("Reference Image")
    axis[1].imshow(gen_image)
    axis[1].set_title("Generated Image")
    for ax in axis:
        ax.axis("off")
    plt.show()


# =================================================================
# Imagen 3: Subject Customization
# Useful for stylizing the subject: animal, human, product and not good at preseving details or extracting product
# =================================================================

customization_model = "imagen-3.0-capability-001"

from google.genai.types import (
    ControlReferenceConfig,
    ControlReferenceImage,
    EditImageConfig,
    GenerateImagesConfig,
    Image,
    RawReferenceImage,
    SubjectReferenceConfig,
    SubjectReferenceImage,
)

subject_image = Image(gcs_uri="gs://your-gcs-bucket/path-to-your-image.png")

subject_reference_image = SubjectReferenceImage(
    reference_id=1,
    reference_image=subject_image,
    config=SubjectReferenceConfig(
        subject_description="a jacket", subject_type="SUBJECT_TYPE_PRODUCT"
    ),
)

prompt = "Create a flat lay product shot of the jacket[1] to match the description: An ecommerce product image of the jacket [1] against pure white background. The jacket is neatly laid flat, perfectly centered, zipped up, and facing directly forward in full shot. Use soft, diffused studio lighting to ensure even illumination across the entire garment, eliminating all shadows and highlighting the fabric texture and details without any creases or wrinkles. The entire jacket must be in sharp focus. Exclude any models, mannequins, hangers, or props."

# prompt = "Generate the image of the jacket [1] but in orange color."

image = client.models.edit_image(
    model=customization_model,
    prompt=prompt,
    reference_images=[subject_reference_image], #control_reference_image],
    config=EditImageConfig(
        edit_mode="EDIT_MODE_DEFAULT",
        number_of_images=1,
        safety_filter_level="BLOCK_MEDIUM_AND_ABOVE"
    ),
)

image_ref = display_image("gs://your-gcs-bucket/path-to-your-image.png")

display_ref_and_gen_images(image_ref, image.generated_images[0].image._pil_image)



# =================================================================
#  Imagen 3: Style Transfer
# =================================================================

image = Image(gcs_uri="gs://your-gcs-bucket/path-to-your-image.png")

raw_ref_image = RawReferenceImage(reference_image=image, reference_id=1)

prompt = "transform the subject in the image so that the jacket[1] is made entirely out of white faux fur"

style_image = client.models.edit_image(
    model=customization_model,
    prompt=prompt,
    reference_images=[raw_ref_image],
    config=EditImageConfig(
        edit_mode="EDIT_MODE_DEFAULT",
        number_of_images=1,
        safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
        person_generation="ALLOW_ADULT",
    ),
)

image_ref = display_image("gs://your-gcs-bucket/path-to-your-image.png")

display_ref_and_gen_images(image_ref, style_image.generated_images[0].image._pil_image)

# =================================================================
# Imagen 3: Control Customization
# Useful for producing images that adhere to the outline of the Canny Image or Scribbles
# =================================================================
import cv2

generation_prompt = """
a jacket on a white background. Don't include hangers or mannequins. High detail, professional photography
"""
generated_image = client.models.generate_images(
    model=generation_model,
    prompt=generation_prompt,
    config=GenerateImagesConfig(
        number_of_images=1,
        aspect_ratio="1:1",
        safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
        person_generation="DONT_ALLOW",
    ),
)

generated_image.generated_images[0].image.save("./data/jacket.png")
img = cv2.imread("./data/jacket.png")
display_image("./data/jacket.png")


# Setting parameter values
t_lower = 100  # Lower Threshold
t_upper = 150  # Upper threshold

# Applying the Canny Edge filter
edge = cv2.Canny(img, t_lower, t_upper)
cv2.imwrite("./data/jacket_edge.png", edge)

control_image = ControlReferenceImage(
    reference_id=1,
    reference_image=Image.from_file(location="./data/jacket_edge.png"),
    config=ControlReferenceConfig(control_type="CONTROL_TYPE_CANNY"),
)

edit_prompt = "A photorealistic image along the lines of a leather jacket in a closet hanging on wooden hanger."

control_image = client.models.edit_image(
    model=customization_model,
    prompt=edit_prompt,
    reference_images=[control_image],
    config=EditImageConfig(
        edit_mode="EDIT_MODE_CONTROLLED_EDITING",
        number_of_images=1,
        safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
        person_generation="ALLOW_ADULT",
    ),
)


fig, axis = plt.subplots(1, 3, figsize=(12, 6))
axis[0].imshow(generated_image.generated_images[0].image._pil_image)
axis[0].set_title("Original Image")
axis[1].imshow(edge, cmap="gray")
axis[1].set_title("Canny Edge")
axis[2].imshow(control_image.generated_images[0].image._pil_image)
axis[2].set_title("Edited Image")
for ax in axis:
    ax.axis("off")
plt.show()



# =================================================================
#  Imagen 3: Product Recontext
# =================================================================

product_recontext = "imagen-product-recontext-preview-06-30"

prompt="A high-resolution product photograph of the jacket. The jacket should be laid flat, perfectly centered, and facing forward against a pure white background. The image needs soft, even studio lighting and a sharp focus on the entire garment. Exclude any models, mannequins, hangers, or props."

image_1 = "./data/men jacket/1.png"
image_2 = "./data/men jacket/3.png"
image_3 = "./data/men jacket/5.png"

# image_1 = "./data/women jacket/2.png"
# image_2 = "./data/women jacket/3.png"
# image_3 = "./data/women jacket/4.png"

display_image(image_1)

response = client.models.recontext_image(
    model=product_recontext,
    source=RecontextImageSource(
        prompt=prompt,
        product_images=[
            ProductImage(product_image=Image.from_file(location=image_1)),
            ProductImage(product_image=Image.from_file(location=image_2)),
            ProductImage(product_image=Image.from_file(location=image_3))
        ]
    ),
    config=RecontextImageConfig(
        base_steps=32,
        number_of_images=4,
        safety_filter_level="BLOCK_LOW_AND_ABOVE",
        enhance_prompt=False,
        # output_gcs_uri="gs://your-gcs-bucket"
    )
)



fig, axes = plt.subplots(2, 2, figsize=(48, 48))
for i, ax in enumerate(axes.flatten()):
    image = response.generated_images[i].image._pil_image
    ax.imshow(image)
    ax.axis("off")
plt.show()

# =================================================================
#  Gemini 2.5 Flash Image Generation
# =================================================================
from google.genai.types import GenerateContentConfig, Part

MODEL_ID = "gemini-2.5-flash-image-preview"

subject_image = "./data/men jacket/1.png"

with open(subject_image, "rb") as f:
    image = f.read()

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        Part.from_bytes(
            data=image,
            mime_type="image/jpeg",
        ),
        prompt,
    ],
    config=GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        candidate_count=1,
        safety_settings=[
            {"method": "PROBABILITY"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT"},
            {"threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ],
    ),
)

for part in response.candidates[0].content.parts:
    if part.text:
        display(Markdown(part.text))
    if part.inline_data:
        display(Image(data=part.inline_data.data, width=400))