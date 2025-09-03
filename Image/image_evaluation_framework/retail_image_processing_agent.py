import base64
import io
import json
import logging
import os
import uuid

import google.auth
import matplotlib.image as img
import matplotlib.pyplot as plt
from google import genai
from google.cloud import storage
from google.genai.types import (
    Image,
    ProductImage,
    RecontextImageConfig,
    RecontextImageSource,
)
from IPython.display import Markdown, display
from langfuse import Langfuse, observe
from PIL import Image as PIL_Image

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

    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT", "maylyan-test"))
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


import urllib.parse

# =================================================================
#  Create Test Dataset
# =================================================================
import requests
from bs4 import BeautifulSoup


def download_specific_images(url, output_folder="downloaded_images"):
    """
    Crawls a given URL, finds images within specified classes, and downloads them.

    Args:
        url (str): The URL of the webpage to crawl.
        output_folder (str): The name of the folder to save the images in.
                             Defaults to 'downloaded_images'.
    """
    try:
        logger.info(f"Accessing webpage: {url}")
        response = requests.get(
            "https://www.johnlewis.com/and-or-blouson-pure-suede-jacket-tan/p113753410"
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Created directory: {output_folder}")

        # Find all elements with the specified classes
        container_elements = soup.find_all(
            class_=["Carousel_galleryItem__7ii3O Carousel_galleryItemPeek__1PpAo"]
        )

        if not container_elements:
            print("No containers with the specified classes found on this page.")
            return

        image_urls = []
        for container in container_elements:
            # Find all <img> tags within each container
            img_tags = container.find_all("img")
            for img in img_tags:
                img_url = img.get("src")
                if img_url:
                    # Handle relative URLs
                    if not img_url.startswith(("http", "https")):
                        img_url = urllib.parse.urljoin("https:", img_url)
                    image_urls.append(img_url)

        if not image_urls:
            print("No images found inside the specified containers.")
            return

        print(f"Found {len(image_urls)} images. Starting download...")

        for img_url in set(image_urls):
            img_name = os.path.basename(urllib.parse.urlparse(img_url).path)

            if not img_name:
                continue

            img_path = os.path.join(output_folder, img_name)

            # Check if the file already exists to avoid re-downloading
            if os.path.exists(img_path):
                print(f"Skipping {img_name} as it already exists.")
                continue

            try:
                img_data = requests.get(img_url)
                img_data.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(img_data.content)
                print(f"Downloaded: {img_name}")
            except requests.exceptions.RequestException as e:
                print(f"Could not download {img_url}: {e}")
            except IOError as e:
                print(f"Could not save file {img_name}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error accessing the URL {url}: {e}")


# # Test
# webpage_url = "https://www.johnlewis.com/and-or-blouson-pure-suede-jacket-tan/p113753410"
# download_specific_images(webpage_url)


def download_image(urlList: list[str], save_path: str) -> None:
    """Download an image from a URL and save it locally."""
    logger.info(f"Downloading image from {url} to {save_path}")

    for url in urlList:

        image = img.imread(url)
        img.imsave(save_path, image)


def save_image_locally(
    image_data: bytes, output_file_name: str, output_directory: str
) -> list[str]:
    """Save Imagen generated image to local directory"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = os.path.join(output_directory, output_file_name)

    try:
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"Image successfully saved to {output_path}")

    except IOError as e:
        print(f"Error saving image: {e}")

    return [output_path]


def load_json(filename: str):
    """Load responses from the JSON file."""
    try:
        with open(filename, "r") as f:
            print("Loading JSON File")
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Could not read product_recontext_events.json.")
        return []


# ============================================
#  Helper function to display images
# ============================================


def display_image(uri: str) -> PIL_Image.Image:
    """Display an image from a GCS URI."""
    if uri.startswith("gs://"):
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                "The gcs_uri must be in the format 'gs://bucket_name/blob_name'"
            )
        bucket_name, blob_name = parts
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        image_data = blob.download_as_bytes()
        pil_image = PIL_Image.open(io.BytesIO(image_data))

    else:
        pil_image = PIL_Image.open(uri)

    return pil_image


# =================================================================
#  Registering Image Processing Agent Prompt in Langfuse
# =================================================================

# Initialize LangFuse
langfuse = Langfuse()

product_recontext = "imagen-product-recontext-preview-06-30"
langfuse.create_prompt(
    name="Product Recontext Prompt",
    prompt="A high-resolution product photograph of the {{product}}. The {{product}} should be laid flat, perfectly centered, and facing forward against a pure white background. The image needs soft, even studio lighting and a sharp focus on the entire {{product}}. Exclude any models, mannequins, hangers, human body part or props.",
    config={"model": product_recontext, "base_steps": 32, "number_of_images": 1},
    labels=["development"],
)

prompt = langfuse.get_prompt("Product Recontext Prompt", label="development")


# =================================================================
#  Main function to extract the product
# =================================================================


@observe()
def extract_product(input_images: list[str], product_name: str, product_url: str):
    """Extract and create studio product image from user provided images."""

    logger.info(f"Extracting {product_name} from user images.")

    test_valid_image_filepath(input_images)
    test_input_image_length(input_images)
    test_prompt_injection(product_name)
    test_prompt_injection(product_url)

    compiled_prompt = prompt.compile(product=product_name)

    product_image_list = [
        ProductImage(product_image=Image.from_file(location=image_location))
        for image_location in input_images
    ]

    response = client.models.recontext_image(
        model=prompt.config["model"],
        source=RecontextImageSource(
            prompt=compiled_prompt, product_images=product_image_list
        ),
        config=RecontextImageConfig(
            base_steps=prompt.config["base_steps"],
            number_of_images=prompt.config["number_of_images"],
            safety_filter_level="BLOCK_LOW_AND_ABOVE",
            enhance_prompt=False,
        ),
    )

    test_output_image_length(response)

    if response:

        # Save output image locally
        image_data = response.generated_images[0].image.image_bytes
        output_image_path = save_image_locally(
            image_data, f"{product_name}-output.png", f"./data/{product_name}"
        )

        # Create a dict obj for event response
        response_dict = {
            "trace_id": str(uuid.uuid4()),  # replace this with langfuse trace id later,
            "input_prompt": prompt.compile(product=product_name),
            "product_name": product_name,
            "input_images": input_images,
            "output_images": output_image_path,
            "product_url": product_url,
            "metadata": "some_value",
        }
    else:
        logger.error("No response generated")
        response_dict = {}

    # Update main observation with overall metrics
    langfuse.update_current_trace(
        name="product_image_extraction_pipeline",
        tags=["imagen_product_recontext", "product_studio_photograph"],
    )
    # Add metadata to LangFuse
    langfuse.update_current_span(
        metadata={
            "base_step": prompt.config["base_steps"],
            "model": prompt.config["model"],
        },
        input={"input_images": input_images, "product_name": product_name},
        output={
            "output_images": base64.b64encode(
                response.generated_images[0].image.image_bytes
            ).decode("utf-8"),
            "response": response,
        },
    )

    langfuse.flush()

    return response_dict


# --------------------------------------------------------------
# Store the response in a JSON file
# --------------------------------------------------------------


def store_product_recontext_response(response) -> None:
    """Store the response in a JSON file."""
    logger.info(f"Storing response in JSON file.")

    file_name = "product_recontext_events.json"
    responses_list = []

    try:
        with open(file_name, "r") as json_file:
            responses_list = json.load(json_file)

    except (FileNotFoundError, json.JSONDecodeError):
        with open(file_name, "w") as json_file:
            json.dump(responses_list, json_file, indent=4)

    responses_list.append(response)

    with open(file_name, "w") as json_file:
        json.dump(responses_list, json_file, indent=4)
        print(f"New response successfully appended to {file_name}")


# fig, axes = plt.subplots(2, 2, figsize=(48, 48))
# for i, ax in enumerate(axes.flatten()):
#     image = response.generated_images[i].image._pil_image
#     ax.imshow(image)
#     ax.axis("off")
# plt.show()


# --------------------------------------------------------------
# Individual test functions
# --------------------------------------------------------------
def test_valid_image_filepath(input_images: list[str]) -> None:
    """Test if the image filepath are valid or not"""
    try:
        for file_path in input_images:
            assert os.path.exists(file_path), f"File not found at: {file_path}"
    except AssertionError as e:
        print(f"❌ input image file not found")
        raise e


def test_input_image_length(input_images: list[str]) -> None:
    """Test if the input image is at least 3 images. Not more or less."""
    try:
        assert len(input_images) == 3
    except AssertionError as e:
        print(f"❌ test_input_image_length: Expected 3 images, got {len(input_images)}")
        raise e


def test_prompt_injection(user_input: str) -> None:
    """Check for prompt injection in the product name."""

    # Define a list of forbidden phrases or keywords
    forbidden_phrases = [
        "ignore previous instructions",
        "act as a",
        "forget everything",
    ]

    try:
        for phrase in forbidden_phrases:
            assert (
                phrase not in user_input.lower()
            ), f"Prompt injection detected: '{phrase}'"

    except AssertionError as e:
        print(f"❌ Prompt injection detected'")
        raise e


def test_output_image_length(response) -> None:
    try:
        assert len(response.generated_images) >= 1
    except AssertionError as e:
        print(
            f"❌ test_output_image_length: Expected at least 1 generated image, got {len(response.generated_images)}"
        )
        raise e


if __name__ == "__main__":

    # initialize GenAI client
    client = initialize_client()

    # initialize storage client
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(credentials=credentials)

    # load images to process (later to be replaced by frontend)
    data = load_json("./data/images_to_process.json")

    for item in data:
        response = extract_product(
            item["user_input_images"], item["product_name"], item["product_url"]
        )
        store_product_recontext_response(response)

    # # unit tests
    # tests = [
    #     test_input_image_length,
    #     test_input_product_name,
    #     test_output_image_length,
    # ]
    # passed = 0

    # for test in tests:
    #     try:
    #         test()
    #         print(f"✅ {test.__name__}")
    #         passed += 1
    #     except AssertionError as e:
    #         print(f"❌ {test.__name__}")

    # print(f"\nResults: {passed}/{len(tests)} tests passed")
