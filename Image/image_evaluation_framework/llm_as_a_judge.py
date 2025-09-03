import json
import logging
import os
from typing import Literal

from google import genai
from google.genai import types
from langfuse import Langfuse, observe
from pydantic import BaseModel, Field

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


class llm_evaluation_response(BaseModel):
    """LLM as a judge evaluation response schema."""

    llm_eval_outcome: Literal["Good", "Bad"] = Field(
        description="LLM evaluation response: Good, or Bad"
    )
    llm_eval_critique: str = Field(
        description="Brief reasoning of the llm_eval_outcome. Why does the LLM decide good or bad?"
    )


# =================================================================
#  Registering LLM as a Judge Prompt in Langfuse
# =================================================================

# Initialize LangFuse
langfuse = Langfuse()

langfuse.create_prompt(
    name="LLM as a Judge Prompt",
    prompt=f"""You are ecommerce images evaluator with advanced capabilities to judge if a {{product}} in an image matches the {{product}} in another image.

    You understand what a flat lay studio look of the {{product}} should look like, and you can compare it with the input images to see if they match. Given the product name and the input images, evaluate if the final studio image matches the product from the input images. Respond with 'Good' if it matches, otherwise respond with 'Bad'. Provide a reasoning for your decision.

    Here are some guidelines for evaluating if the studio image matches the product from the input images:
    1. Check if the main features of the product are present in the studio image.
    2. Look for any discrepancies in color, shape, or design.
    3. Make sure the entire {{product}} is visible and not cropped out.
    4. Make sure the logo or branding is clearly visible and matches the input images.
    5. Make sure there is no human model in the studio image.
    6. Evaluate the overall quality and clarity of the studio image.

    What is your evaluation for the fifth image of the {{product}}? It's input image is the fourth image named input_to_evaluate.

   """,
    config={"model": "gemini-2.5-flash", "base_steps": 32, "number_of_images": 1},
    labels=["development"],
)

prompt = langfuse.get_prompt("LLM as a Judge Prompt", label="development")


# =================================================================
#  Functions
# =================================================================


def import_images(image_paths: list[str]) -> list:
    """method to import local images for llm evaluation

    Args:
        image_paths (list[str]): list of local image file paths
    """
    image_bytes_list = []

    for f in image_paths:
        with open(f, "rb") as f:
            image_bytes_list.append(f.read())

    print(f"Read {len(image_bytes_list)} image(s).")
    return image_bytes_list


@observe
def evaluate(
    product_name: str, input_image: list[bytes], output_image: list[bytes]
) -> llm_evaluation_response:

    few_shots_image_paths = [
        "./data/men jacket/1.png",
        "./data/men jacket/good_output.png",
        "./data/men jacket/okay_output.png",
    ]

    few_shots_image_byte_list = import_images(few_shots_image_paths)

    compiled_prompt = prompt.compile(product=product_name)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=compiled_prompt,
            response_mime_type="application/json",
            response_schema=llm_evaluation_response,
        ),
        contents=[
            # Examples for few-shot prompting
            types.Part.from_bytes(
                data=few_shots_image_byte_list[0],
                mime_type="image/png",
            ),
            types.Part.from_bytes(
                data=few_shots_image_byte_list[1],
                mime_type="image/png",
            ),
            types.Part.from_bytes(
                data=few_shots_image_byte_list[2],
                mime_type="image/png",
            ),
            # Actual input and output to evaluate
            types.Part.from_bytes(
                data=input_image[0],
                mime_type="image/png",
            ),
            types.Part.from_bytes(
                data=output_image[0],
                mime_type="image/png",
            ),
            """
            Example evaluations:

            <examples>
            <example-1>
            <product>Men Jacket</product>
            <input_images>
                First image named few_shots_image_byte_list[0] is the input image
            </input_images>
            <studio_image>
                Second image named few_shots_image_byte_list[1] is the output image
            </studio_image>
            <critique>
            {
            "critique": "The studio image accurately represents the jacket shown in the input images. The color, design, and branding match perfectly. The entire product is visible, it is front facing and there is no human model present. The image quality is high, making it easy to see the details of the jacket.",
            "outcome": "Good"
            }
            </critique>
            </example-1>

            <example-2>
            <product>Men Jacket</product>
            <input_images>
                First image named few_shots_image_byte_list[0] is the input image
            </input_images>
            <studio_image>
                Third image named few_shots_image_byte_list[2] is the output image
            </studio_image>
            <critique>
            {
            "critique": "The studio image although correctly represents the jacket with correct color, and branding. However, it has the hood which is not present in the input images. The image quality is good, but the presence of the hood makes it less accurate compared to the input images.",
            "outcome": "Bad"
            }
            </critique>
            </example-2>

            """,
        ],
    )

    return response.parsed


def load_responses(filename: str):
    """Load responses from the JSON file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Could not read product_recontext_events.json.")
        return []


import pandas as pd


def convert_json_to_csv(json_file_path: str, csv_file_path: str):
    try:
        # Step 1: Read the JSON file into a pandas DataFrame
        df = pd.read_json(json_file_path)

        # Step 2: Write the DataFrame to a CSV file
        # Setting index=False prevents pandas from writing the DataFrame index as a column
        df.to_csv(csv_file_path, index=False)

        print(f"Successfully converted '{json_file_path}' to '{csv_file_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except ValueError as e:
        print(
            f"Error reading JSON file: {e}. The JSON structure might be complex or invalid."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------

if __name__ == "__main__":

    client = initialize_client()

    # Load events to evaluate
    responses_list = load_responses("product_recontext_events.json")

    for event in responses_list:

        # Load the input and output images
        input_image_bytes_list = import_images(event["input_images"])
        output_image_bytes_list = import_images(event["output_images"])

        # LLM as a Judge
        llm_eval = evaluate(
            event["product_name"], input_image_bytes_list, output_image_bytes_list
        )
        logger.info(f"The LLM Eval results: {llm_eval}")

        event["llm_eval_outcome"] = llm_eval.llm_eval_outcome
        event["llm_eval_critique"] = llm_eval.llm_eval_critique

    # Update the LLM as a Judge response in the genai_response.json file
    with open("product_recontext_events.json", "w") as json_file:
        json.dump(responses_list, json_file, indent=4)
        print(
            f"New llm eval response successfully appended to product_recontext_events.json"
        )
