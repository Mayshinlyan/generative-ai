import json

import streamlit as st
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(page_title="Image Evaluation Framework", layout="wide")
st.title("Image Evaluation Framework")
st.write("Simple site to evaluate llm image outputs with human feedback.")

if "current_index" not in st.session_state:
    st.session_state.current_index = 0


# --------------------------------------------------------------
# Functions
# --------------------------------------------------------------


def load_responses(filename: str):
    """Load responses from the JSON file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Could not read product_recontext_events.json.")
        return []


responses_list = load_responses("product_recontext_events.json")


def update_evaluation_response(trace_id: str, rating: str, reason: str):
    """Update the evaluation response in the genai_response.json list."""

    print(f"Updating evaluation response... for button click with rating, {rating}")

    for item in responses_list:
        if item["trace_id"] == trace_id:
            item["human_eval_outcome"] = rating
            item["human_eval_critique"] = reason
            break

    with open("product_recontext_events.json", "w") as json_file:
        json.dump(responses_list, json_file, indent=4)
        print(f"New response successfully appended to product_recontext_events.json")

    next_event()
    st.rerun()


def next_event():
    """Move to the next image in the list."""
    if st.session_state.current_index < len(responses_list) - 1:
        print("hiii" + str(st.session_state.current_index))
        st.session_state.current_index += 1
        print("hiii" + str(st.session_state.current_index))
    elif st.session_state.current_index == len(responses_list) - 1:
        done_dialog()


@st.dialog("Completed")
def done_dialog():
    st.write(f"All evaluation completed. Thank you for you time!")


def prev_event():
    """Move to the previous image in the list."""
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1


# --------------------------------------------------------------
# App Layout
# --------------------------------------------------------------

if not responses_list:
    st.info("No responses to display. Please generate some images first.")
else:

    current_event = responses_list[st.session_state.current_index]
    print(f"current event: {current_event}")
    trace_id = current_event["trace_id"]

    st.markdown(
        f"## Does the final studio image of the *{current_event['product_name']}* matches the product from the input images?"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Input Images")
        image_container = st.container(horizontal=True, horizontal_alignment="center")
        with image_container:
            for img in current_event["input_images"]:
                st.image(width=400, image=img)

    with col2:
        st.markdown("### Output Images")
        image_container = st.container(horizontal=True, horizontal_alignment="center")
        with image_container:
            # output_image = current_event["output_image"]["base64_image"]
            # mime_type = current_event["output_image"]["mime_type"]
            # data_url = f"data:image/{mime_type};base64,{output_image}"
            # st.image(width=400, image=data_url)
            for img in current_event["output_images"]:
                st.image(width=400, image=img)

    # --------------------------------------------------------------
    # Buttons for feedback and navigation
    # --------------------------------------------------------------

    @st.dialog("Please share your reasoning:")
    def comment_dialog(trace_id: str, rating: str):
        st.write(f"Reasoning for rating: {rating}")
        reason = st.text_input("Evaluation Comment:")
        if st.button("Submit"):
            update_evaluation_response(trace_id, rating, reason)

    col1, col2, col3, col4 = st.columns([3, 1, 2, 2])

    button_container = st.container(horizontal=True, horizontal_alignment="center")

    with button_container:

        with col2:
            st.subheader("Rate the output image")

            button_container_one = st.container(
                horizontal=True, horizontal_alignment="center"
            )

            with button_container_one:
                with stylable_container(
                    "accept-btn",
                    css_styles="""
                    button {
                        background-color: green;
                        color: white;
                    }""",
                ):
                    st.button("Good", on_click=lambda: comment_dialog(trace_id, "Good"))

                with stylable_container(
                    "reject-btn",
                    css_styles="""
                    button {
                        background-color: red;
                        color: white;
                    }""",
                ):
                    st.button("Bad", on_click=lambda: comment_dialog(trace_id, "Bad"))

        with col3:
            st.subheader("Navigation")
            button_container_two = st.container(
                horizontal=True, horizontal_alignment="center"
            )

            with button_container_two:
                with stylable_container(
                    "back-btn",
                    css_styles="""
                    button {
                        background-color: grey;
                        color: white;
                    }""",
                ):

                    st.button("Back", on_click=lambda: prev_event())

                with stylable_container(
                    "reset-btn",
                    css_styles="""
                    button {
                        background-color: orange;
                        color: black;
                    }""",
                ):
                    st.button("Reset")

                with stylable_container(
                    "next-btn",
                    css_styles="""
                    button {
                        background-color: grey;
                        color: white;
                    }""",
                ):
                    st.button("Next", on_click=lambda: next_event())
