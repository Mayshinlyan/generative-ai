# -------------------------------------------------------------
# Config setup
# -------------------------------------------------------------

from dotenv import load_dotenv
import os, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Update the .env path if needed
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# -------------------------------------------------------------
# Instantiate the model
# -------------------------------------------------------------

from langchain_google_vertexai import ChatVertexAI

# Add your prefer model name here
llm = ChatVertexAI(model="gemini-2.0-flash-001")

# -------------------------------------------------------------
# Define tools / function
# -------------------------------------------------------------

import base64
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def load_image(question:str, imagepath: str):
    """Detect objects in the image.

    Args:
        question: question that the user ask about the object
        imagepath: the file path of the image
    """

    with open(imagepath, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    messages = [HumanMessage(content=[
                {"type": "text", "text": f"{question} Also explain your reasoning"},
                {"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{image_data}"}}
            ])]

    response = llm.invoke(messages)

    return response.content

def load_audio(question:str, filepath: str):
    """Detect objects in the audio file.

    Args:
        question: question that the user ask about the object
        filepath: the file path of the audio file
    """

    with open(filepath, 'rb') as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")

    # Create a message
    messages = [HumanMessage(content=[
                {"type": "text", "text": f"{question} Also explain your reasoning"},
                {"type": "media", "mime_type": "audio/mpeg", "data": audio_data}
            ])]

    response = llm.invoke(messages)

    return response.content


def load_video(question:str, filepath: str):
    """Detect objects in the video file.

    Args:
        question: question that the user ask about the object
        filepath: the file path of the video file
    """

    with open(filepath, 'rb') as f:
       video_data = base64.b64encode(f.read()).decode("utf-8")

    # Create a message
    messages = [HumanMessage(content=[
                {"type": "text", "text": f"{question} Also explain your reasoning"},
                {"type": "media", "mime_type": "video/mp4", "data": video_data}
            ])]

    response = llm.invoke(messages)

    return response.content

# -------------------------------------------------------------
# Define the Supervisor Node
# -------------------------------------------------------------

from typing import Literal
from typing_extensions import TypedDict
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph, START, END

# Create a list of agents for routing
members = ["image_agent", "audio_agent", "video_agent"]
options = members + ["FINISH"]

system_prompt =  (
    f"""You are a supervisor tasked with managing a conversation between the following workers: {members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status."""
)

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


class State(MessagesState):
    """State with next variable for routing and completed_agents to track which agents have responded."""
    next: str
    completed_agents: set = set()


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    """
    Supervisor node that routes to the next agent based on the response.

    Returns:
        Command: command to update the state and route to the next agent.
    """

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)

    # Check the messages to see which agents have already responded
    goto = response["next"]
    combined_response = []
    completed_agents = set()

    for message in state.get("messages", []):
        if isinstance(message, AIMessage) and message.name:
            completed_agents.add(message.name)
            combined_response.append(message)

    # Summarize the responses from all agents when all agents have responded
    if len(list(completed_agents)) == len(members):
        human_message = ""
        for message in state.get("messages", []):
            if isinstance(message, HumanMessage):
                human_message = message.content


        summarize_prompt = (f"you are decision agent. summarize based on all three response from the agents"
                            f"and return the final decision for the input prompt question {human_message}")

        messages = ([SystemMessage(content=summarize_prompt)]
                    + combined_response)

        decision_response = llm.invoke(messages)
        return Command(
            update={
                "messages": [
                    decision_response
                ]
        },
            goto=END,
        )

    # ensure all agent is called only once
    elif goto in completed_agents:
        remainders = set(members) - completed_agents
        if remainders:
            goto = list(remainders)[0]
        else:
            goto = END

    return Command(goto=goto, update={"next": goto, "completed_agents": completed_agents})

# -------------------------------------------------------------
# Define the Worker Nodes
# -------------------------------------------------------------

from langgraph.prebuilt import create_react_agent

image_agent = create_react_agent(
    llm, tools=[load_image], prompt="You are image_agent and an expert at detecting objects in images. Only provide answer to the image file. You must use a tool in every response. Do not answer directly. Always reason step by step and invoke a tool"
)

def image_node(state: State) -> Command[Literal["supervisor"]]:

    result = image_agent.invoke(state)

    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="image_agent")
            ]
        },
        goto="supervisor",
    )

audio_agent = create_react_agent(
    llm, tools=[load_audio], prompt="You are audio_agent and an expert at detecting objects in audio files. Only provide answer to the audio file. You must use a tool in every response. Do not answer directly. Always reason step by step and invoke a tool"
)

def audio_node(state: State) -> Command[Literal["supervisor"]]:

    result = audio_agent.invoke(state)

    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="audio_agent")
            ]
        },
        goto="supervisor",
    )

video_agent = create_react_agent(
    llm, tools=[load_video], prompt="You are video_agent and an expert at detecting objects in videos. Only provide answer to the video file. You must use a tool in every response. Do not answer directly. Always reason step by step and invoke a tool"
)

def video_node(state: State) -> Command[Literal["supervisor"]]:

    result = video_agent.invoke(state)

    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="video_agent")
            ]
        },
        goto="supervisor",
    )

# -------------------------------------------------------------
# Build Graph
# -------------------------------------------------------------

from IPython.display import Image, display

builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("image_agent", image_node)
builder.add_node("audio_agent", audio_node)
builder.add_node("video_agent", video_node)
builder.add_edge(START, "supervisor")
graph = builder.compile()

# see the current graph structure
display(Image(graph.get_graph().draw_mermaid_png()))

# build the graph
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)


# -------------------------------------------------------------
# Tests
# -------------------------------------------------------------

# Positive dog result
config = {"configurable": {"thread_id": "1"}, "subgraph": True}
messages = react_graph_memory.invoke(
    {"messages": [HumanMessage(content=[
            {"type": "text", "text": "Question: Can you see dog in these files? Explain your reasoning. File path: For image_agent, use the image path ./data/dog_yes.jpg. For audio_agent, use the audio file path ./data/dog_audio_yes.mp3. For video_agent, use the video file path ./data/dog_video_yes.mp4"}
        ])]}, config
)

for m in messages['messages']:
    m.pretty_print()

# Negative Dog Result
config = {"configurable": {"thread_id": "3"}, "subgraph": True}
messages = react_graph_memory.invoke(
    {"messages": [HumanMessage(content=[
             {"type": "text", "text": "Question: Can you see dog in these files? Explain your reasoning. For image_agent, use the image path ./data/scenery.jpg. For audio_agent, use the audio file path ./data/crowd_cheering.mp3. For video_agent, use the video file path ./data/people_thinking.mp4"}
        ])]}, config
)

for m in messages['messages']:
    m.pretty_print()





