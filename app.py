import asyncio
import chainlit as cl
from model import GraphRAGPipeline

rag = GraphRAGPipeline.load()

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Welcome to CarGPT! 🚗 Ask me anything about cars.",
        author="CarGPT",
        actions=[cl.Action(name="customize-btn", label="Customize", payload={})]
    ).send()

@cl.action_callback("customize-btn")
async def customize(action: cl.Action):
    await cl.Message(
        content="Customization options will be available soon!",
        author="CarGPT"
    ).send()
    await action.remove()

@cl.on_message
async def message_handler(message: cl.Message):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, rag.run_agent, message.content)
    await cl.Message(content=response, author="CarGPT").send()
