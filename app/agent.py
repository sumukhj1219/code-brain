import os
import asyncio
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.plugins import groq

load_dotenv()

async def entrypoint(ctx: JobContext):
    model = groq.LLM(model="llama-3.3-70b-versatile")
    
    chat_ctx = llm.ChatContext()
    
    chat_ctx.messages.append(
        llm.ChatMessage(
            role="system",
            content="You are a helpful CLI assistant powered by Groq."
        )
    )

    await ctx.connect()
    print("\n--- Groq CLI Ready ---")

    while True:
        try:
            user_text = await asyncio.to_thread(input, "You: ")
            
            if user_text.lower() in ["exit", "quit"]:
                break

            chat_ctx.messages.append(
                llm.ChatMessage(role="user", content=user_text)
            )

            stream = model.chat(chat_ctx=chat_ctx)
            print("Assistant: ", end="", flush=True)
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print("\n")
            
            chat_ctx.messages.append(
                llm.ChatMessage(role="assistant", content=full_response)
            )

        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))