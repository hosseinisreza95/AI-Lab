import os
from openai import OpenAI
from dotenv import load_dotenv
import subprocess

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API"))

import json
messages = [{
    "role": "system",
    "content": """You are a helpful python coding agent.
                you can read the files using Read tool."""
}]

tools = [{
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file form disk",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string"
                }
            },
            "required": ["file_path"]
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    }
                }
            }
        }
    }
]

while True:
    user_input = input("User: ")

    messages.append({
        "role": "user",
        "content": user_input
    })
    chat = client.chat.completions.create(
        model= 'gpt-4o-mini',
        messages= messages,
        tools= tools
    )

    message = chat.choices[0].message

    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "Read":
                argeuments = json.loads(tool_call.function.arguments)
                file_path = argeuments.get("file_path")

                with open (file_path, "r") as f:
                    content = f.read()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })

                print("\n [FILE CONTENT]: \n", content)

            elif tool_call.function.name == "Write":
                    arguments = json.loads(tool_call.function.arguments)
                    file_path = arguments.get("file_path")
                    file_content = arguments.get("content")
                    
                    
                    with open(file_path, 'w') as file:
                        file.write(file_content)
                    tool_response = "File written successfully."
                
                
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response
                    })
            
            elif tool_call.function.name == "Bash":
                    arguments = json.loads(tool_call.function.arguments)
                    command = arguments.get("command")
                    
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    # Combine standard output and error output for the model
                    tool_response = result.stdout + result.stderr
                
                    # Append the bash execution result back to the messages loop
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response
                    })
    
    else:
        if message.content:
            print(f"\n Agent: {message.content}")
        
        break

    

