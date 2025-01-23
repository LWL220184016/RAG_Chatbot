import os 
import sys
import asyncio
import websockets
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

async def received_data(websocket, audio_input_queue, text_input_queue):
    loop = asyncio.get_event_loop()
    async for message in websocket:
        print(f"Received message: {message}")
        # Non-blocking put into multiprocessing queue
        # await loop.run_in_executor(None, text_input_queue, audio_input_queue.put, message)
        await loop.run_in_executor(None, text_input_queue.put, message)

async def send_data(websocket, llm_output_queue, audio_queue):
    loop = asyncio.get_event_loop()
    while True:
        # Non-blocking get from multiprocessing queues
        llm_output = await loop.run_in_executor(None, llm_output_queue.get)
        audio_chunk = await loop.run_in_executor(None, audio_queue.get)
        
        # Send both pieces of data (modify formatting as needed)
        await websocket.send(f"LLM: {llm_output}")
        await websocket.send(f"AUDIO: {audio_chunk}")

async def handler(websocket, audio_input_queue, text_input_queue, llm_output_queue, audio_queue):
    receive_task = asyncio.create_task(received_data(websocket, audio_input_queue, text_input_queue))
    send_task = asyncio.create_task(send_data(websocket, llm_output_queue, audio_queue))
    
    done, pending = await asyncio.wait(
        [receive_task, send_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    
    for task in pending:
        task.cancel()
    for task in done:
        if task.exception():
            print(f"Task error: {task.exception()}")

async def ws_main(audio_input_queue, text_input_queue, llm_output_queue, audio_queue):
    async with websockets.serve(
        lambda ws: handler(ws, audio_input_queue, text_input_queue, llm_output_queue, audio_queue),
        "localhost",
        6789
    ):
        await asyncio.Future()  # Run forever

def run_ws_server(audio_input_queue, text_input_queue, llm_output_queue, audio_queue):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ws_main(audio_input_queue, text_input_queue, llm_output_queue, audio_queue))

if __name__ == "__main__":
    # Create multiprocessing queues
    input_queue = multiprocessing.Queue()
    llm_queue = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    # Start WebSocket server in a process
    server_process = multiprocessing.Process(
        target=run_ws_server,
        args=(input_queue, llm_queue, audio_queue)
    )
    server_process.start()

    # Example of sending data from another process
    llm_queue.put("Sample LLM output")
    audio_queue.put(b"Sample audio data")

    try:
        server_process.join()
    except KeyboardInterrupt:
        server_process.terminate()

# import os 
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# import asyncio
# import websockets
# import multiprocessing


# async def received_data(websocket, text_input_queue):
#     async for message in websocket:
#         print(f"Received message: {message}")
#         text_input_queue.put(message)
#         await websocket.send(f"{message}")

# def send_data(websocket, llm_output_queue, audio_queue):
#     llm_output = llm_output_queue.get()
#     audio_chunk = audio_queue.get()

# async def ws_main(text_input_queue):
#     async with websockets.serve(lambda ws: received_data(ws, text_input_queue), "localhost", 6789):
#         await asyncio.Future()  # Run forever

# def run_ws_server(text_input_queue):
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(ws_main(text_input_queue))

    