import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import asyncio
import websockets
import multiprocessing


async def received_data(websocket, queue):
    async for message in websocket:
        print(f"Received message: {message}")
        queue.put(message)
        await websocket.send(f"{message}")

async def ws_main(queue):
    async with websockets.serve(lambda ws: received_data(ws, queue), "localhost", 6789):
        await asyncio.Future()  # Run forever

def run_ws_server(queue):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ws_main(queue))

    