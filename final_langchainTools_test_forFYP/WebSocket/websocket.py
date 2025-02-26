import os 
import sys
import asyncio
import websockets
import queue
import multiprocessing.queues
import base64
import traceback
import soundfile as sf
import io
import time

from websockets.exceptions import ConnectionClosed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# 全局字典，用于跟踪每个客户端的处理任务
client_tasks = {}

async def received_data(
        websocket, 
        audio_input_queue: multiprocessing.Queue = None, 
        text_input_queue: multiprocessing.Queue = None, 
    ):

    try:
        async for message in websocket:
            if isinstance(message, str):
                print(f"Received text: {message}")
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    text_input_queue.put,
                    message
                )

            elif isinstance(message, bytes):
                print(f"Received audio: len {len(message)}")
                print(f"Data type: {type(message)}")
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    audio_input_queue.put,
                    message
                )
                # print("Invalid message received")
    except ConnectionClosed:
        print("接收循环检测到连接关闭")
    # except Exception as e:
        # print(f"receiving loop Exception: {str(e)}")

async def send_data(
        websocket, 
        asr_queue: multiprocessing.Queue = None, 
        llm_queue: multiprocessing.Queue = None, 
        tts_queue: multiprocessing.Queue = None, 
    ):

    try:
        sample_rate = 16000
        while True:

            try:
                asr_output = await asyncio.get_event_loop().run_in_executor(
                    None,
                    asr_queue.get_nowait
                )
                await websocket.send(f"You: {asr_output}")
            except queue.Empty:
                asyncio.sleep(0.1)
                pass

            try:
                llm_output = await asyncio.get_event_loop().run_in_executor(
                    None,
                    llm_queue.get_nowait
                )
                await websocket.send(f"LLM: {llm_output}")
            except queue.Empty:
                asyncio.sleep(0.1)
                pass

            try:
                audio_chunk = await asyncio.get_event_loop().run_in_executor(
                    None,
                    tts_queue.get_nowait
                )
                buffer = io.BytesIO()
                sf.write(buffer, audio_chunk, sample_rate, format='wav')
                buffer.seek(0)
                audio_chunk = buffer.read()
                base64_chunk = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send(f"AUDIO: {base64_chunk}")  # 发送 base64 编码的数据
            except queue.Empty:
                asyncio.sleep(0.1)
                pass
            
    except ConnectionClosed:
        print("LLM发送通道检测到连接关闭")
    except Exception as e:
        print(f"send text Exception: {str(e)}")
        print(traceback.format_exc())

async def connection_watcher(websocket):
    """连接状态监控协程"""
    try:
        await websocket.wait_closed()
        print("连接监控器检测到连接关闭")
    except Exception as e:
        print(f"监控器异常: {str(e)}")

async def handler(
        websocket, 
        audio_input_queue: multiprocessing.Queue = None, 
        text_input_queue: multiprocessing.Queue = None, 
        asr_output_queue: multiprocessing.Queue = None, 
        llm_output_queue: multiprocessing.Queue = None, 
        tts_queue: multiprocessing.Queue = None, 
    ):

    client_id = websocket.remote_address
    print(f"新连接来自 {client_id}")
    print("websocket.remote_address", websocket.remote_address)

    # 清理同一来源的旧任务
    if client_id in client_tasks:
        old_task = client_tasks[client_id]
        print(f"发现旧任务 {client_id}，正在取消...")
        old_task.cancel()
        try:
            await old_task
        except asyncio.CancelledError:
            print(f"旧任务 {client_id} 已被取消")
        except Exception as e:
            print(f"等待旧任务时发生异常: {e}")
        # 确保移除旧任务条目
        if client_id in client_tasks and client_tasks[client_id] is old_task:
            del client_tasks[client_id]

    # 注册当前任务
    current_task = asyncio.current_task()
    client_tasks[client_id] = current_task
    print(f"注册新任务 {client_id}")

    try:
        watcher_task = asyncio.create_task(connection_watcher(websocket))
        
        tasks = [
            asyncio.create_task(received_data(websocket, audio_input_queue, text_input_queue)),
            asyncio.create_task(send_data(websocket, asr_output_queue, llm_output_queue, tts_queue)),
            watcher_task
        ]

        try:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                if task.exception():
                    print(f"任务异常: {task.exception()}")
                    
        except Exception as e:
            print(f"主handler异常: {str(e)}")
        finally:
            print("queue_size: ", 
                  "audio input: ", audio_input_queue.qsize(), 
                  ", text input: ", text_input_queue.qsize(), 
                  ", asr text output: ", asr_output_queue.qsize(),
                  ", llm text output: ", llm_output_queue.qsize(), 
                  ", audio output: ", tts_queue.qsize())
            print("开始清理连接...")
            # 取消所有未完成的任务
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # 等待所有任务结束
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 主动关闭连接
            await websocket.close()
            print("连接清理完成")
    finally:
        # 从字典中移除当前任务
        if client_id in client_tasks and client_tasks[client_id] is current_task:
            del client_tasks[client_id]
            print(f"任务 {client_id} 已从client_tasks中移除")

async def ws_main(
        host: str = "localhost",
        port: int = 6789,
        audio_input_queue: multiprocessing.Queue = None, 
        text_input_queue: multiprocessing.Queue = None, 
        asr_output_queue: multiprocessing.Queue = None, 
        llm_output_queue: multiprocessing.Queue = None, 
        tts_queue: multiprocessing.Queue = None, 
    ):

    # 配置服务器参数
    server_config = {
        "host": host,
        "port": port,
        "ping_interval": 1,
        "ping_timeout": 1,
        "close_timeout": 1
    }
    
    async with websockets.serve(
        lambda ws: handler(ws, audio_input_queue, text_input_queue, asr_output_queue, llm_output_queue, tts_queue),
        **server_config
    ):
        print(f"WebSocket服务器启动在 {server_config['host']}:{server_config['port']}")
        await asyncio.Future()

def run_ws_server(
        host: str = "localhost",
        port: int = 6789,
        is_asr_ready_event = None, 
        is_llm_ready_event = None, 
        is_tts_ready_event = None, 
        audio_input_queue: multiprocessing.Queue = None, 
        text_input_queue: multiprocessing.Queue = None, 
        asr_output_queue: multiprocessing.Queue = None, 
        llm_output_queue: multiprocessing.Queue = None, 
        tts_queue: multiprocessing.Queue = None, 
    ):

    while not all([is_asr_ready_event.is_set(), is_llm_ready_event.is_set(), is_tts_ready_event.is_set()]):
        time.sleep(0.1)

    # 配置事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(
            ws_main(
                host,
                port,
                audio_input_queue, 
                text_input_queue, 
                asr_output_queue, 
                llm_output_queue, 
                tts_queue
            )
        )
    except KeyboardInterrupt:
        print("服务器正常关闭")
    finally:
        loop.close()