#!/usr/bin/env python3
import asyncio
import websockets
import json
import numpy as np
import sys
import argparse
import logging
import time
import io
import soundfile as sf
from whisper_online import (
    add_shared_args, set_logging, asr_factory, logger
)

# run_server.py --model medium --vac
# run_client.py

# Configure logging
logging.basicConfig(format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)

class WebSocketASRProcessor:
    """Process audio chunks via WebSocket and return transcriptions"""
    
    SAMPLING_RATE = 16000
    
    def __init__(self, online_asr_proc):
        self.online_asr_proc = online_asr_proc
        self.last_end = None
        self.is_first = True
    
    def reset(self):
        """Reset the ASR processor for a new session"""
        self.online_asr_proc.init()
        self.last_end = None
        self.is_first = True
    
    def process_audio_chunk(self, audio_data):
        """Process an audio chunk and return any transcription results"""
        self.online_asr_proc.insert_audio_chunk(audio_data)
        result = self.online_asr_proc.process_iter()
        
        return self.format_result(result)
    
    def format_result(self, result):
        """Format the ASR result for client response"""
        beg, end, text = result
        
        if beg is not None:
            beg_ms = beg * 1000
            end_ms = end * 1000
            
            # Ensure non-overlapping segments for better client-side handling
            if self.last_end is not None:
                beg_ms = max(beg_ms, self.last_end)
            
            self.last_end = end_ms
            
            return {
                "start": beg_ms,
                "end": end_ms,
                "text": text,
                "final": False
            }
        return None
    
    def finish(self):
        """Process final audio and return any remaining transcription"""
        result = self.online_asr_proc.finish()
        formatted = self.format_result(result)
        if formatted:
            formatted["final"] = True
        return formatted

async def handle_client(websocket, asr_processor):
    """Handle a client WebSocket connection"""
    client_ip = websocket.remote_address
    logger.info(f"Client connected from {client_ip}")
    
    # Reset ASR processor for this new connection
    asr_processor.reset()
    
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Process binary audio data
                try:
                    # Direct conversion from bytes to numpy array instead of using SoundFile
                    # Assuming the audio is sent as 16-bit PCM samples
                    audio_int16 = np.frombuffer(message, dtype=np.int16)
                    
                    # Convert to float32 format required by the ASR processor
                    audio_data = audio_int16.astype(np.float32) / 32767.0
                    
                    # Process the audio chunk
                    result = asr_processor.process_audio_chunk(audio_data)
                    
                    # Send result if there's transcription
                    if result:
                        await websocket.send(json.dumps(result))
                
                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    await websocket.send(json.dumps({
                        "error": f"Failed to process audio: {str(e)}"
                    }))
            
            elif isinstance(message, str):
                # Handle text commands
                try:
                    cmd = json.loads(message)
                    if cmd.get("command") == "finish":
                        # Process any final audio
                        result = asr_processor.finish()
                        if result:
                            await websocket.send(json.dumps(result))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON command: {message}")
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_ip} disconnected")
    
    except Exception as e:
        logger.error(f"Error handling client: {str(e)}")
    
    finally:
        logger.info(f"Connection closed for client {client_ip}")

async def start_server(host, port, asr_processor):
    """Start the WebSocket server"""
    server = await websockets.serve(
        lambda ws: handle_client(ws, asr_processor),
        host, port
    )
    logger.info(f"ASR WebSocket server running at ws://{host}:{port}")
    await server.wait_closed()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WebSocket ASR Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host address to bind the server to")
    parser.add_argument("--port", type=int, default=8765, 
                        help="Port number to bind the server to")
    parser.add_argument("--force-cpu", action="store_true", default=False,
                        help="Force CPU processing even if GPU is available")
    
    add_shared_args(parser)
    args = parser.parse_args()
    
    set_logging(args, logger, other="")
    
    logger.info("Initializing ASR model...")
    start_time = time.time()
    
    try:
        if args.force_cpu:
            logger.info("Forcing CPU processing as requested")
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        asr, online_processor = asr_factory(args)
        logger.info(f"ASR model loaded in {time.time() - start_time:.2f} seconds")
        
        # Test the model with a small audio chunk to ensure it works
        logger.info("Verifying ASR model...")
        dummy_audio = np.zeros(1600, dtype=np.float32)
        test_result = asr.transcribe(dummy_audio)
        logger.info("ASR model verified and ready")
        
        # Create ASR processor for WebSocket usage
        ws_asr_processor = WebSocketASRProcessor(online_processor)
        
        # Start the WebSocket server
        logger.info(f"Starting WebSocket server on {args.host}:{args.port}")
        asyncio.run(start_server(args.host, args.port, ws_asr_processor))
    
    except Exception as e:
        logger.error(f"Failed to initialize ASR model or server: {e}")
        logger.error("Please ensure you have the correct CUDA and cuDNN versions installed for GPU processing")
        logger.error("Or try running with --force-cpu flag to disable GPU usage")
        sys.exit(1)

if __name__ == "__main__":
    main()
