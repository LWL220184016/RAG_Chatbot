#!/usr/bin/env python3
import argparse
import numpy as np
import sounddevice as sd
import whisper_online
from queue import Queue
import threading
import logging
import sys
import time

# Set up logging
logging.basicConfig(format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AudioStreamer:
    """Captures audio from microphone and processes it in real-time with Whisper ASR"""
    
    def __init__(self, args):
        self.args = args
        self.audio_queue = Queue()
        self.is_running = False
        self.sample_rate = 16000
        
        # Initialize ASR
        self.asr, self.online_processor = whisper_online.asr_factory(args)
        
        # Warm up the ASR model
        logger.info("Warming up the ASR model...")
        dummy_audio = np.zeros(1600, dtype=np.float32)
        self.asr.transcribe(dummy_audio)
        logger.info("ASR model ready!")
        
        # Check is using VAC
        self.chunk_size = args.vac_chunk_size if args.vac else args.min_chunk_size
        self.samples_per_chunk = int(self.chunk_size * self.sample_rate)
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to capture audio"""
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        # Convert audio to mono if needed and ensure it's float32
        audio_data = indata.copy()
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]  # Take first channel if stereo
        audio_data = audio_data.astype(np.float32)
        
        # Add to queue
        self.audio_queue.put(audio_data)
    
    def process_audio(self):
        """Process audio chunks from the queue"""
        while self.is_running:
            try:
                # Get audio chunk from queue
                if self.audio_queue.empty():
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning
                    continue
                    
                audio_chunk = self.audio_queue.get()
                
                # Process the audio
                self.online_processor.insert_audio_chunk(audio_chunk)
                result = self.online_processor.process_iter()
                
                # Output the transcript
                if result[0] is not None:
                    start_time, end_time, text = result
                    print(f"[{start_time:.2f}s - {end_time:.2f}s] {text}")
                    sys.stdout.flush()
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
    
    def start(self):
        """Start the streaming and processing"""
        self.is_running = True
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start audio capture
        try:
            logger.info("Starting audio capture. Speak into your microphone...")
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.samples_per_chunk
            ):
                # Keep running until interrupted
                while self.is_running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Stopping...")
        except Exception as e:
            logger.error(f"Error capturing audio: {str(e)}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop streaming and processing"""
        self.is_running = False
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        # Process any final audio that might be in the buffer
        result = self.online_processor.finish()
        if result[0] is not None:
            start_time, end_time, text = result
            print(f"[FINAL: {start_time:.2f}s - {end_time:.2f}s] {text}")
            sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Real-time Speech Recognition with Whisper")
    
    # Add arguments from whisper_online
    whisper_online.add_shared_args(parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    whisper_online.set_logging(args, logger, other="")
    
    # Create and start the audio streamer
    streamer = AudioStreamer(args)
    streamer.start()

if __name__ == "__main__":
    main()