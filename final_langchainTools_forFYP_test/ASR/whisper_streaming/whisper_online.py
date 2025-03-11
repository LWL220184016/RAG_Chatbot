#!/usr/bin/env python3
import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HypothesisBuffer:
    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = 0
        self.last_commited_word = None
        self.logfile = logfile
        
    def insert(self, new, offset):
        # Add offset to timestamps and filter items after last committed time
        self.new = [(a+offset, b+offset, t) for a, b, t in new if a+offset > self.last_commited_time-0.1]
        
        if not self.new or not self.commited_in_buffer:
            return
            
        # Check if new content is close to last committed time
        if abs(self.new[0][0] - self.last_commited_time) < 1:
            # Compare n-grams (up to 5 words) between committed and new content
            cn, nn = len(self.commited_in_buffer), len(self.new)
            
            for i in range(1, min(min(cn, nn), 5) + 1):
                committed_ngram = " ".join(self.commited_in_buffer[-j][2] for j in range(1, i+1))[::-1]
                new_ngram = " ".join(self.new[j-1][2] for j in range(1, i+1))
                
                if committed_ngram == new_ngram:
                    removed_words = [repr(self.new.pop(0)) for _ in range(i)]
                    logger.debug(f"removing last {i} words: {' '.join(removed_words)}")
                    break

    def flush(self):
        commit = []
        
        # Find common prefix between buffer and new
        while self.new and self.buffer and self.new[0][2] == self.buffer[0][2]:
            na, nb, nt = self.new[0]
            commit.append((na, nb, nt))
            self.last_commited_word = nt
            self.last_commited_time = nb
            self.buffer.pop(0)
            self.new.pop(0)
            
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        # Remove committed items that end before given time
        i = 0
        for i, item in enumerate(self.commited_in_buffer):
            if item[1] > time:
                break
        if i > 0:
            self.commited_in_buffer = self.commited_in_buffer[i:]

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    SAMPLING_RATE = 16000
    sep = ""

    def __init__(self, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        """
        tokenizer: sentence tokenizer object for the target language
        buffer_trimming: a pair of (option, seconds) for buffer trimming strategy
        logfile: where to store the log
        """
        self.tokenizer = tokenizer
        self.logfile = logfile
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming
        self.init()

    def init(self, offset=None):
        """Run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0 if offset is None else offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns (prompt, context) tuple"""
        # Find the position where committed text crosses buffer_time_offset
        k = len(self.commited)
        for i in range(len(self.commited)-1, -1, -1):
            if self.commited[i][1] <= self.buffer_time_offset:
                k = i + 1
                break
                
        prefix = self.commited[:k]
        context = self.commited[k:]
        
        # Build prompt (up to 200 chars from end of prefix)
        prompt_parts = []
        char_count = 0
        for item in reversed(prefix):
            text = item[2]
            if char_count + len(text) + 1 > 200:
                break
            prompt_parts.append(text)
            char_count += len(text) + 1
            
        return self.sep.join(reversed(prompt_parts)), self.sep.join(t for _, _, t in context)

    def process_iter(self):
        """Process current audio buffer and return confirmed transcript"""
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        
        res = self.transcribe(self.audio_buffer, init_prompt=prompt)
        tsw = self.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        newly_committed = self.transcript_buffer.flush()
        self.commited.extend(newly_committed)
        
        completed = self.to_flush(newly_committed)
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        # Handle buffer trimming based on strategy
        buffer_length = len(self.audio_buffer) / self.SAMPLING_RATE
        
        if newly_committed and self.buffer_trimming_way == "sentence" and buffer_length > self.buffer_trimming_sec:
            self.chunk_completed_sentence()
        
        trim_threshold = self.buffer_trimming_sec if self.buffer_trimming_way == "segment" else 30
        if buffer_length > trim_threshold:
            self.chunk_completed_segment(res)

        logger.debug(f"len of buffer now: {buffer_length:2.2f}")
        return tsw

    def chunk_completed_sentence(self):
        if not self.commited:
            return
            
        logger.debug(self.commited)
        sents = self.words_to_sentences(self.commited)
        
        for s in sents:
            logger.debug(f"\t\tSENT: {s}")
            
        if len(sents) < 2:
            return
            
        # Keep only the last two sentences
        chunk_at = sents[-2][1]
        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if not self.commited:
            return

        ends = self.segments_end_ts(res)
        last_commit_time = self.commited[-1][1]

        if len(ends) <= 1:
            logger.debug("--- not enough segments to chunk")
            return
            
        # Find suitable segment end time
        for i in range(len(ends)-2, -1, -1):
            end_time = ends[i] + self.buffer_time_offset
            if end_time <= last_commit_time:
                logger.debug(f"--- segment chunked at {end_time:2.2f}")
                self.chunk_at(end_time)
                return
                
        logger.debug("--- last segment not within committed area")

    def chunk_at(self, time):
        """Trims the hypothesis and audio buffer at given time"""
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        cut_samples = int(cut_seconds * self.SAMPLING_RATE)
        
        if cut_samples > 0:
            self.audio_buffer = self.audio_buffer[cut_samples:]
            self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """Split words into sentences using tokenizer"""
        if not words:
            return []
            
        text = " ".join(word[2] for word in words)
        sentences = self.tokenizer.split(text)
        
        result = []
        word_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_start = None
            remaining = sentence
            
            while word_index < len(words) and remaining:
                begin, end, word = words[word_index]
                word = word.strip()
                
                if sentence_start is None and remaining.startswith(word):
                    sentence_start = begin
                    
                if remaining == word:
                    result.append((sentence_start, end, sentence))
                    word_index += 1
                    break
                    
                if not remaining.startswith(word):
                    break
                    
                remaining = remaining[len(word):].strip()
                word_index += 1
                
        return result

    def finish(self):
        """Flush incomplete text when processing ends"""
        incomplete = self.transcript_buffer.complete()
        result = self.to_flush(incomplete)
        logger.debug(f"last, noncommited: {result}")
        self.buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        return result

    def to_flush(self, sents, sep=None, offset=0):
        """Concatenate timestamped words/sentences into a sequence"""
        if not sents:
            return (None, None, "")
            
        sep = self.sep if sep is None else sep
        text = sep.join(item[2] for item in sents)
        begin = offset + sents[0][0]
        end = offset + sents[-1][1]
        
        return (begin, end, text)


class VACOnlineASRProcessor(OnlineASRProcessor):
    '''Voice Activity Controller wrapper for OnlineASRProcessor'''

    def __init__(self, online_chunk_size, *args, **kwargs):
        self.online_chunk_size = online_chunk_size
        self.online = OnlineASRProcessor(*args, **kwargs)
        self.SAMPLING_RATE = self.online.SAMPLING_RATE
        
        # Initialize VAD model
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        from silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(model)
        
        self.logfile = self.online.logfile
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        self.status = None  # 'voice' or 'nonvoice'
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is None:
            # No VAD event detected
            if self.status == 'voice':
                # Continue sending audio during speech
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # Keep only the last second of audio for potential speech detection
                if len(self.audio_buffer) > self.SAMPLING_RATE:
                    self.buffer_offset += len(self.audio_buffer) - self.SAMPLING_RATE
                    self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]
            return
            
        # VAD event detected
        frame = list(res.values())[0] - self.buffer_offset
        
        if 'start' in res and 'end' not in res:
            # Speech start detected
            self.status = 'voice'
            send_audio = self.audio_buffer[frame:]
            self.online.init(offset=(frame + self.buffer_offset) / self.SAMPLING_RATE)
            self.online.insert_audio_chunk(send_audio)
            self.current_online_chunk_buffer_size += len(send_audio)
            self.clear_buffer()
            
        elif 'end' in res and 'start' not in res:
            # Speech end detected
            self.status = 'nonvoice'
            send_audio = self.audio_buffer[:frame]
            self.online.insert_audio_chunk(send_audio)
            self.current_online_chunk_buffer_size += len(send_audio)
            self.is_currently_final = True
            self.clear_buffer()
            
        else:
            # Complete speech segment detected
            beg = res["start"] - self.buffer_offset
            end = res["end"] - self.buffer_offset
            self.status = 'nonvoice'
            send_audio = self.audio_buffer[beg:end]
            self.online.init(offset=(beg + self.buffer_offset) / self.SAMPLING_RATE)
            self.online.insert_audio_chunk(send_audio)
            self.current_online_chunk_buffer_size += len(send_audio)
            self.is_currently_final = True
            self.clear_buffer()

    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            return self.online.process_iter()
        else:
            print("no online update, only VAD", self.status, file=self.logfile)
            return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret