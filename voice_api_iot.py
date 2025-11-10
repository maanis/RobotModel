#!/usr/bin/env python3

import os
import time
import tempfile
import traceback
import requests
import speech_recognition as sr
from gtts import gTTS
import re
import pygame

# ---------------------------------------------------------
# CONFIG — replace with YOUR API URL
API_URL = "https://d3bfed67086a.ngrok-free.app/get-answer"   # << replace
# ---------------------------------------------------------

recognizer = sr.Recognizer()
mic = sr.Microphone()

def sanitize_text(text):
    """
    Remove/sanitize characters that break prompts.
    Escapes quotes instead of removing meaning.
    """
    text = text.replace('"', '\\"')  # escape double-quotes
    text = text.replace("'", "\\'")  # escape single-quotes
    return text.strip()

def listen_and_transcribe(timeout=5, phrase_time_limit=8):
    """Listen from microphone and return text."""
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return None

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except:
        print("Could not understand the audio.")
        return None

def call_my_api(user_text):
    """Call your custom API and get response."""
    try:
        # sanitize / escape quotes
        user_text = sanitize_text(user_text)

        payload = {"question": user_text}  # adjust JSON key as required
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            print(response.json())
            return response.json().get("answer")  # adjust based on your API
        else:
            print("API Error:", response.status_code)
            return None
    except Exception as e:
        print("API Call failed:", e)
        traceback.print_exc()
        return None

def speak(text):
    """Speak text using gTTS and pygame for playback."""
    try:
        if not text:
            return
        
        # Create a temporary file with a simple path to avoid Windows issues
        temp_dir = tempfile.gettempdir()
        mp3_path = os.path.join(temp_dir, "tts_output.mp3")
            
        tts = gTTS(text=text, lang="en")
        tts.save(mp3_path)
        
        # Use pygame for playback (more reliable on Windows)
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.quit()
        
        # Clean up
        try:
            os.remove(mp3_path)
        except:
            pass  # Ignore cleanup errors
            
    except Exception as e:
        print("TTS Error:", e)
        traceback.print_exc()

def main():
    print("=== Voice IoT Bot using your custom API ===")
    print("Press Ctrl+C to exit")

    while True:
        user_text = listen_and_transcribe()
        if not user_text:
            continue

        # Send text to your API
        print("Sending to your API...")
        answer = call_my_api(user_text)

        if not answer:
            print("No response from API.")
            continue

        print("API replied:", answer)

        # speak
        print("Speaking response...")
        speak(answer)

        time.sleep(0.5)

if __name__ == "__main__":
    main()
