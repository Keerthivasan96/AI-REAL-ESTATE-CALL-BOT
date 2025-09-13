"""
chat.py - AI-Powered Real Estate Voice Assistant
------------------------------------------------
This script implements a voice-enabled real estate assistant that:
- Listens to user input via microphone
- Uses Gemini API for natural conversational responses
- Uses Google Cloud TTS for speech output
- Maintains conversation flow and handles polite endings

Note:
- Replace API keys and credentials in a `.env` file
- Client data is sample/demo only
"""

import os
import time
import speech_recognition as sr
from dotenv import load_dotenv
import pygame
from google.cloud import texttospeech
from google.generativeai import GenerativeModel

# Load environment variables
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GCP_TTS_KEY")

# Initialize audio + model
pygame.mixer.init()
tts_client = texttospeech.TextToSpeechClient()
model = GenerativeModel("gemini-2.5-flash")


class VoiceAssistant:
    """Voice-enabled real estate assistant"""

    def __init__(self, call_id=None):
        self.call_id = call_id or f"local_{int(time.time())}"
        self.reset_state()

    def reset_state(self):
        """Reset bot memory for a new conversation"""
        self.recognizer = sr.Recognizer()
        self.client_data = {
            "name": "Demo Client",
            "location": "Sample City",
            "bedrooms": 2,
            "bought_price": 1_200_000,
            "current_price": 3_300_000,
            "purchase_year": 2020,
        }

        self.conversation_history = []
        self.conversation_ended = False

    def speak(self, text: str):
        """Convert text to speech using Google TTS"""
        if not text or len(text.strip()) < 5:
            text = "I apologize, let me rephrase that."

        clean_text = self.clean_text_for_tts(text)
        print(f"[Assistant]: {clean_text}")

        try:
            input_text = texttospeech.SynthesisInput(text=clean_text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-F",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            response = tts_client.synthesize_speech(
                input=input_text, voice=voice, audio_config=audio_config
            )

            temp_file = f"temp_tts_{self.call_id}.mp3"
            with open(temp_file, "wb") as out:
                out.write(response.audio_content)

            sound = pygame.mixer.Sound(temp_file)
            sound.play()
            while pygame.mixer.get_busy():
                time.sleep(0.1)

            os.remove(temp_file)

        except Exception as e:
            print(f"TTS Error: {e}")

    def clean_text_for_tts(self, text: str) -> str:
        """Clean text for natural speech synthesis"""
        text = (
            text.replace("AED", "Dirhams")
            .replace("ROI", "return on investment")
            .replace("AI", "A I")
        )
        return " ".join(text.split())

    def listen(self) -> str:
        """Capture user input from microphone"""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")

            try:
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                print(f"[User]: {text}")
                return text.lower()
            except Exception:
                return ""

    def generate_response(self, user_input: str) -> str:
        """Generate a natural conversational response with Gemini"""
        prompt = f"""
You are an AI property consultant at RealEstateCo.

Client: {self.client_data['name']} owns a {self.client_data['bedrooms']}-bedroom property
in {self.client_data['location']}, purchased in {self.client_data['purchase_year']}
for {self.client_data['bought_price']} Dirhams. It is now worth {self.client_data['current_price']} Dirhams.

User just said: "{user_input}"

Your role:
- Respond naturally, in 2-4 sentences
- Be professional, friendly, and informative
- If user declines or ends, respond politely
"""
        try:
            response = model.generate_content(prompt).text.strip()
            return response
        except Exception as e:
            print(f"Gemini Error: {e}")
            return "I understand. Let me connect you with a senior advisor who can provide detailed information."

    def check_conversation_end(self, user_input: str) -> bool:
        """Check if conversation should end"""
        end_signals = [
            "not interested",
            "don't want",
            "call later",
            "busy now",
            "goodbye",
            "bye",
            "thank you",
            "stop calling",
        ]
        return any(signal in user_input for signal in end_signals)

    def run(self):
        """Run the voice assistant"""
        print(f"Starting conversation: {self.call_id}")

        greeting = (
            f"Good day, {self.client_data['name']}. "
            f"This is your AI assistant from RealEstateCo. "
            f"Can we discuss your property in {self.client_data['location']}?"
        )
        self.speak(greeting)

        while not self.conversation_ended:
            user_input = self.listen()
            if not user_input:
                self.speak("Sorry, could you repeat that?")
                continue

            if self.check_conversation_end(user_input):
                self.speak("Thank you for your time. Have a great day!")
                self.conversation_ended = True
                break

            response = self.generate_response(user_input)
            self.speak(response)

            self.conversation_history.append(
                {"user": user_input, "bot": response, "timestamp": time.time()}
            )


if __name__ == "__main__":
    bot = VoiceAssistant()
    bot.run()
