import os
import time
import speech_recognition as sr
from dotenv import load_dotenv
import pygame
from google.cloud import texttospeech
from google.generativeai import GenerativeModel

# Load env
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GCP_TTS_KEY")

# Init TTS + Audio
pygame.mixer.init()
tts_client = texttospeech.TextToSpeechClient()
model = GenerativeModel("gemini-2.5-flash")

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.client_data = {
            'name': 'John Smith',
            'location': 'Downtown Dubai',
            'bedrooms': 2,
            'bought_price': 1_200_000,
            'current_price': 3_300_000,
            'purchase_year': 2020
        }
        self.last_intent = None
        self.reject_count = 0
        self.confirm_count = 0
        self.finalized = False

    def speak(self, text):
        print(f"[Alexa]: {text}")
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-F",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        temp_file = "temp_google_tts.mp3"
        with open(temp_file, "wb") as out:
            out.write(response.audio_content)

        sound = pygame.mixer.Sound(temp_file)
        sound.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)

    def listen(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("👂 Listening for user input...")
            try:
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                print(f"[User]: {text}")
                return text.lower()
            except Exception:
                self.speak("Sorry, could you repeat that?")
                return ""

    def generate_response(self, user_input):
        prompt = f'''
You are Alexa, a sharp and persuasive senior property consultant at Baaz Landmark in Dubai.

Client: {self.client_data['name']} owns a {self.client_data['bedrooms']}-bedroom property in {self.client_data['location']}, purchased in {self.client_data['purchase_year']} for AED {self.client_data['bought_price']}. It's now worth AED {self.client_data['current_price']}.

You're on a call with the client. Your job is to:
1. Present clear SELL and RENT options — with pros and cons.
2. Recommend SELLING using urgency, market timing, and ROI logic.
3. If the client hesitates, explain portfolio reinvestment strategies — like selling the villa and investing in multiple apartments with 8–10% rental yield and 15–20% annual appreciation.
4. Be professional but passionate — act like you're helping them gain, not just selling something.
5. If they agree, say a senior advisor will contact them. If they reject twice, thank them politely and exit.

You MUST evolve the conversation. Avoid repeating phrases. Keep replies under 4 lines. Make it feel like a real consultation.

User said: "{user_input}"

Respond as Alexa, in a friendly and confident tone.
'''
        start = time.time()
        response = model.generate_content(prompt).text.strip()
        print(f"⚡ Gemini-only response took {round(time.time() - start, 2)}s")
        return response

    def handle_intents(self, user_input):
        confirm_triggers = ["yes", "let's do it", "go ahead", "i'm ready", "interested", "please do", "okay", "ok"]
        reject_triggers = ["no", "not now", "not interested", "leave me", "don't", "not today"]

        if any(word in user_input for word in confirm_triggers):
            return "confirm"
        elif any(word in user_input for word in reject_triggers):
            return "reject"
        return "unknown"

    def run(self):
        print("🎙️ Calibrating microphone (one-time)...")
        print("🚀 Warming up Gemini model...")
        _ = self.generate_response("Just warming up. Say hi!")

        greeting = (
            f"Good morning, {self.client_data['name']}. I'm Alexa from Baaz Landmark Real Estate. "
            f"I've reviewed your {self.client_data['bedrooms']}-bedroom property in {self.client_data['location']}. "
            f"Would now be a good time to discuss it?"
        )
        self.speak(greeting)

        while not self.finalized:
            user_input = self.listen()
            if not user_input:
                continue

            intent = self.handle_intents(user_input)

            if intent == "confirm":
                self.confirm_count += 1
                if self.confirm_count >= 2:
                    self.speak("Perfect. One of our senior advisors from Baaz Landmark will contact you shortly. Thank you for your time!")
                    self.finalized = True
                    break
                else:
                    self.speak("Great! Let's explore what works best for your situation.")
                    continue

            elif intent == "reject":
                self.reject_count += 1
                if self.reject_count >= 2:
                    self.speak("No problem. Thanks for your time. If anything changes, feel free to contact Baaz Landmark anytime.")
                    self.finalized = True
                    break
                else:
                    self.speak("I understand. Let me share one strategy that might change your perspective.")
                    continue

            reply = self.generate_response(user_input)
            self.speak(reply)

# Run bot
if __name__ == "__main__":
    bot = VoiceAssistant()
    bot.run()
