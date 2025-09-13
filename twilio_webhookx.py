"""
twilio_webhookx.py - AI Voice Assistant Webhook with Twilio
-----------------------------------------------------------
This Flask app integrates Twilio Voice with a Generative AI real estate assistant.
It enables real-time phone conversations where:
- User speaks → Twilio transcribes speech and sends text to Flask
- Flask processes with Gemini + optional RAG context
- Flask responds with Twilio <Say> (text-to-speech) back to the caller

Note:
- Configure your `.env` with TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, GOOGLE_API_KEY
- Expose this app using ngrok or deploy to a server for Twilio to reach it
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
import google.generativeai as genai
from properties_RAG import RealEstateRAG  # Import your RAG system

# Configure logging
logging.basicConfig(level=logging.INFO)

# Flask app
app = Flask(__name__)

# Init RAG + Gemini
rag = RealEstateRAG()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Thread pool for parallel tasks
executor = ThreadPoolExecutor(max_workers=4)

# In-memory session storage (simple demo)
sessions = {}


class CallAssistant:
    """Simple stateful AI call assistant"""

    def __init__(self, call_id: str):
        self.call_id = call_id
        self.reset()

    def reset(self):
        """Initialize conversation state"""
        self.client_data = {
            "name": "Demo Client",
            "location": "Sample City",
            "bedrooms": 2,
            "bought_price": 1_200_000,
            "current_price": 3_300_000,
            "purchase_year": 2020,
        }
        self.confirm_count = 0
        self.reject_count = 0
        self.finalized = False

    def handle_intents(self, user_input: str) -> str:
        """Basic intent detection"""
        confirm_triggers = ["yes", "sure", "go ahead", "ok", "interested"]
        reject_triggers = ["no", "not now", "not interested", "stop", "leave me"]

        if any(word in user_input for word in confirm_triggers):
            return "confirm"
        elif any(word in user_input for word in reject_triggers):
            return "reject"
        return "unknown"

    def generate_response(self, user_input: str) -> str:
        """Generate AI response via Gemini"""
        prompt = f"""
You are Alexa, an AI real estate consultant at RealEstateCo.

Client details:
- {self.client_data['bedrooms']}-bedroom in {self.client_data['location']}
- Bought in {self.client_data['purchase_year']} for {self.client_data['bought_price']} Dirhams
- Current value: {self.client_data['current_price']} Dirhams

User said: "{user_input}"

Guidelines:
- Speak naturally (2–3 sentences)
- If they hesitate, emphasize timing and opportunities
- If they confirm, suggest connecting with an advisor
- If they reject twice, politely close the call
"""
        try:
            response = model.generate_content(prompt).text.strip()
            return response
        except Exception as e:
            logging.error(f"Gemini error: {e}")
            return "I understand. Let me connect you with an advisor who can assist further."


@app.route("/voice", methods=["POST"])
def voice():
    """Initial Twilio call entrypoint"""
    call_sid = request.form.get("CallSid")
    sessions[call_sid] = {"bot": CallAssistant(call_sid), "start": time.time()}
    bot = sessions[call_sid]["bot"]

    resp = VoiceResponse()
    greeting = (
        f"Hello {bot.client_data['name']}, this is Alexa from RealEstateCo. "
        f"I’d like to discuss your property in {bot.client_data['location']}. "
        f"Is now a good time?"
    )

    gather = Gather(input="speech", action="/process", timeout=5, speech_timeout="auto")
    gather.say(greeting, voice="alice")
    resp.append(gather)

    resp.say("Sorry, I didn't catch that. Goodbye.", voice="alice")
    resp.hangup()
    return Response(str(resp), mimetype="text/xml")


@app.route("/process", methods=["POST"])
def process():
    """Handle user speech input during call"""
    call_sid = request.form.get("CallSid")
    user_input = request.form.get("SpeechResult", "").lower()
    logging.info(f"User said: {user_input}")

    resp = VoiceResponse()

    if call_sid not in sessions:
        resp.say("Session expired. Please call back.", voice="alice")
        resp.hangup()
        return Response(str(resp), mimetype="text/xml")

    bot = sessions[call_sid]["bot"]

    if not user_input:
        resp.say("Could you repeat that?", voice="alice")
        return Response(str(resp), mimetype="text/xml")

    intent = bot.handle_intents(user_input)

    if intent == "confirm":
        bot.confirm_count += 1
        if bot.confirm_count >= 2:
            resp.say(
                "Perfect. Our senior advisor will contact you soon. Thank you for your time.",
                voice="alice",
            )
            resp.hangup()
            del sessions[call_sid]
            return Response(str(resp), mimetype="text/xml")
        else:
            reply = "Great! Let's explore what works best for you."
    elif intent == "reject":
        bot.reject_count += 1
        if bot.reject_count >= 2:
            resp.say(
                "No problem. Thanks for your time. Have a wonderful day!",
                voice="alice",
            )
            resp.hangup()
            del sessions[call_sid]
            return Response(str(resp), mimetype="text/xml")
        else:
            reply = "I understand. Let me share one strategy that might change your perspective."
    else:
        # Parallel RAG + Gemini for richer response
        rag_future = executor.submit(rag.query_knowledge_base, user_input)
        ai_future = executor.submit(bot.generate_response, user_input)

        try:
            rag_context = rag_future.result(timeout=3)
            ai_response = ai_future.result(timeout=4)
            reply = f"{ai_response} {rag_context}"
        except Exception as e:
            logging.error(f"Error combining RAG + Gemini: {e}")
            reply = bot.generate_response(user_input)

    resp.say(reply, voice="alice")

    # Keep conversation alive
    gather = Gather(input="speech", action="/process", timeout=5, speech_timeout="auto")
    gather.say("What are your thoughts?", voice="alice")
    resp.append(gather)

    return Response(str(resp), mimetype="text/xml")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return {"status": "healthy", "active_sessions": len(sessions)}


if __name__ == "__main__":
    # For dev, run locally + use ngrok to expose port
    app.run(host="0.0.0.0", port=5000, debug=True)
