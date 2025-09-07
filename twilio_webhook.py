from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from voice2 import VoiceAssistant
from property_rag import RealEstateRAG   # your RAG system

# Init Flask + Bots
app = Flask(__name__)
bot = VoiceAssistant()
rag = RealEstateRAG()

@app.route("/voice", methods=["POST"])
def voice():
    """Entry point when a call is received"""
    resp = VoiceResponse()

    greeting = (
        f"Good day, {bot.client_data['name']}. "
        f"I'm Alexa from Baaz Landmark Real Estate. "
        f"I noticed your property in {bot.client_data['location']} has increased in value since {bot.client_data['purchase_year']}. "
        f"Would now be a good time to discuss your options?"
    )

    # Gather speech input
    gather = Gather(input="speech", action="/process", timeout=6)
    gather.say(greeting, voice="alice")
    resp.append(gather)

    # If no input detected
    resp.say("Sorry, I didn’t catch that. Goodbye!", voice="alice")
    resp.hangup()
    return Response(str(resp), mimetype="text/xml")


@app.route("/process", methods=["POST"])
def process():
    """Process user speech and generate AI + RAG response"""
    user_input = request.form.get("SpeechResult", "").lower()
    resp = VoiceResponse()

    if not user_input.strip():
        resp.say("Sorry, could you repeat that?", voice="alice")
        return Response(str(resp), mimetype="text/xml")

    # Detect intent
    intent = bot.handle_intents(user_input)

    if intent == "confirm":
        bot.confirm_count += 1
        if bot.confirm_count >= 2:
            resp.say("Perfect. One of our senior advisors will contact you shortly. Thank you for your time!", voice="alice")
            resp.hangup()
            bot.finalized = True
        else:
            resp.say("Great! Let's explore what works best for your situation.", voice="alice")

    elif intent == "reject":
        bot.reject_count += 1
        if bot.reject_count >= 2:
            resp.say("No problem. Thanks for your time. If anything changes, feel free to contact Baaz Landmark anytime.", voice="alice")
            resp.hangup()
            bot.finalized = True
        else:
            resp.say("I understand. Let me share one strategy that might change your perspective.", voice="alice")

    else:
        # Query RAG for property-related intelligence
        rag_answer = rag.query_knowledge_base(user_input)

        # Blend RAG context into Alexa-style pitch
        reply = f"{bot.generate_response(user_input)} By the way, {rag_answer}"

        resp.say(reply, voice="alice")

        # Ask for next input
        gather = Gather(input="speech", action="/process", timeout=6)
        gather.say("What do you think about this option?", voice="alice")
        resp.append(gather)

    return Response(str(resp), mimetype="text/xml")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
