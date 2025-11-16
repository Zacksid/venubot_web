## app.py (Gemini / Google Gen AI version + USER NAME MEMORY)
import time
import traceback
import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = Path(r'D:\venubot_web\intents.json')
WORDS_PKL = Path(r'D:\venubot_web\words.pkl')
CLASSES_PKL = Path(r'D:\venubot_web\classes.pkl')
MODEL_FILE = Path(r'D:\venubot_web\venubot_model.h5')
AVATAR_PATH = Path(r'D:\venubot_web\static\Avtar.png')

# ---------------- Validate files ----------------
missing = []
for file in [INTENTS_PATH, WORDS_PKL, CLASSES_PKL, MODEL_FILE, AVATAR_PATH]:
    if not file.exists():
        missing.append(str(file))
if missing:
    raise FileNotFoundError(
        "❌ Missing required files:\n" + "\n".join(missing) +
        "\n\nPlease make sure all files exist in the venubot_web folder."
    )

# ---------------- NLTK + model load ----------------
lemmatizer = WordNetLemmatizer()
try:
    nltk.word_tokenize("test")
except Exception:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

with INTENTS_PATH.open(encoding="utf-8") as f:
    intents = json.load(f)

words = pickle.load(open(WORDS_PKL, "rb"))
classes = pickle.load(open(CLASSES_PKL, "rb"))
model = load_model(MODEL_FILE)

# ---------------- Gemini setup ----------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = None
_gemini_ready = False

if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        _gemini_ready = True
    except ImportError:
        print("⚠️ 'google-genai' SDK missing. Install: pip install google-genai")
    except Exception as e:
        print("⚠️ Gemini init failed:", e)

# ---------------- USER NAME MEMORY ----------------
user_name = None

def extract_user_name(message):
    """
    Detect user's name from:
    - "I am Sid"
    - "I'm Sid"
    - "hi I'm Sid"
    - "my name is Sid"
    """
    msg = message.lower()

    triggers = ["i am ", "i'm ", "im ", "my name is "]
    for t in triggers:
        if t in msg:
            name_part = message[msg.index(t) + len(t):].strip()
            name = name_part.split()[0]
            if name.isalpha():
                return name.capitalize()
    return None


# ---------------- updated personalize_reply ----------------
def personalize_reply(reply):
    """
    Add name to greeting-like responses, but avoid duplicating the name.
    If the user's name already appears in the reply (case-insensitive), return as-is.
    """
    global user_name
    if not user_name:
        return reply

    # If name already present in the reply, do nothing
    if user_name.lower() in reply.lower():
        return reply

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "good night"]
    reply_low = reply.lower()

    for g in greetings:
        if reply_low.startswith(g):
            # replace only the first occurrence (preserve capitalization of greeting)
            return reply.replace(reply[:len(g)], f"{g.capitalize()} {user_name}", 1)

    return reply


# ---------------- Prediction helpers ----------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    sentence_word_set = set(sentence_words)
    bag = [1 if word in sentence_word_set else 0 for word in words]
    return np.array(bag, dtype=np.float32)

def predict_class(sentence, threshold=0.25):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': float(r[1])} for r in results]

def get_response_from_intent_tag(tag):
    for intent in intents.get('intents', []):
        if intent.get('tag') == tag:
            return random.choice(intent.get('responses'))
    return "Hmm, I didn't understand that."

# ---------------- Gemini fallback ----------------
def generate_with_gemini(user_message, model_name="gemini-2.5-flash"):
    if not gemini_client:
        raise RuntimeError("Gemini not available")

    system_prompt = (
        "You are VenuBot, a friendly Indian flute-themed chatbot. "
        "Keep replies short, sweet, and polite."
    )

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=[system_prompt, user_message],
            temperature=0.7,
            max_output_tokens=150
        )
        text = getattr(response, "text", None)
        if not text:
            text = getattr(response, "output_text", str(response))
        return text.strip()
    except Exception as e:
        print("[Gemini error]", e)
        raise

def local_fallback(message):
    if len(message.split()) <= 2:
        return "Could you say a bit more?"
    return f"I didn't understand that, {user_name or 'friend'}. Try rephrasing it."

# ---------------- Options logic ----------------
FIXED_OPTION_TAGS = ['fun_fact', 'ragas_info', 'indian_flute_players']

def sample_option_label(intent):
    patterns = intent.get("patterns", [])
    if patterns:
        return random.choice(patterns).strip().capitalize()
    return intent["tag"].replace("_", " ").capitalize()

def build_options():
    all_intents = intents["intents"]
    tag_to_intent = {i["tag"]: i for i in all_intents}
    options = [tag_to_intent[tag] for tag in FIXED_OPTION_TAGS if tag in tag_to_intent]

    remaining = [i for i in all_intents if i not in options]
    if remaining:
        options.append(random.choice(remaining))

    # Ensure 4
    while len(options) < 4 and remaining:
        options.append(remaining.pop())

    return options[:4]

# ---------------- Routes ----------------
@app.route("/")
def home():
    hour = datetime.now().astimezone().hour
    if hour < 12:
        greet = "Good morning"
    elif hour < 17:
        greet = "Good afternoon"
    elif hour < 21:
        greet = "Good evening"
    else:
        greet = "Good night"

    return render_template("index.html", avatar_url="/avatar", startup_greeting=greet)

@app.route("/avatar")
def avatar():
    return send_from_directory(AVATAR_PATH.parent, AVATAR_PATH.name)

# Test Gemini
@app.route("/test_gemini")
def test_gemini():
    if not gemini_client:
        return jsonify({"ok": False, "error": "Gemini not configured"})
    try:
        r = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["Say hi in one sentence."]
        )
        return jsonify({"ok": True, "reply": r.text})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# Main Chat Route
@app.route("/chat", methods=["POST"])
def chat():
    global user_name

    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    option_tag = data.get("option_tag")

    # Step A: detect name first
    detected_name = extract_user_name(message)
    if detected_name:
        user_name = detected_name

        # Try to use the friendly flute response from intents if available
        flute_reply = None
        for it in intents.get('intents', []):
            if it.get('tag') == 'love_flute_yes':
                flute_reply = random.choice(it.get('responses', []))
                break

        if not flute_reply:
            flute_reply = "That's wonderful! 🎵 It's great to meet someone who appreciates the soothing sound of the flute."

        # Build greeting with name once
        bot_text = f"Hi {user_name}! {flute_reply}"

        # Personalize (safe because personalize_reply won't duplicate the name)
        bot_text = personalize_reply(bot_text)

        # Build options and return immediately
        options = build_options()
        options_json = [{"id": str(i+1), "label": sample_option_label(opt), "tag": opt.get("tag")} for i, opt in enumerate(options)]
        return jsonify({"bot_message": bot_text, "options": options_json})

    # Normal flow if no immediate name-intro
    try:
        if option_tag:
            bot_text = get_response_from_intent_tag(option_tag)
        else:
            if message:
                ints = predict_class(message)
                if ints:
                    tag = ints[0]['intent']
                    bot_text = get_response_from_intent_tag(tag)
                else:
                    if gemini_client is not None:
                        try:
                            bot_text = generate_with_gemini(message)
                        except Exception as e:
                            print("[Gemini] generate failed, falling back to local_fallback. Error:", type(e).__name__, e)
                            bot_text = local_fallback(message)
                    else:
                        bot_text = local_fallback(message)
            else:
                bot_text = "🎶 Say something and I'll sing back!"

        # Personalize the bot_text for future replies (idempotent)
        bot_text = personalize_reply(bot_text)

    except Exception as e:
        traceback.print_exc()
        bot_text = "Sorry — server encountered an error. Try again."

    # Build options and return
    options = build_options()
    options_json = [{"id": str(i+1), "label": sample_option_label(opt), "tag": opt.get("tag")} for i,opt in enumerate(options)]
    return jsonify({"bot_message": bot_text, "options": options_json})


# ---------------- Run ----------------
if __name__ == "__main__":
    print("✅ Model and intents loaded.")
    if _gemini_ready:
        print("✅ Gemini ready.")
    else:
        print("⚠️ Gemini not available — using fallback.")
    app.run(host="127.0.0.1", port=5000, debug=True)
