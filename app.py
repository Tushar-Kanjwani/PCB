from flask import Flask, render_template, request, session
from chatbot_core import answer_query

app = Flask(__name__)
app.secret_key = "REPLACE_WITH_A_RANDOM_SECRET_KEY"

@app.route("/", methods=["GET", "POST"])
def index():
    # initialize chat history
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_q = request.form.get("user_query", "").strip()
        if user_q:
            # record user message
            session["chat_history"].append(("You", user_q))

            # get the model's answer
            answer, _tokens = answer_query(user_q)

            # record bot response
            session["chat_history"].append(("Bot", answer))

    return render_template("index.html", chat_history=session["chat_history"])

if __name__ == "__main__":
    # accessible at http://127.0.0.1:5000/
    app.run(debug=True)
