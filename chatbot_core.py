import pandas as pd
from functools import lru_cache
import google.generativeai as genai # Import the Google Generative AI library

# --- Configure the Gemini API client ---
# Using the API key provided by the user.
# For security in production, consider loading this from an environment variable.
genai.configure(api_key="AIzaSyCBbV_vzEqR1aqqZtdq6iXbukcPUm2o1u8")

# --- In-memory storage for chat logs ---
chat_logs = [] # Each entry: {"user_query": ..., "model_response": ..., "total_tokens": ...}
last_user_query = None  # For follow-up context
last_model_response = None

# --- Load dataset table into an array of dicts ---
def load_dataset():
    """
    Loads data from 'Dataset.csv' into a pandas DataFrame,
    drops unnamed columns, and converts it to a list of dictionaries.
    """
    try:
        df = pd.read_csv("Dataset.csv")
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        print("Error: Dataset.csv not found. Please make sure the file is in the same directory.")
        return []

dataset = load_dataset()

@lru_cache(maxsize=1)
def load_context():
    """
    Loads and formats the dataset into a single text blob,
    suitable for providing as context to the LLM.
    """
    if not dataset:
        return "No dataset available."

    df = pd.DataFrame(dataset)
    # Define a mapping for column names to more descriptive labels for the LLM
    mapping = {
        "Name": "Name",
        "DOB": "Date of Birth",
        "Age": "Age",
        "Place": "Birthplace",
        "Test Runs": "Test Runs",
        "ODI Runs": "ODI Runs",
        "T20 Runs": "T20 Runs",
        "International Runs": "Runs",
        "Maximum Score": "Highest Score",
        "Last Match Venue": "Last Match Venue",
        "Last Match Date": "Last Match Date",
        "Runs in Last Match": "Last Match Runs"
    }
    lines = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            if col in mapping:
                label = mapping[col]
                parts.append(f"{label}: {row[col]}")
        lines.append(", ".join(parts))
    return "\n".join(lines)

# Generate the context data once
context_data = load_context()

def get_cached_response(user_query: str):
    """
    Checks if a response for the given user query exists in the chat logs (cache).
    Returns the cached response if found, otherwise returns None.
    """
    for log in reversed(chat_logs): # Search from most recent
        if log["user_query"] == user_query:
            return log["model_response"]
    return None

def log_chat(user_query: str, model_response: str, total_tokens: int):
    """
    Logs a new question/answer pair along with token count into the chat_logs.
    """
    global last_user_query, last_model_response
    chat_logs.append({
        "user_query": user_query,
        "model_response": model_response,
        "total_tokens": total_tokens
    })
    last_user_query = user_query
    last_model_response = model_response

def answer_query(user_query: str) -> tuple[str, int]:
    global last_user_query, last_model_response
    """
    Answers a user query about cricket players.
    First, it tries to retrieve a cached response.
    If not found, it calls the Gemini LLM with the dataset as context
    and then logs the result before returning.
    Returns a tuple of (answer_string, total_tokens_used).
    """
    # 1) Try cache
    cached = get_cached_response(user_query)
    if cached:
        print("Returning cached response.")
        return cached, 0 # 0 tokens used for cached response

    is_follow_up = any(word in user_query.lower() for word in ["more", "another", "next", "one more", "give me another"])
    if is_follow_up and last_user_query and last_model_response:
        followup_instruction = (
            f"\nThis is a follow-up. Previous question:\n"
            f"'{last_user_query}'\n"
            f"Answer:\n'{last_model_response}'\n"
            f"Now answer this:\n{user_query}"
        )
    else:
        followup_instruction = f"Now answer this question:\n{user_query}"

    answer = "I couldn't find an answer."
    total_tokens = 0
    
    # 2) Otherwise, ask the Gemini model
   

    if not context_data or context_data == "No dataset available.":
        answer = "I cannot answer questions as the dataset could not be loaded."
        return answer, total_tokens

    try:
        # Initialize the GenerativeModel for chat interactions
        model = genai.GenerativeModel(model_name="gemini-2.0-flash") # Using gemini-2.0-flash

        # Construct the messages for the Gemini API
        messages = [
            {
                "role": "user",
                "parts": [{"text": "You are an assistant who answers questions about cricket players from a dataset. Provide direct answers based on the provided data, without extra commentary or explanation."}]
            },
            {
                "role": "user",
                "parts": [{"text": f"Here is the dataset:\n\n{context_data}\n\nNow answer this question:\n{followup_instruction}"}]
            }
        ]

        # Make the API call
        response = model.generate_content(
            contents=messages,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=512, # Limit output length
                temperature=0.0,      # Make responses more deterministic
            )
        )

        answer = response.text
        # Retrieve total token count from the response metadata
        total_tokens = response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") else 0

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        answer = "I encountered an error while trying to get an answer. Please try again later."

    # 3) Log and return
    log_chat(user_query, answer, total_tokens)
    return answer, total_tokens

# --- Example Usage (for testing the script) ---
if __name__ == "__main__":
    print("Cricket Player Q&A Bot (Type 'exit' to quit)")
    print("-" * 30)

    # Create a dummy Dataset.csv for testing if it doesn't exist
    try:
        with open("Dataset.csv", "x") as f:
            f.write("Name,DOB,Age,Place,Test Runs,ODI Runs,T20 Runs,International Runs,Maximum Score,Last Match Venue,Last Match Date,Runs in Last Match\n")
            f.write("Virat Kohli,1988-11-05,35,Delhi,8848,13848,4008,26696,254*,Lord's,2024-07-20,75\n")
            f.write("Rohit Sharma,1987-04-30,37,Nagpur,4137,10709,3974,18820,264,MCG,2024-07-15,50\n")
            f.write("Babar Azam,1994-10-15,29,Lahore,3772,5729,3698,13199,196,Dubai,2024-07-18,102\n")
    except FileExistsError:
        pass # File already exists, no need to create

    # Reload dataset and context if dummy file was just created
    dataset = load_dataset()
    context_data = load_context()

    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() == 'exit':
            break

        if not user_input:
            print("Please enter a question.")
            continue

        response, tokens = answer_query(user_input)
        print(f"Bot: {response}")
        print(f"(Tokens used for this query: {tokens})")

    print("\nChat history:")
    for i, log in enumerate(chat_logs):
        print(f"--- Chat {i+1} ---")
        print(f"Q: {log['user_query']}")
        print(f"A: {log['model_response']}")
        print(f"Tokens: {log['total_tokens']}")
