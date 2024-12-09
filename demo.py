from fastapi import FastAPI, File, UploadFile
import os
import json

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

user_states = {}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = ""):
    # Save the uploaded file
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Update user state with file information
    if user_id in user_states:
        user_states[user_id]["responses"]["pre_existing_conditions_file"] = file_location

    return {"message": "File uploaded successfully", "file_path": file_location}


@app.post("/chat/")
def chat_with_bot(user_input: dict):
    user_id = user_input["user_id"].strip()
    user_message = user_input["message"].strip()

    if user_id not in user_states:
        user_states[user_id] = {
            "current_question_index": 0,
            "responses": {},
            "current_flow": "initial",
            "welcome_shown": False
        }

    conversation_state = user_states[user_id]
    responses = conversation_state["responses"]
    questions = ["Are you suffering from any pre-existing or chronic conditions?", "Next question..."]

    current_index = conversation_state["current_question_index"]

    if current_index < len(questions):
        question = questions[current_index]

        if question == "Are you suffering from any pre-existing or chronic conditions?":
            if user_message.lower() in ["yes", "no"]:
                responses[question] = user_message
                if user_message.lower() == "yes":
                    return {
                        "response": "Please upload a document related to your condition using the upload API.",
                        "upload_api": "/upload/"
                    }
                else:
                    conversation_state["current_question_index"] += 1
                    return {"response": questions[conversation_state["current_question_index"]]}

        else:
            responses[question] = user_message
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                return {"response": questions[conversation_state["current_question_index"]]}

    # Save responses after completing the questions
    with open("user_responses.json", "w") as file:
        json.dump(responses, file, indent=4)

    return {"response": "Thank you for completing the form.", "final_responses": responses}