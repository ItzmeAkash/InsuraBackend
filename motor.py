# # Determine the correct file to save
# save_file = "claim.json" if conversation_state["current_flow"] == "claim" else "user_responses.json"

# # Save the responses
# with open(save_file, "w") as file:
#     json.dump(responses, file, indent=4)

# # Return the response with appropriate key
# return {
#     "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
#     "claim" if save_file == "claim.json" else "final_response": responses
# }
