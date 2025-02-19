from rapidfuzz import process

# Insurance list with corresponding numbers
insurance_options = {
    1: "Takaful Emarat (Ecare)",
    2: "National Life & General Insurance (Innayah)",
    3: "Takaful Emarat (Aafiya)",
    4: "National Life & General Insurance (NAS)",
    6: "Orient UNB Takaful (Nextcare)",
    7: "Orient Mednet (Mednet)",
    8: "Al Sagr Insurance (Nextcare)",
    9: "RAK Insurance (Mednet)",
    10: "Dubai Insurance (Dubai Care)",
    11: "Fidelity United (Nextcare)",
    12: "Salama April International (Salama)",
    13: "Sukoon (Sukoon)",
    14: "Orient basic"
}

def find_matching_insurance(text):
    words = text.split()  # Split text into words
    matched_results = []
    for word in words:
        match = process.extractOne(word, insurance_options.values(), score_cutoff=70)
        if match:
            matched_string, score, index = match  # Unpack all three values
            # Find the key (number) for the matched value
            option_number = next(k for k, v in insurance_options.items() if v == matched_string)
            matched_results.append((option_number, matched_string))
    if matched_results:
        return {"matches": matched_results, "count": len(matched_results)}
    else:
        return {"matches": [], "count": 0}

# Example usage
text = "Send me an email for Takaful"
result = find_matching_insurance(text)
print(result)