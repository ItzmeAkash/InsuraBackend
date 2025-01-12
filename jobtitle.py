from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Define your list of job titles
job_titles = [
    # Healthcare and Medicine
    "Nurse", "Doctor", "Surgeon", "Pharmacist", "Dentist", "Veterinarian", "Psychologist",
    "Psychiatrist", "Physical Therapist", "Occupational Therapist", "Radiologist", 
    "Paramedic/EMT", "Nutritionist", "Dietitian", "Medical Laboratory Technician", 
    "Optometrist", "Dermatologist",
    
    # Technology and IT
    "Software Developer", "Web Developer", "UX/UI Designer", "IT Support Specialist", 
    "Data Analyst", "Data Scientist", "AI/ML Engineer", "Cybersecurity Specialist", 
    "DevOps Engineer", "Blockchain Developer", "Cloud Solutions Architect", 
    "Database Administrator", "Network Administrator", "Systems Analyst", 
    "QA Tester", "QA Analyst", "Product Manager", "Game Developer", "Mobile App Developer","Software Engineer",
    
    # Business and Management
    "Manager", "Marketing Manager", "Operations Manager", "Project Manager", 
    "Business Analyst", "Human Resources Specialist", "Supply Chain Manager", 
    "Procurement Specialist", "Sales Manager", "Financial Analyst", "Investment Banker", 
    "Compliance Officer", "Management Consultant",
    
    # Sales and Marketing
    "Sales Representative", "Marketing Coordinator", "Social Media Manager", 
    "Content Marketing Specialist", "Digital Marketing Specialist", "Brand Manager", 
    "Advertising Specialist", "Public Relations Specialist", "Copywriter", "Telemarketer",
    
    # Education
    "Teacher", "College Professor", "Educational Consultant", "Instructional Designer", 
    "School Administrator", "Tutor", "Special Education Teacher",
    
    # Arts, Design, and Media
    "Graphic Designer", "Graphic Artist", "Video Editor", "Animator", "Photographer", 
    "Musician", "Actor", "Actress", "Art Director", "Journalist", "Writer", "Author", 
    "Copywriter", "Editor", "Proofreader", "Illustrator", "Fashion Designer", 
    "Interior Designer",
    
    # Trades and Skilled Labor
    "Electrician", "Plumber", "Carpenter", "Welder", "Mason", "Auto Mechanic", 
    "HVAC Technician", "Construction Worker", "Machinist", "Heavy Equipment Operator", 
    "Painter",
    
    # Finance and Accounting
    "Accountant", "Auditor", "Financial Analyst", "Bookkeeper", "Tax Consultant", 
    "Loan Officer", "Actuary", "Credit Analyst",
    
    # Public Service and Law
    "Lawyer", "Judge", "Police Officer", "Firefighter", "Paralegal", "Social Worker", 
    "Policy Analyst",
    
    # Hospitality and Service Industry
    "Chef", "Bartender", "Hotel Manager", "Event Planner", "Travel Agent", 
    "Tour Guide", "Housekeeper", "Flight Attendant",
    
    # Science and Research
    "Research Scientist", "Environmental Scientist", "Marine Biologist", "Astronomer", 
    "Biochemist", "Epidemiologist", "Geologist",
    
    # Freelance and Self-Employment
    "Freelance Writer", "Freelance Designer", "Freelance Developer", "Consultant", 
    "Virtual Assistant", "Entrepreneur",
    
    # Miscellaneous
    "Pilot", "Librarian", "Real Estate Agent", "Fitness Trainer", "Security Guard", 
    "Interpreter", "Delivery Driver", "Retail Sales Associate", "Warehouse Supervisor", 
    "Zoologist", "Customer Service Representative",
    
    # Additional Jobs/Subcategories
    "Event Coordinator", "Transcriptionist", "Voice Actor", "Sign Language Interpreter", 
    "Auctioneer", "Pet Groomer", "Forensic Scientist", "Archaeologist"
]

def is_job_title(input_job_title):
    # Use fuzzy matching to find the most similar job titles in the list
    matched_job = process.extractOne(input_job_title, job_titles, scorer=fuzz.partial_ratio)
    if matched_job and matched_job[1] > 80:  # Use a threshold of 80 for the similarity score
        return True
    else:
        return False

# Test the function
input_job = "mobile developer"
print(is_job_title(input_job))  # Output: True