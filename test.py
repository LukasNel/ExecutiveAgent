import requests
from datetime import datetime

# The Modal endpoint URL
url = "https://lukasnel--executive-agent-api-agentapi-query.modal.run"

# Create the query with current datetime
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
query = f"hi, I'm meeting someone tomorrow at 3pm, could you please remind me? {current_time}"
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SYSTEM_PROMPT = """
You are a helpful assistant that can schedule tasks for the user.
Always check to see if the task is already scheduled.
If it is, return the task details.
If it is not, schedule the task and return the task details.
"""
query = (
    SYSTEM_PROMPT
    + "\n\n"
    + f"Message: \n hi, I'm meeting someone tomorrow at 3pm, could you please remind me? Today is {current_time}"
)
data = {
    "query": query,
    "metadata": {"email": "x@x.com", "phone_number": "+12035083967"},
}
# Prepare the request data
print(data)
# Send the POST request
response = requests.post(url, json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.json())
