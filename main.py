import modal
from agent import ExecutiveAgent, MessageT
from typing import Dict, Any, Optional
import json
import requests
from datetime import datetime, UTC
import os
from pydantic import BaseModel
from smolagents import ActionStep, AgentType
import json
from typing import Any

class QueryRequestT(BaseModel):
    query: str
    metadata: dict
    

class CustomEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        # First try to use built-in serialization
        try:
            return super().default(obj)
        except TypeError:
            # Try to call .dict() method (common in Pydantic models)
            try:
                return obj.dict()
            except (AttributeError, TypeError):
                # Fall back to string representation
                return str(obj)

# Example usage
def serialize_to_json(data: Any) -> str:
    return json.dumps(data, cls=CustomEncoder, indent=2)
    
# Create a Modal image with required dependencies
image = modal.Image.debian_slim("3.12").pip_install_from_requirements("requirements.txt")
# Create Modal app
app = modal.App(name="executive-agent-api", image=image)
VOLUME_DIR = "/alldata"
# Create a secret for the Zapier webhook URL
webhook_secret = modal.Secret.from_name("executive-agent", required_keys=["ZAPIER_WEBHOOK_URL", "ANTHROPIC_API_KEY"])

@app.cls(volumes={VOLUME_DIR: modal.Volume.from_name("executiveagent", create_if_missing=True)}, secrets=[webhook_secret])
class AgentAPI:
    def __init__(self):
        self.webhook_url = None
    
    @modal.enter()
    def setup(self):
        """Initialize the webhook URL from Modal secrets."""
        self.webhook_url = os.getenv("ZAPIER_WEBHOOK_URL")
    
    def send_to_zapier(self, query: str, metadata: dict, response: dict) -> bool:
        """
        Send the query and response to Zapier webhook.
        
        Args:
            query: The original query
            response: The agent's response
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.webhook_url:
            return False
            
        payload = {
            "query": query,
            "response": response,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": metadata
        }
        
        response = requests.post(
            self.webhook_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return True

    @modal.fastapi_endpoint(method="POST", docs=True,)
    def query(self, data: QueryRequestT) -> Dict[str, Any]:
        query = data.query
        metadata = data.metadata
        agent = ExecutiveAgent(enable_extended_thinking=True, db_path=os.path.join(VOLUME_DIR, "tasks.db"))
        """
        Process a query using the ExecutiveAgent.
        
        Args:
            data: Dictionary containing the query string and optional zapier flag
            
        Returns:
            Dictionary containing the agent's response
        """
        if not query:
            return {"error": "No query provided"}
            
        # Run the agent and collect all steps
        steps = []
        for step in agent.run(query):
            print(step)
            steps.append(step)
        with open(os.path.join(VOLUME_DIR, "steps.txt"), "w") as f:
            for step in steps:
                if isinstance(step, ActionStep):
                    dictionary = step.dict()
                    f.write(serialize_to_json(dictionary) + "\n")
                else:
                    f.write(serialize_to_json(step) + "\n")
        # Return the last step as the final response
        if steps:
            message: dict = steps[-1]
            
            # Send to Zapier if requested
            zapier_success = self.send_to_zapier(query, metadata, message)
            return {
                "response": message,
                "zapier_sent": zapier_success
            }
            
        else:
            return {"error": "No response generated"}
                
