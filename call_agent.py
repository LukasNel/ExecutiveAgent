import os
import requests
from typing import Optional
import json
from datetime import datetime, UTC

class WhatsAppMessenger:
    def __init__(self, access_token: Optional[str] = None, phone_number_id: Optional[str] = None):
        """
        Initialize WhatsApp messenger with access token.
        
        Args:
            access_token: WhatsApp Business API access token. If not provided, will look for WHATSAPP_TOKEN env var.
        """
        self.access_token = access_token or os.getenv("WHATSAPP_TOKEN")
        if not self.access_token:
            raise ValueError("WhatsApp access token not provided and WHATSAPP_TOKEN environment variable not set")
        
        self.base_url = "https://graph.facebook.com/v17.0"
        self.phone_number_id = phone_number_id or os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        if not self.phone_number_id:
            raise ValueError("WhatsApp phone number ID not provided and WHATSAPP_PHONE_NUMBER_ID environment variable not set")

    def send_message(self, to_number: str, message: str) -> dict:
        """
        Send a WhatsApp message to a specified number.
        
        Args:
            to_number: The recipient's phone number (with country code, e.g., "1234567890")
            message: The message to send
            
        Returns:
            dict: Response from the WhatsApp API
        """
        # Format the phone number (remove any non-digit characters)
        to_number = ''.join(filter(str.isdigit, to_number))
        
        # Ensure the number has country code
            
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": message}
        }
        print(data)
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to send WhatsApp message: {str(e)}")

def send_whatsapp_message(phone_number: str, message: str) -> dict:
    """
    Send a WhatsApp message using the WhatsApp Business API.
    
    Args:
        phone_number: The recipient's phone number (with or without country code)
        message: The message to send
        
    Returns:
        dict: Response from the WhatsApp API
    """
    messenger = WhatsAppMessenger()
    return messenger.send_message(phone_number, message)

def format_message_for_whatsapp(message: str) -> str:
    """
    Format a message for WhatsApp, adding any necessary formatting or metadata.
    
    Args:
        message: The original message
        
    Returns:
        str: Formatted message
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"{message}\n\nSent at: {timestamp}"

def test_whatsapp_messaging():
    """
    Test the WhatsApp messaging functionality with a test message.
    
    Returns:
        dict: Response from the WhatsApp API if successful
    """
    test_phone = "27793171232"  # Replace with actual test phone number
    test_message = "This is a test message from the WhatsApp Business API"
    test_business_phone_number_id = "288534607680963"
    access_token="EABuK9dp39DIBO9PRuigE5j51IEZAnpVpGWDHjce4TknPARvKou4eSoMoooLSc2rSfI0Jn87Sh6hJpZAymg2PSEPUUfcZCMJZB5END1gWSsHBNskNfovd2ksZArj4UaB98TZChxSNLBHP63z0p9z3yRxRPZBZBjHTuyLFGAEFkMpWqZAdqKRmINFfGBDIVBFMhl0fRJgZDZD",
    access_token = "EABuK9dp39DIBO8tZAvxBrFBv6EFuCCwE1dZCxmyLY0MhYqGYZAZAwU316bF2ZATE0Vq3TBKahHuwEfRShxLDvhgomDX8ZA5VVVZBHt4UZAsLuASPb430IE2mg7GKwx40z1OCPOkzVfldNlPNzR66bDmDzWmn7DY8ZBVu08AjQNi5sldp39ElyWSWlZCICjSAGqfDtarjo7nZC4mKy721dLK3sTqOMGpz1Md"
    try:
        # Create messenger instance with test token
        messenger = WhatsAppMessenger(
            access_token=access_token,
            phone_number_id=test_business_phone_number_id
        )
        
        # Send test message
        response = messenger.send_message(test_phone, test_message)
        print("Test message sent successfully!")
        print(f"API Response: {response}")
        return response
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise

test_whatsapp_messaging()