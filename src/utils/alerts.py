import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from src.utils.inference import translate_text

def send_alert(phone_number, message, language_code='en'):
    """
    Sends an SMS alert using Twilio after translating the message.
    """
    if not phone_number:
        return False, "No phone number provided."
        
    # 1. Translate the message
    translated_message = translate_text(message, language_code)
    
    # 2. Fetch Twilio credentials
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_number = os.getenv('TWILIO_PHONE_NUMBER')
    
    # 3. Send real SMS
    try:
        client = Client(account_sid, auth_token)
        msg = client.messages.create(
            body=translated_message,
            from_=from_number,
            to=phone_number
        )
        print(f"SMS sent successfully. SID: {msg.sid}")
        return True, "SMS sent successfully."
    except TwilioRestException as e:
        print(f"Twilio API Error: {e}")
        return False, f"Twilio API Error: {e.msg}"
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return False, f"Failed to send SMS: {str(e)}"
