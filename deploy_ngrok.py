import subprocess
import time
from pyngrok import ngrok

def main():
    print("Starting Streamlit server on port 8501...")
    # Start Streamlit in the background
    streamlit_process = subprocess.Popen(["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])

    print("Waiting 5 seconds for Streamlit to boot...")
    time.sleep(5)

    print("Opening ngrok tunnel...")
    try:
        # Open a tunnel to port 8501
        public_url = ngrok.connect(8501)
        print(f"\n=======================================================")
        print(f"🎉 AgriNova is LIVE on ngrok!")
        print(f"🌍 Public URL: {public_url}")
        print(f"=======================================================\n")
        
        print("Press Ctrl+C to shut down the server and tunnel.")
        # Keep the script running
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting ngrok: {e}")
        print("Note: You may need to configure your ngrok auth token if this is your first time using it.")
        print("Run: ngrok config add-authtoken <YOUR_TOKEN>")
    finally:
        ngrok.kill()
        streamlit_process.terminate()

if __name__ == "__main__":
    main()
