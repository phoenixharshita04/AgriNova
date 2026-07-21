import subprocess
import time
import re

def main():
    print("Starting Streamlit server on port 8501...")
    # Start Streamlit in the background
    streamlit_process = subprocess.Popen(["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])

    print("Waiting 5 seconds for Streamlit to boot...")
    time.sleep(5)

    print("Opening public tunnel via SSH (localhost.run)...")
    try:
        # Open a tunnel using SSH
        ssh_process = subprocess.Popen(
            ["ssh", "-R", "80:localhost:8501", "nokey@localhost.run", "-o", "StrictHostKeyChecking=accept-new"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        url = None
        # Read output line by line to grab the URL
        for line in iter(ssh_process.stdout.readline, ''):
            # Print SSH output just in case
            print(line.strip())
            
            match = re.search(r'(https://[a-zA-Z0-9.-]+\.lhr\.life)', line)
            if match:
                url = match.group(1)
                break
                
            if ssh_process.poll() is not None:
                break
                
        if url:
            print(f"\n=======================================================")
            print(f"AgriNova is LIVE!")
            print(f"Public URL: {url}")
            print(f"=======================================================\n")
            print("Press Ctrl+C to shut down the server and tunnel.")
        else:
            print("\nCould not automatically parse the public URL from the SSH output.")
            print("Check the logs above to find it.")

        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting tunnel: {e}")
    finally:
        if 'ssh_process' in locals():
            ssh_process.terminate()
        streamlit_process.terminate()

if __name__ == "__main__":
    main()
