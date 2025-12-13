
import sys
import os
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.machine_client import MachineClient

def test_machine():
    client = MachineClient()
    # Assuming local Klipper or the one in settings. 
    # Use the IP from the logs: 192.168.60.26
    url = "http://192.168.60.26:7125" 
    # Note: Port might be 7125 (Moonraker) or 80. User used port in UI.
    # From previous logs, it seems user sets IP and Port in UI.
    # Let's try 7125 first, commonly Moonraker.
    
    print(f"Connecting to {url}...")
    client.configure(url)
    if not client.connect():
        print("Failed to connect.")
        return

    print("Connected.")
    print(f"Position: {client.get_position()}")
    print(f"Homed: {client.is_homed}")
    
    # Check internal attributes set by _sync_toolhead_status
    print(f"Print Time: {getattr(client, '_print_time', 'N/A')}")
    print(f"Est Print Time: {getattr(client, '_est_print_time', 'N/A')}")
    
    print("Testing wait_until_idle (should be instant if idle)...")
    start = time.time()
    client.wait_until_idle(timeout=5.0)
    end = time.time()
    print(f"wait_until_idle took {end - start:.4f}s")
    
    print("Test Complete.")

if __name__ == "__main__":
    test_machine()
