import socket
import json
import zlib
import struct
import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import io
import threading


SERVER_HOST = '172.16.26.21' 
SERVER_PORT = 9999
TIMEOUT = 5.0 
POLLING_INTERVAL_MS = 2000 


class AppData:
    def __init__(self):
        self.last_analysis = {}
        self.last_image_data = None
        

app_data = AppData()
# Global widget references
root = None
metrics_label = None
image_label = None
risk_label = None

# ---------------------------
# ===== NETWORK FUNCTIONS ===
# ---------------------------

def receive_data(sock):
    """Receives a block of data prefixed with its 4-byte size (unsigned long)."""
    
    # --------------------------------------------------------------------------
    # FIX: Use a robust loop to guarantee exactly 4 bytes for the header read.
    # This avoids using the socket.MSG_WAITALL flag, which can sometimes trigger
    # WinError 10045 (Operation not supported) on Windows.
    # --------------------------------------------------------------------------
    
    header_size = 4
    raw_size = b''
    
    try:
        sock.settimeout(TIMEOUT)
        
        # 1. Receive the 4-byte size header
        while len(raw_size) < header_size:
            # Calculate how many bytes are still needed
            chunk = sock.recv(header_size - len(raw_size))
            if not chunk:
                # Connection closed by server before receiving header
                return None
            raw_size += chunk
        
        if len(raw_size) < header_size:
            # Should not happen if loop completes, but as a safeguard
            return None
            
    except socket.timeout:
        if root: messagebox.showwarning("Timeout", "No response from server during header read.")
        return None
    except Exception as e:
        if root: messagebox.showerror("Receive Error", f"Socket error during header read: {e}")
        return None

    # Unpack the size
    size = struct.unpack("!L", raw_size)[0]
    
    # 2. Receive the full data payload
    data = b''
    try:
        while len(data) < size:
            # We are reading the remaining size needed
            chunk = sock.recv(size - len(data))
            if not chunk:
                break # Connection closed unexpectedly
            data += chunk
            
    except socket.timeout:
        if root: messagebox.showwarning("Timeout", "Incomplete data received from server.")
        return None
    except Exception as e:
        if root: messagebox.showerror("Receive Error", f"Socket error during data read: {e}")
        return None

    # Only return data if the full expected size was received
    return data if len(data) == size else None

def _fetch_data(get_image=False):
    """
    Core function to connect, request, and receive data.
    Uses a temporary socket for robustness.
    """
    sock = None
    try:
        # 1. Establish connection (per-request)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        
        # Set a shorter connect timeout, separate from the read/recv timeout
        connect_timeout = 3.0
        sock.settimeout(connect_timeout)
        sock.connect((SERVER_HOST, SERVER_PORT))
        sock.settimeout(TIMEOUT) # Revert to full timeout for receiving data

        # We assume the server is set up to send:
        # 1. JSON analysis data
        # 2. JPEG frame data (optional based on command/protocol)

        # 1. Receive JSON Data (compressed)
        compressed_json = receive_data(sock)
        if compressed_json:
            decompressed_data = zlib.decompress(compressed_json)
            app_data.last_analysis = json.loads(decompressed_data.decode('utf-8'))
        else:
            # If we don't get JSON, we can't proceed.
            raise Exception("Failed to receive JSON analysis data.")

        # Update metrics immediately
        if root:
            root.after(0, update_gui_metrics)

        # 2. Receive Frame Data (JPEG) only if requested
        if get_image:
            frame_bytes = receive_data(sock)
            if frame_bytes:
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    raise Exception("Failed to decode image frame.")
                    
                # Convert OpenCV image to PIL Image for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                app_data.last_image_data = Image.fromarray(rgb_frame)
                
                # Update GUI image
                if root:
                    root.after(0, display_image)
            else:
                # This is only an error if we explicitly asked for the image
                raise Exception("Failed to receive image data after successful JSON.")


    except Exception as e:
        if root: 
            # Only show communication errors on demand, not every polling failure
            if get_image:
                messagebox.showerror("Communication Error", f"Failed to get data: {e}")
            
            # Ensure disconnected status is reflected on the risk label
            if risk_label and root:
                root.after(0, lambda: risk_label.config(text="DISCONNECTED", background='gray'))
    finally:
        if sock:
            sock.close()

def start_continuous_analysis():
    """
    Schedules the non-blocking thread to fetch analysis data periodically.
    Runs on the main thread using root.after().
    """
    # Start the actual data fetching in a separate thread (get_image=False)
    threading.Thread(target=lambda: _fetch_data(get_image=False), daemon=True).start()
    
    # Reschedule itself to run again after the polling interval
    if root:
        root.after(POLLING_INTERVAL_MS, start_continuous_analysis)

def on_demand_image_request():
    """Handles button press to request a full update (analysis + image)."""
    # Run the full fetch in a separate thread (get_image=True)
    threading.Thread(target=lambda: _fetch_data(get_image=True), daemon=True).start()


# ---------------------------
# ===== GUI FUNCTIONS =======
# ---------------------------

def update_gui_metrics():
    """Updates the labels with the latest analysis data."""
    if not metrics_label or not risk_label:
        return
        
    data = app_data.last_analysis
    
    heads_1 = data.get('heads', 'N/A')
    heads_2 = data.get('crowd_heads_count', 'N/A')
    max_cell = data.get('max_cell', 'N/A')
    motion = data.get('avg_motion', 'N/A')
    risk = data.get('final_risk', 'N/A')
    
    # Calculate crowd density percentage relative to an arbitrary maximum (e.g., 50 heads)
    try:
        density_pct = min(100, int(heads_2) * 2) if isinstance(heads_2, (int, str)) and str(heads_2).isdigit() else 'N/A'
    except:
        density_pct = 'N/A'

    metrics_text = (
        f"Frame: {data.get('frame', 'N/A')}\n"
        f"--- DETECTION ---\n"
        f"Standard Heads: {heads_1}\n"
        f"Crowd Heads: {heads_2}\n"
        f"Max Cell Density: {max_cell}\n"
        f"Crowd Density %: {density_pct}%\n"
        f"--- STAMPEDE RISK ---\n"
        f"Average Motion: {motion:.3f}%\n"
        f"Active Conditions: {data.get('active_conditions', 'N/A')}\n"
        f"RISK LEVEL: {risk}"
    )
    metrics_label.config(text=metrics_text)
    
    # Set risk color
    if risk == 'HIGH':
        risk_label.config(text=" STAMPEDE ALERT ", background='red', foreground='white')
    elif risk == 'MEDIUM':
        risk_label.config(text=" High Risk ", background='yellow', foreground='black')
    else:
        # If successfully updated, mark as connected
        risk_label.config(text=" Low Risk", background='green', foreground='white')
        

def display_image():
    """Displays the last received image in the canvas, resizing it to fit the current label size."""
    if not image_label or not app_data.last_image_data:
        return
        
    img = app_data.last_image_data
    
    # Get current size of the label where the image will be displayed
    label_width = image_label.winfo_width()
    label_height = image_label.winfo_height()
    
    # Default size if not yet rendered
    if label_width == 1 or label_height == 1:
        # Use a sensible default size if the widget hasn't fully rendered yet
        label_width, label_height = 450, 450 

    # Resize image for display using calculated bounds
    display_img = img.copy()
    display_img.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)
    
    # Convert PIL Image to Tkinter PhotoImage
    tk_img = ImageTk.PhotoImage(display_img)
    
    image_label.config(image=tk_img)
    image_label.image = tk_img # Keep a reference!

# ---------------------------
# ===== MAIN GUI SETUP ======
# ---------------------------

def setup_gui():
    """Initializes the Tkinter window and widgets."""
    global root, metrics_label, image_label, risk_label
    
    root = tk.Tk()
    root.title("Real-Time Stampede Detector Client (Continuous Analysis)")
    
    # Configure style for better appearance
    style = ttk.Style()
    style.configure('TFrame', background='#e8e8e8')
    style.configure('TLabelFrame', background='#e8e8e8')
    style.configure('TLabel', background='#e8e8e8')
    style.configure('Metrics.TLabel', font=('Courier', 11))
    
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # --- LEFT PANE: CONTROLS AND METRICS ---
    control_frame = ttk.LabelFrame(main_frame, text="Controls & Metrics", padding="10")
    control_frame.grid(row=0, column=0, padx=10, pady=5, sticky=(tk.N, tk.W, tk.E, tk.S)) 
    control_frame.grid_rowconfigure(3, weight=1)

    # Display the correct server IP in the GUI
    ttk.Label(control_frame, text=f"Server: {SERVER_HOST}:{SERVER_PORT}", font=('Arial', 10)).pack(pady=5)
    
    # Risk Label (Dynamic background)
    risk_label = ttk.Label(control_frame, text="CONNECTING...", font=('Arial', 14, 'bold'),
                             background='gray', foreground='white', padding=10)
    risk_label.pack(pady=10, fill='x')

    # Request Button (now only for image/forced full update)
    request_button = ttk.Button(control_frame, text="GET NEW IMAGE (On Demand)",
                                   command=on_demand_image_request,
                                   style='TButton')
    request_button.pack(pady=10, fill='x')
    
    # Metrics Display Label
    metrics_label = ttk.Label(control_frame, 
                              text="Starting continuous analysis thread...", 
                              justify=tk.LEFT, 
                              style='Metrics.TLabel',
                              anchor='nw')
    metrics_label.pack(pady=10, fill='both', expand=True)

    # --- RIGHT PANE: IMAGE DISPLAY ---
    image_frame = ttk.LabelFrame(main_frame, text="Last Captured Image", padding="10")
    image_frame.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.N, tk.W, tk.E, tk.S)) 
    image_frame.columnconfigure(0, weight=1)
    image_frame.rowconfigure(0, weight=1)
    
    image_label = ttk.Label(image_frame, text="No Image Available", anchor='center')
    image_label.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))
    
    # Start the continuous background analysis loop
    start_continuous_analysis()
    
    # Bind the display_image function to the Configure event so the image resizes when the window resizes
    image_label.bind('<Configure>', lambda e: display_image())

    root.mainloop()

if __name__ == '__main__':
    # Reminder for required packages
    # Need to install: pip install opencv-python numpy pillow
    setup_gui()