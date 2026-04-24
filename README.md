# Real-Time Stampede Detection & Alert Drone System 🚁

An advanced, edge-computing AI pipeline designed for aerial deployment. This system utilizes a drone's camera feed to monitor crowd dynamics in real-time, leveraging a dual-model computer vision architecture to detect sudden chaotic movements indicative of a potential stampede. Upon detection, it triggers an immediate alert to ground control.

Optimized strictly for resource-constrained edge hardware (NVIDIA Jetson Nano), this system provides critical early warnings to prevent human-wildlife or crowd-related disasters.

## 🧠 System Architecture: The Dual-Model Approach

To achieve high accuracy from varying drone altitudes while maintaining real-time inference speeds on edge hardware, this pipeline utilizes a hybrid dual-model approach, combined with algorithmic motion tracking.

### 1. The Pretrained Model (Base Localization)
The system initializes with a **pretrained YOLOv8** model. 
* **Purpose:** Acts as the foundational layer for generalized human detection. 
* **Function:** It quickly scans the aerial frames to identify individual people and establish base bounding boxes. Relying on generalized weights allows the system to boot up and recognize standard human forms immediately without cold-starting the complex analysis.

### 2. The Custom-Trained Model (Aerial & Density Optimization)
Operating in tandem is a **custom-trained deep learning model**.
* **Purpose:** Solves the "drone perspective" problem. Standard pretrained models fail when viewing humans from a top-down, 45-degree, or high-altitude perspective. 
* **Function:** This model is specifically fine-tuned on aerial drone datasets (dense crowds, varying lighting conditions, and top-down angles). It refines the bounding boxes of the pretrained model, recovers false negatives (people missed by the base model in tight clusters), and accurately segments dense crowd blobs.

### 3. Optical Flow (Motion & Anomaly Analysis)
Detection alone cannot predict a stampede; motion is the key indicator. 
* **Purpose:** Calculates pixel-level velocity and motion vectors.
* **Function:** Once the dual-model system identifies the crowd structure, the system applies **Dense Optical Flow** across the detected bounding boxes. It maps the standard flow of the crowd. If the system detects sudden, erratic, or abnormally fast directional shifts (e.g., a radial burst of movement away from a central point), it classifies the event as a panic/stampede anomaly.

## ⚙️ Pipeline Flow

1. **Video Ingestion:** Drone streams live feed to the Jetson Nano.
2. **Hybrid Detection:** Pretrained and Custom-Trained models map the crowd density and localize individuals.
3. **Vector Mapping:** Optical Flow calculates the speed and trajectory of the identified clusters.
4. **Threshold Evaluation:** If motion vectors exceed the baseline "normal walking/standing" threshold, an anomaly is flagged.
5. **Alert Generation:** System logs the event and transmits an alert signal via the communication module to the ground station.

## ⚡ Edge Hardware Optimization

Running two models plus Optical Flow is computationally expensive. This system is heavily optimized for the **NVIDIA Jetson Nano**:
* **TensorRT Optimization:** Models are exported and optimized to run directly on the Nano's GPU architecture.
* **Frame Skipping:** Intelligently skips intermediate frames for detection while maintaining Optical Flow continuity, ensuring the system hits real-time FPS targets without thermal throttling.
* **FP16 Precision:** Model weights are quantized to half-precision (FP16) to maximize memory efficiency on the edge device.

## 🛠️ Hardware & Software Requirements

### Hardware
* **NVIDIA Jetson Nano** (Primary edge compute node)
* UAV/Drone platform with an underslung camera payload
* Active cooling fan for the Jetson Nano (Mandatory for continuous inference)

### Software & Libraries
* JetPack SDK
* Python 3.x
* Ultralytics (YOLOv8)
* OpenCV (Compiled strictly with CUDA support for hardware acceleration)
* NumPy

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Drone-Stampede-Detector-Edge.git](https://github.com/yourusername/Drone-Stampede-Detector-Edge.git)
   cd Drone-Stampede-Detector-Edge
