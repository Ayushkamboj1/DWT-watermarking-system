# DWT-watermarking-system
This application is a desktop tool for embedding invisible watermarks into images using the Discrete Wavelet Transform (DWT). It provides a graphical user interface (GUI) to hide a watermark within a host image, making it imperceptible to the human eye while being robust enough for copyright protection.

Features
Invisible Watermarking: Embeds watermarks without visibly altering the host image.
DWT-Based Algorithm: Uses the robust Discrete Wavelet Transform for embedding, ensuring the watermark is resilient to common image manipulations.
Adaptive Embedding: Automatically adjusts the watermark's strength based on the host image's characteristics to balance invisibility and durability.
User-Friendly GUI: A simple, tabbed interface built with Tkinter for easy navigation between embedding and detection tasks.
Broad Image Compatibility: Automatically converts various image formats (like PNG, BMP, TIFF) to JPG for processing.
Default Watermark: Includes a pre-generated default watermark for quick testing.
Technology Stack
Programming Language: Python
GUI Framework: Tkinter (ttk for modern widgets)
Core Libraries:
PyWavelets (pywt): For performing the Discrete Wavelet Transform.
OpenCV (cv2): For image processing tasks like reading, writing, and color space conversions.
Pillow (PIL): For handling image display within the GUI and format conversions.
NumPy: For efficient numerical operations on image arrays.
Installation
Clone the repository from GitHub.
Navigate to the project directory.
Install the dependencies using pip:
Bash

pip install Pillow opencv-python numpy PyWavelets
Usage
Run the application by executing the main Python script.
To Embed a Watermark:
In the "Embed Watermark" tab, browse for and select your host image.
Browse for and select your watermark image.
Specify an output path to save the final image.
Click "Embed Watermark". The original and watermarked images will be displayed side-by-side.
To Detect a Watermark:
Navigate to the "Detect Watermark" tab.
Browse for and select the image you want to check.
Click "Detect Watermark". The application will show the result.
Note: The current detection feature is a placeholder and only confirms if the image matches the one just watermarked in the same session.

Future Enhancements
Implement a full DWT extractor: Develop a proper detection algorithm that can extract and verify a watermark from any image, rather than relying on a simple comparison.
Add Batch Processing: Allow users to embed watermarks in multiple images at once.
Increase User Control: Let users select the wavelet type (e.g., db1, bior1.3) and the DWT level.
Performance Metrics: Display the Peak Signal-to-Noise Ratio (PSNR) to show the quality of the watermarked image.
Robustness Testing: Integrate tools to test the watermark's resilience against attacks like compression, noise, and cropping.
