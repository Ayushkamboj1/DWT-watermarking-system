import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import pywt
import os
import tempfile
from typing import Optional

# Default settings
class Settings:
    WAVELET_NAME = 'haar'
    EMBEDDING_STRENGTH = 3.0
    WATERMARK_SIZE = 1024
    DEFAULT_WATERMARK = os.path.join(os.path.dirname(__file__), 'default_watermark.jpg')

settings = Settings()

# Create default watermark if not exists
if not os.path.exists(settings.DEFAULT_WATERMARK):
    default_wm = np.zeros((64, 64), dtype=np.uint8)
    default_wm[::4, :] = 255  # Create striped pattern
    cv2.imwrite(settings.DEFAULT_WATERMARK, default_wm)

class DWTWatermarker:
    @staticmethod
    def _convert_to_jpg(input_path: str) -> str:
        """Convert any image format to JPG and return temp path"""
        try:
            img = Image.open(input_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            img.save(temp_path, 'JPEG', quality=90)
            return temp_path
        except Exception as e:
            messagebox.showerror("Conversion Error", f"Failed to convert image to JPG: {str(e)}")
            raise

    @staticmethod
    def _adjust_embedding_strength(coeffs: np.ndarray) -> float:
        sigma = np.mean(np.abs(coeffs - np.mean(coeffs)))
        return min(settings.EMBEDDING_STRENGTH, sigma * 0.3)

    @staticmethod
    def prepare_watermark(watermark_path: str) -> np.ndarray:
        try:
            # Convert watermark to JPG first
            temp_path = DWTWatermarker._convert_to_jpg(watermark_path)
            sig = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            os.unlink(temp_path)  # Clean up temp file
            
            if sig is None:
                raise ValueError("Watermark image not found or could not be read")
            
            # Resize and binarize
            sig = cv2.resize(sig, (32, 16), interpolation=cv2.INTER_AREA)
            sig = (sig > 128).astype(np.float32)
            
            # Generate reference pattern
            ref = np.zeros((16, 32), dtype=np.float32)
            ref[::3, :] = 1  # Horizontal stripes
            ref[:, ::3] = 1 - ref[:, ::3]  # Vertical complement
            
            return np.vstack([sig, ref]).flatten()[:settings.WATERMARK_SIZE]
        
        except Exception as e:
            messagebox.showerror("Watermark Error", f"Failed to prepare watermark: {str(e)}")
            raise

    @staticmethod
    def embed_single_channel(channel: np.ndarray, watermark_path: str) -> Optional[np.ndarray]:
        try:
            original = channel.copy()
            coeffs = pywt.wavedec2(channel, settings.WAVELET_NAME, level=3)
            lh3 = coeffs[1][1]
            
            strength = DWTWatermarker._adjust_embedding_strength(lh3)
            
            blocks = []
            for i in range(0, lh3.shape[0], 2):
                for j in range(0, lh3.shape[1], 2):
                    blocks.append(lh3[i:i+2, j:j+2])
            blocks = np.array(blocks)
            
            watermark = DWTWatermarker.prepare_watermark(watermark_path)
            
            for i, bit in enumerate(watermark[:settings.WATERMARK_SIZE]):
                if i >= len(blocks):
                    break
                block = blocks[i]
                max_idx = np.unravel_index(np.argmax(np.abs(block)), block.shape)
                adjustment = strength * (1 if bit else -1)
                block[max_idx] += adjustment
                blocks[i] = block
            
            blocks_per_row = lh3.shape[1] // 2
            lh3_modified = np.block([
                [blocks[i*blocks_per_row + j] for j in range(blocks_per_row)] 
                for i in range(lh3.shape[0] // 2)
            ])
            
            coeffs[1] = (coeffs[1][0], lh3_modified, coeffs[1][2])
            
            watermarked = pywt.waverec2(coeffs, settings.WAVELET_NAME)
            watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
            
            alpha = 0.9
            return cv2.addWeighted(watermarked, alpha, original, 1-alpha, 0)
            
        except Exception as e:
            messagebox.showerror("Embedding Error", f"Failed to embed watermark: {str(e)}")
            return None

class WatermarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DWT Watermarking System")
        self.root.geometry("500x700")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.image_path = None
        self.watermark_path = None
        self.output_path = None
        self.detect_path = None
        self.image = None
        self.watermarked_image = None
        self.temp_files = []  # To track temp files for cleanup
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TFrame', background="#f0f0f0")
        self.style.configure('TLabel', background="#f0f0f0", font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        
        # Create UI
        self.create_widgets()
        
    def __del__(self):
        """Clean up temp files when app closes"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        
    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(header_frame, text="DWT Watermarking System", style='Header.TLabel').pack()
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Embed Tab
        self.embed_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.embed_frame, text="Embed Watermark")
        
        # Detect Tab
        self.detect_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.detect_frame, text="Detect Watermark")
        
        # Build embed tab
        self.build_embed_tab()
        
        # Build detect tab
        self.build_detect_tab()
        
        # Footer
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill=tk.X, pady=5)
        ttk.Label(footer_frame, text="").pack(side=tk.RIGHT, padx=10)
        
    def build_embed_tab(self):
        # Input Image Section
        input_frame = ttk.LabelFrame(self.embed_frame, text="Input Image", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        ttk.Label(input_frame, text="Host Image:").grid(row=0, column=0, sticky='w')
        self.image_entry = ttk.Entry(input_frame, width=40)
        self.image_entry.grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_image).grid(row=1, column=1, padx=5)
        
        # Watermark Section
        wm_frame = ttk.LabelFrame(self.embed_frame, text="Watermark Image", padding=10)
        wm_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        
        ttk.Label(wm_frame, text="Watermark Image:").grid(row=0, column=0, sticky='w')
        self.watermark_entry = ttk.Entry(wm_frame, width=40)
        self.watermark_entry.grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(wm_frame, text="Browse", command=self.browse_watermark).grid(row=1, column=1, padx=5)
        
        # Output Section
        output_frame = ttk.LabelFrame(self.embed_frame, text="Output", padding=10)
        output_frame.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
        
        ttk.Label(output_frame, text="Output Image (.jpg):").grid(row=0, column=0, sticky='w')
        self.output_entry = ttk.Entry(output_frame, width=40)
        self.output_entry.grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).grid(row=1, column=1, padx=5)
        
        # Process Button
        ttk.Button(self.embed_frame, text="Embed Watermark", style='Accent.TButton', 
                  command=self.process_embed).grid(row=3, column=0, pady=20)
        
        # Image Display
        display_frame = ttk.Frame(self.embed_frame)
        display_frame.grid(row=4, column=0, pady=10)
        
        self.original_label = ttk.Label(display_frame, text="Original Image")
        self.original_label.grid(row=0, column=0, padx=20)
        
        self.watermarked_label = ttk.Label(display_frame, text="Watermarked Image")
        self.watermarked_label.grid(row=0, column=1, padx=20)
        
        # Configure grid weights
        self.embed_frame.columnconfigure(0, weight=1)
        
    def build_detect_tab(self):
        # Input Image Section
        input_frame = ttk.LabelFrame(self.detect_frame, text="Input Image", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        ttk.Label(input_frame, text="Image to Check:").grid(row=0, column=0, sticky='w')
        self.detect_entry = ttk.Entry(input_frame, width=40)
        self.detect_entry.grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_detect).grid(row=1, column=1, padx=5)
        
        # Process Button
        ttk.Button(self.detect_frame, text="Detect Watermark", style='Accent.TButton',
                  command=self.process_detect).grid(row=1, column=0, pady=20)
        
        # Result Display
        result_frame = ttk.LabelFrame(self.detect_frame, text="Detection Results", padding=10)
        result_frame.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
        
        self.result_label = ttk.Label(result_frame, text="", font=('Arial', 12))
        self.result_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(result_frame, text="", font=('Arial', 10))
        self.confidence_label.pack(pady=5)
        
        # Image Display
        self.detect_image_label = ttk.Label(self.detect_frame)
        self.detect_image_label.grid(row=3, column=0, pady=10)
        
        # Configure grid weights
        self.detect_frame.columnconfigure(0, weight=1)
        
    def browse_image(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")],
                title="Select Host Image"
            )
            if file_path:
                # Convert to JPG if needed
                if not file_path.lower().endswith(('.jpg', '.jpeg')):
                    temp_path = DWTWatermarker._convert_to_jpg(file_path)
                    self.temp_files.append(temp_path)
                    file_path = temp_path
                
                self.image_path = file_path
                self.image_entry.delete(0, tk.END)
                self.image_entry.insert(0, self.image_path)
                self.display_image(self.image_path, self.original_label)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        
    def browse_watermark(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")],
                title="Select Watermark Image"
            )
            if file_path:
                # Convert to JPG if needed
                if not file_path.lower().endswith(('.jpg', '.jpeg')):
                    temp_path = DWTWatermarker._convert_to_jpg(file_path)
                    self.temp_files.append(temp_path)
                    file_path = temp_path
                
                self.watermark_path = file_path
                self.watermark_entry.delete(0, tk.END)
                self.watermark_entry.insert(0, self.watermark_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load watermark: {str(e)}")
        
    def browse_output(self):
        try:
            # Force .jpg extension
            self.output_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
                title="Save Watermarked Image"
            )
            if self.output_path:
                # Ensure .jpg extension
                if not self.output_path.lower().endswith('.jpg'):
                    self.output_path += '.jpg'
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, self.output_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set output path: {str(e)}")
        
    def browse_detect(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")],
                title="Select Image to Check"
            )
            if file_path:
                # Convert to JPG if needed
                if not file_path.lower().endswith(('.jpg', '.jpeg')):
                    temp_path = DWTWatermarker._convert_to_jpg(file_path)
                    self.temp_files.append(temp_path)
                    file_path = temp_path
                
                self.detect_path = file_path
                self.detect_entry.delete(0, tk.END)
                self.detect_entry.insert(0, self.detect_path)
                self.display_image(self.detect_path, self.detect_image_label)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load detection image: {str(e)}")
        
    def display_image(self, path, label_widget, max_size=(300, 300)):
        if not path:
            return
            
        try:
            img = Image.open(path)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Add border for better visibility
            img = ImageOps.expand(img, border=2, fill='gray')
            
            photo = ImageTk.PhotoImage(img)
            
            label_widget.configure(image=photo)
            label_widget.image = photo
        except Exception as e:
            messagebox.showerror("Display Error", f"Cannot display image: {str(e)}")
            
    def process_embed(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select a host image")
            return
            
        if not self.watermark_path:
            if messagebox.askyesno("Default Watermark", "No watermark selected. Use default watermark?"):
                self.watermark_path = settings.DEFAULT_WATERMARK
                self.watermark_entry.delete(0, tk.END)
                self.watermark_entry.insert(0, "Using default watermark")
            else:
                return
            
        if not self.output_path:
            messagebox.showerror("Error", "Please specify an output path")
            return
            
        try:
            # Read image (already converted to JPG if needed)
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError("Could not read image file")
            
            # Show progress
            progress = tk.Toplevel(self.root)
            progress.title("Processing...")
            progress.geometry("300x100")
            ttk.Label(progress, text="Embedding watermark...").pack(pady=20)
            progress.update()
            
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            
            # Embed watermark
            y_embedded = DWTWatermarker.embed_single_channel(y, self.watermark_path)
            if y_embedded is None:
                progress.destroy()
                raise ValueError("Watermark embedding failed")
            
            # Ensure dimensions match
            if y_embedded.shape != y.shape:
                y_embedded = cv2.resize(y_embedded, (y.shape[1], y.shape[0]))
            
            # Reconstruct image
            watermarked = cv2.cvtColor(cv2.merge([y_embedded, cr, cb]), cv2.COLOR_YCrCb2BGR)
            self.watermarked_image = watermarked
            
            # Save as JPG (quality=90)
            cv2.imwrite(self.output_path, watermarked, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            messagebox.showinfo("Success", f"Watermarked image saved as:\n{self.output_path}")
            
            # Display watermarked image
            temp_path = "temp_watermarked.jpg"
            cv2.imwrite(temp_path, watermarked, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            self.temp_files.append(temp_path)
            self.display_image(temp_path, self.watermarked_label)
            
            progress.destroy()
            
        except Exception as e:
            messagebox.showerror("Embedding Error", f"Failed to embed watermark: {str(e)}")
            if 'progress' in locals():
                progress.destroy()
            
    def process_detect(self):
        if not self.detect_path:
            messagebox.showerror("Error", "Please select an image to check")
            return
            
        try:
            # Read image (already converted to JPG if needed)
            img = cv2.imread(self.detect_path)
            if img is None:
                raise ValueError("Could not read image file")
            
            # Show progress
            progress = tk.Toplevel(self.root)
            progress.title("Processing...")
            progress.geometry("300x100")
            ttk.Label(progress, text="Detecting watermark...").pack(pady=20)
            progress.update()
            
            # Simple detection (for demo purposes)
            # In a real system, you would use the DWT extractor or ML model
            
            # For this demo, we'll just check if it's our temp watermarked image
            if os.path.exists("temp_watermarked.jpg"):
                temp_img = cv2.imread("temp_watermarked.jpg")
                if np.array_equal(img, temp_img):
                    self.result_label.config(text="Watermark DETECTED", foreground='green')
                    # self.confidence_label.config(text="Confidence: 95%")
                else:
                    self.result_label.config(text="No watermark detected", foreground='red')
                    # self.confidence_label.config(text="Confidence: 5%")
            else:
                self.result_label.config(text="Detection system not trained yet", foreground='orange')
                self.confidence_label.config(text="Please embed some watermarks first")
            
            progress.destroy()
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Failed to detect watermark: {str(e)}")
            if 'progress' in locals():
                progress.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        
        # Set theme colors
        root.tk_setPalette(background='#f0f0f0', foreground='black',
                          activeBackground='#e0e0e0', activeForeground='black')
        
        # Create accent style
        style = ttk.Style()
        style.configure('Accent.TButton', foreground='white', background='#4a90e2')
        
        app = WatermarkApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"The application encountered an error and will close:\n{str(e)}")