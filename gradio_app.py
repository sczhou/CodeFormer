import gradio as gr
import cv2
import numpy as np
import os
import sys
import torch
import argparse
from PIL import Image
import tempfile
from pathlib import Path
import subprocess
import shutil

# Add the CodeFormer project root to Python path
sys.path.insert(0, '.')

try:
    from basicsr.utils import imwrite, img2tensor, tensor2img
    from basicsr.utils.download_util import load_file_from_url
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from facelib.utils.misc import is_gray
    from torchvision.transforms.functional import normalize
    from basicsr.archs.codeformer_arch import CodeFormer
except ImportError:
    print("Error: Make sure you have installed CodeFormer and its dependencies")
    print("Follow the installation instructions from: https://github.com/sczhou/CodeFormer")
    sys.exit(1)

class CodeFormerInterface:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.codeformer_net = None
        self.face_helper = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize CodeFormer models and face helper"""
        try:
            # Load CodeFormer model
            model_path = 'weights/CodeFormer/codeformer.pth'
            if not os.path.exists(model_path):
                print("Downloading CodeFormer model...")
                os.makedirs('weights/CodeFormer', exist_ok=True)
                load_file_from_url(
                    url='https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
                    model_dir='weights/CodeFormer',
                    progress=True,
                    file_name='codeformer.pth'
                )
            
            # Initialize CodeFormer network
            self.codeformer_net = CodeFormer(
                dim_embd=512, 
                codebook_size=1024, 
                n_head=8, 
                n_layers=9,
                connect_list=['32', '64', '128', '256']
            ).to(self.device)
            
            checkpoint = torch.load(model_path, map_location=self.device)['params_ema']
            self.codeformer_net.load_state_dict(checkpoint)
            self.codeformer_net.eval()
            
            # Initialize face helper
            self.face_helper = FaceRestoreHelper(
                1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=self.device
            )
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        
        return True
    
    def restore_face(self, image, fidelity_weight=0.5, has_aligned=False, 
                    only_center_face=False, background_enhance=False, face_upsample=False):
        """
        Restore faces in the input image
        
        Args:
            image: PIL Image or numpy array
            fidelity_weight: Balance between quality and fidelity (0-1)
            has_aligned: Whether the input is already cropped and aligned
            only_center_face: Only restore the center face
            background_enhance: Enhance background with Real-ESRGAN
            face_upsample: Upsample face with Real-ESRGAN
        """
        if self.codeformer_net is None:
            return None, "Models not loaded properly"
        
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                input_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                input_img = image
            
            h, w = input_img.shape[0:2]
            
            if has_aligned:
                # For aligned faces, directly process
                cropped_face = cv2.resize(input_img, (512, 512), interpolation=cv2.INTER_LINEAR)
                
                # Prepare input tensor
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.codeformer_net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                
                # Convert back to RGB for display
                restored_face = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)
                return restored_face, "Face restoration completed successfully!"
            
            else:
                # For whole images, detect and restore faces
                self.face_helper.clean_all()
                self.face_helper.read_image(input_img)
                
                # Get face landmarks
                num_det_faces = self.face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, 
                    resize=640, 
                    eye_dist_threshold=5
                )
                print(f'Detected {num_det_faces} faces')
                
                # Align and crop faces
                self.face_helper.align_warp_face()
                
                # Face restoration
                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                    # Prepare input tensor
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.codeformer_net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        del output
                        torch.cuda.empty_cache()
                    
                    self.face_helper.add_restored_face(restored_face, cropped_face)
                
                # Paste faces back to original image
                self.face_helper.get_inverse_affine(None)
                
                # If no face detected
                if len(self.face_helper.cropped_faces) == 0:
                    restored_img = input_img
                else:
                    restored_img = self.face_helper.paste_faces_to_input_image()
                
                # Convert back to RGB for display
                restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                
                return restored_img, f"Restoration completed! Processed {num_det_faces} face(s)."
                
        except Exception as e:
            return None, f"Error during restoration: {str(e)}"

# Initialize the CodeFormer interface
codeformer_interface = CodeFormerInterface()

def process_image(image, fidelity_weight, has_aligned, only_center_face, background_enhance, face_upsample):
    """Wrapper function for Gradio interface"""
    if image is None:
        return None, "Please upload an image first."
    
    result_img, message = codeformer_interface.restore_face(
        image=image,
        fidelity_weight=fidelity_weight,
        has_aligned=has_aligned,
        only_center_face=only_center_face,
        background_enhance=background_enhance,
        face_upsample=face_upsample
    )
    
    return result_img, message

def process_video(video_path, fidelity_weight, background_enhance, face_upsample):
    """Process video file (placeholder - requires additional implementation)"""
    if video_path is None:
        return None, "Please upload a video first."
    
    # This is a simplified placeholder for video processing
    # Full implementation would require frame-by-frame processing
    return None, "Video processing feature requires additional implementation with ffmpeg integration."

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="CodeFormer - AI Face Restoration", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎭 CodeFormer - AI Face Restoration
        
        **Robust face restoration algorithm for old photos and AI-generated faces**
        
        Upload an image with faces to enhance their quality while preserving identity. 
        The app can also upscale images and restore various types of facial degradations.
        
        📝 **Instructions:**
        - Upload your image using the file uploader
        - Adjust the fidelity weight (lower = higher quality, higher = more faithful to original)
        - Configure other options as needed
        - Click "Restore Faces" to process
        """)
        
        with gr.Tab("📸 Image Restoration"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=400
                    )
                    
                    with gr.Accordion("⚙️ Settings", open=True):
                        fidelity_weight = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.1,
                            label="Fidelity Weight",
                            info="Balance between quality (0) and fidelity (1)"
                        )
                        
                        has_aligned = gr.Checkbox(
                            label="Pre-aligned Face",
                            value=False,
                            info="Check if input is already cropped and aligned face (512x512)"
                        )
                        
                        only_center_face = gr.Checkbox(
                            label="Only Center Face",
                            value=False,
                            info="Process only the center/largest face"
                        )
                        
                        background_enhance = gr.Checkbox(
                            label="Background Enhancement",
                            value=False,
                            info="Enhance background with Real-ESRGAN (requires additional setup)"
                        )
                        
                        face_upsample = gr.Checkbox(
                            label="Face Upsampling",
                            value=False,
                            info="Further upsample faces with Real-ESRGAN (requires additional setup)"
                        )
                    
                    restore_btn = gr.Button("🚀 Restore Faces", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Restored Image",
                        height=400
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        lines=3,
                        interactive=False
                    )
            
        with gr.Tab("🎬 Video Restoration"):
            gr.Markdown("### Video face restoration (Feature in development)")
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label="Input Video")
                    video_fidelity = gr.Slider(0, 1, 0.5, label="Fidelity Weight")
                    video_bg_enhance = gr.Checkbox(label="Background Enhancement")
                    video_face_upsample = gr.Checkbox(label="Face Upsampling")
                    video_btn = gr.Button("Process Video", variant="primary")
                
                with gr.Column():
                    output_video = gr.Video(label="Restored Video")
                    video_status = gr.Textbox(label="Status", lines=3)
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About CodeFormer
            
            CodeFormer is a robust face restoration method that uses a Transformer-based architecture 
            with learned discrete codebook priors for high-quality face restoration.
            
            ### Key Features:
            - **Face Restoration**: Enhance old, blurry, or low-quality face images
            - **Identity Preservation**: Maintain facial identity while improving quality
            - **Flexible Control**: Adjust quality vs. fidelity trade-off
            - **Multiple Modes**: Support for aligned faces and whole images
            
            ### Technical Details:
            - **Model**: Transformer-based with VQ-GAN codebook
            - **Input Size**: 512x512 for aligned faces, flexible for whole images
            - **Device**: CUDA GPU recommended for faster processing
            
            ### Citation:
            ```
            @inproceedings{zhou2022codeformer,
                title={Towards Robust Blind Face Restoration with Codebook Lookup TransFormer},
                author={Zhou, Shangchen and Chan, Kelvin C.K. and Li, Chongyi and Loy, Chen Change},
                booktitle={NeurIPS},
                year={2022}
            }
            ```
            
            ### Repository:
            [GitHub - sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)
            """)
        
        # Connect button events
        restore_btn.click(
            fn=process_image,
            inputs=[input_image, fidelity_weight, has_aligned, only_center_face, background_enhance, face_upsample],
            outputs=[output_image, status_text]
        )
        
        video_btn.click(
            fn=process_video,
            inputs=[input_video, video_fidelity, video_bg_enhance, video_face_upsample],
            outputs=[output_video, video_status]
        )
    
    return interface

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting CodeFormer Gradio Interface...")
    print("📋 Make sure you have:")
    print("   ✓ Installed CodeFormer and dependencies")
    print("   ✓ Downloaded pretrained models")
    print("   ✓ CUDA GPU (recommended)")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        inbrowser=True
    )
