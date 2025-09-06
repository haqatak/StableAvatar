import torch
import psutil
import argparse
import gradio as gr
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from omegaconf import OmegaConf
from wan.models.cache_utils import get_teacache_coefficients
from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from wan.models.wan_text_encoder import WanT5EncoderModel
from wan.models.wan_vae import AutoencoderKLWan
from wan.models.wan_image_encoder import CLIPModel
from wan.pipeline.wan_inference_long_pipeline import WanI2VTalkingInferenceLongPipeline
from wan.utils.fp8_optimization import replace_parameters_by_name, convert_weight_dtype_wrapper, convert_model_weight_to_float8
from wan.utils.utils import get_image_to_video_latent, save_videos_grid
import numpy as np
import librosa
import datetime
import random
import math
import subprocess
from moviepy.editor import VideoFileClip
import shutil
try:
    from audio_separator.separator import Separator
except:
    print("Vocal separation function is not available, please install audio-separator[gpu]")


parser = argparse.ArgumentParser()
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP address, change to 0.0.0.0 for LAN access")
parser.add_argument("--server_port", type=int, default=7891, help="Port to use")
parser.add_argument("--share", action="store_true", help="Enable gradio share")
parser.add_argument("--mcp_server", action="store_true", help="Enable mcp server")
args = parser.parse_args()


if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda" 
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


model_path = "checkpoints"
pretrained_model_name_or_path = f"{model_path}/Wan2.1-Fun-V1.1-1.3B-InP"
pretrained_wav2vec_path = f"{model_path}/wav2vec2-base-960h"
transformer_path = f"{model_path}/StableAvatar-1.3B/transformer3d-square.pt"
config = OmegaConf.load("deepspeed_config/wan2.1/wan_civitai.yaml")
sampler_name = "Flow"
# clip_sample_n_frames moved to UI parameter
tokenizer = AutoTokenizer.from_pretrained(config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=dtype,
)
text_encoder = text_encoder.eval()
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
)
wav2vec_processor = Wav2Vec2Processor.from_pretrained(pretrained_wav2vec_path)
wav2vec = Wav2Vec2Model.from_pretrained(pretrained_wav2vec_path).to("cpu")
clip_image_encoder = CLIPModel.from_pretrained(os.path.join(pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')), )
clip_image_encoder = clip_image_encoder.eval()
transformer3d = WanTransformer3DFantasyModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=False,
    torch_dtype=dtype,
)
if transformer_path is not None:
    state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer3d.load_state_dict(state_dict, strict=False)
Choosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler(
    **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)
pipeline = WanI2VTalkingInferenceLongPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    transformer=transformer3d,
    clip_image_encoder=clip_image_encoder,
    scheduler=scheduler,
    wav2vec_processor=wav2vec_processor,
    wav2vec=wav2vec,
)


def generate(
    GPU_memory_mode,
    teacache_threshold,
    num_skip_start_steps,
    clip_sample_n_frames,
    image_path,
    audio_path,
    prompt,
    negative_prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    text_guide_scale,
    audio_guide_scale,
    motion_frame,
    fps,
    overlap_window_length,
    seed_param,
):
    global pipeline, transformer3d
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed_param<0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    
    print("\n" + "="*80)
    print(f"[{timestamp}] Generation Configuration")
    print("="*80)
    print(f"⚙️ System Settings:")
    print(f"  - Device: {device}")
    print(f"  - Dtype: {dtype}")
    print(f"  - GPU Memory Mode: {GPU_memory_mode}")
    print(f"\n⚡ Optimization Settings:")
    print(f"  - TeaCache Enabled: {'Yes' if teacache_threshold > 0 else 'No'}")
    if teacache_threshold > 0:
        print(f"  - TeaCache Threshold: {teacache_threshold}")
        print(f"  - Skip Start Steps: {num_skip_start_steps}")
    print(f"\n🎬 Generation Parameters:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Clip Sample Frames: {clip_sample_n_frames} (→ {(clip_sample_n_frames-1)//4+1} latent frames)")
    print(f"  - Inference Steps: {num_inference_steps}")
    print(f"  - Overlap Window Length: {overlap_window_length}")
    print(f"  - Guidance Scales: text={text_guide_scale}, audio={audio_guide_scale}, overall={guidance_scale}")
    print(f"  - Motion/FPS: motion_frame={motion_frame}, fps={fps}")
    print("="*80)

    # Reset pipeline hooks before applying new mode
    # if hasattr(pipeline, '_all_hooks'):
    #     pipeline._all_hooks.clear()
    # if hasattr(pipeline, '_cpu_offload_hook'):
    #     pipeline._cpu_offload_hook = None
    
    # Clear any existing device placements
    if GPU_memory_mode == "Normal":
        # For Normal mode, ensure all components are on GPU with correct dtype
        pipeline.to(device=device, dtype=dtype)
        
        # pipeline.vae = pipeline.vae.to(device=device, dtype=dtype)
        # pipeline.text_encoder = pipeline.text_encoder.to(device=device, dtype=dtype)
        # pipeline.transformer = pipeline.transformer.to(device=device, dtype=dtype)
        # pipeline.clip_image_encoder = pipeline.clip_image_encoder.to(device=device, dtype=dtype)
        # if hasattr(pipeline, 'wav2vec'):
        #     pipeline.wav2vec = pipeline.wav2vec.to(device=device)
    elif GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer3d, ["modulation", ], device=device)
        transformer3d.freqs = transformer3d.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer3d, exclude_module_name=["modulation", ])
        convert_weight_dtype_wrapper(transformer3d, dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
        
    if teacache_threshold > 0:
        print(f"\n🚀 Enabling TeaCache acceleration...")
        coefficients = get_teacache_coefficients(pretrained_model_name_or_path)
        pipeline.transformer.enable_teacache(
            coefficients,
            num_inference_steps,
            teacache_threshold,
            num_skip_start_steps=num_skip_start_steps,
        )

    print(f"\n📊 Starting inference pipeline...")
    with torch.no_grad():
        video_length = int((clip_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if clip_sample_n_frames != 1 else 1
        input_video, input_video_mask, clip_image = get_image_to_video_latent(image_path, None, video_length=video_length, sample_size=[height, width])
        sr = 16000
        vocal_input, sample_rate = librosa.load(audio_path, sr=sr)
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            vocal_input_values=vocal_input,
            motion_frame=motion_frame,
            fps=fps,
            sr=sr,
            cond_file_path=image_path,
            overlap_window_length=overlap_window_length,
            seed=seed,
            overlapping_weight_scheme="uniform",
            clip_length=clip_sample_n_frames,  # Pass clip_length parameter
        ).videos
        os.makedirs("outputs", exist_ok=True)
        video_path = os.path.join("outputs", f"{timestamp}.mp4")
        save_videos_grid(sample, video_path, fps=fps)
        output_video_with_audio = os.path.join("outputs", f"{timestamp}_audio.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "quiet", "-i", video_path, "-i", audio_path, 
            "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", 
            output_video_with_audio
        ], check=True)
        
    return output_video_with_audio, seed, f"Generated outputs/{timestamp}.mp4"


def exchange_width_height(width, height):
    return height, width, "✅ Width and Height Swapped"


def adjust_width_height(image):
    image = load_image(image)
    width, height = image.size
    original_area = width * height
    default_area = 512*512
    ratio = math.sqrt(original_area / default_area)
    width = width / ratio // 16 * 16
    height = height / ratio // 16 * 16
    return int(width), int(height), "✅ Adjusted Size Based on Image"


def audio_extractor(video_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(f"outputs/{timestamp}.wav", codec='pcm_s16le')
    return f"outputs/{timestamp}.wav", f"Generated outputs/{timestamp}.wav"


def vocal_separation(audio_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_separator_model_file = "checkpoints/Kim_Vocal_2.onnx"
    audio_separator = Separator(
        output_dir=f"outputs/{timestamp}",
        output_single_stem="vocals",
        model_file_dir=os.path.dirname(audio_separator_model_file),
    )
    audio_separator.load_model(os.path.basename(audio_separator_model_file))
    assert audio_separator.model_instance is not None, "Fail to load audio separate model."
    outputs = audio_separator.separate(audio_path)
    vocal_audio_file = os.path.join(audio_separator.output_dir, outputs[0])
    destination_file = f"outputs/{timestamp}.wav"
    shutil.copy(vocal_audio_file, destination_file)
    os.remove(vocal_audio_file)
    return f"outputs/{timestamp}.wav", f"Generated outputs/{timestamp}.wav"




with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">StableAvatar</h2>
            </div>
            """)
    
    with gr.Accordion("Model Settings", open=False):
        with gr.Row():
            GPU_memory_mode = gr.Dropdown(
                label="GPU Memory Mode",
                info="Normal uses 25G VRAM, model_cpu_offload uses 13G VRAM",
                choices=["Normal", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                value="Normal"
            )
            teacache_threshold = gr.Slider(label="TeaCache Threshold", info="Recommended 0.1, 0 disables TeaCache acceleration", minimum=0, maximum=1, step=0.01, value=0)
            num_skip_start_steps = gr.Slider(label="Skip Start Steps", info="Recommended 5", minimum=0, maximum=100, step=1, value=5)
        with gr.Row():
            clip_sample_n_frames = gr.Slider(
                label="Clip Sample Frames",
                info="Video frames, 81=2s@25fps, 161=4s@25fps, must be 4n+1",
                minimum=41,
                maximum=321,
                step=4,
                value=81
            )
    with gr.TabItem("StableAvatar"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_path = gr.Image(label="Upload Image", type="filepath", height=280)
                    audio_path = gr.Audio(label="Upload Audio", type="filepath")
                prompt = gr.Textbox(label="Prompt", value="")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="Colorful, overexposed, static, blurry details, subtitles, style, work, painting, picture, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, still image, cluttered background, many people in the background, walking backwards")
                generate_button = gr.Button("🎬 Start Generation", variant='primary')
                with gr.Accordion("Parameter Settings", open=True):
                    with gr.Row():
                        width = gr.Slider(label="Width", minimum=256, maximum=2048, step=16, value=512)
                        height = gr.Slider(label="Height", minimum=256, maximum=2048, step=16, value=512)
                    with gr.Row():
                        exchange_button = gr.Button("🔄 Swap Width/Height")
                        adjust_button = gr.Button("Adjust Size Based on Image")
                    with gr.Row():
                        guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=6.0)
                        num_inference_steps = gr.Slider(label="Sampling Steps (Recommended 50)", minimum=1, maximum=100, step=1, value=10)
                    with gr.Row():
                        text_guide_scale = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=3.0)
                        audio_guide_scale = gr.Slider(label="Audio Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=5.0)
                    with gr.Row():
                        motion_frame = gr.Slider(label="Motion Frame", minimum=1, maximum=50, step=1, value=25)
                        fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=25)
                    with gr.Row():
                        overlap_window_length = gr.Slider(label="Overlap Window Length", minimum=1, maximum=20, step=1, value=5)
                        seed_param = gr.Number(label="Seed (positive integer, -1 for random)", value=-1)
            with gr.Column():
                info = gr.Textbox(label="Status", interactive=False)
                video_output = gr.Video(label="Generated Result", interactive=False)
                seed_output = gr.Textbox(label="Seed")
    with gr.TabItem("Audio Extraction"):
        with gr.Row():
            with gr.Column():
                video_path = gr.Video(label="Upload Video", height=500)
                extractor_button = gr.Button("🎬 Start Extraction", variant='primary')
            with gr.Column():
                info2 = gr.Textbox(label="Status", interactive=False)
                audio_output = gr.Audio(label="Generated Result", interactive=False)
    with gr.TabItem("Vocal Separation"):
        with gr.Row():
            with gr.Column():
                audio_path3 = gr.Audio(label="Upload Audio", type="filepath")
                separation_button = gr.Button("🎬 Start Separation", variant='primary')
            with gr.Column():
                info3 = gr.Textbox(label="Status", interactive=False)
                audio_output3 = gr.Audio(label="Generated Result", interactive=False)

    gr.on(
        triggers=[generate_button.click, prompt.submit, negative_prompt.submit],
        fn = generate,
        inputs = [
            GPU_memory_mode,
            teacache_threshold,
            num_skip_start_steps,
            clip_sample_n_frames,
            image_path,
            audio_path,
            prompt,
            negative_prompt,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            text_guide_scale,
            audio_guide_scale,
            motion_frame,
            fps,
            overlap_window_length,
            seed_param,
        ],
        outputs = [video_output, seed_output, info]
    )
    exchange_button.click(
        fn=exchange_width_height, 
        inputs=[width, height], 
        outputs=[width, height, info]
    )
    adjust_button.click(
        fn=adjust_width_height, 
        inputs=[image_path], 
        outputs=[width, height, info]
    )
    extractor_button.click(
        fn=audio_extractor, 
        inputs=[video_path], 
        outputs=[audio_output, info2]
    )
    separation_button.click(
        fn=vocal_separation, 
        inputs=[audio_path3], 
        outputs=[audio_output3, info3]
    )


if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )
