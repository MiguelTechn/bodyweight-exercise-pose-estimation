import cv2 as cv
import time
import psutil
import pynvml
import pandas as pd

from pathlib import Path
from src.utils.streamer import VideoStreamer
from src.utils.visualizer import PoseVisualizer
from src.pose_estimators.mediapipe_pose.mediapipe_estimator import MediaPipePoseEstimator
from src.pose_estimators.yolo_pose.yolo_estimator import YoloPoseEstimator

def model_init(model_name):
    if model_name.lower() == "mediapipe":
        return MediaPipePoseEstimator(video_processing = 'video')
    elif model_name.lower() == "yolo":
        return YoloPoseEstimator()
    else:
        raise NotImplementedError(f"The model {model_name} is not implemented yet.")

def capture_hardware_metrics(gpu_handle, initial_consumption = {}, wait = 5):
    start_time = time.time()
    if wait > 0:
        print(f"Capture hardware metrics for {wait}s")
        cpu_usage = []
        ram_used_gb = []
        gpu_usage = []
        vram_used_gb = []
        while time.time() - start_time < wait:

            # CPU
            cpu_usage.append(psutil.cpu_percent(interval=None))
            
            # RAM
            ram_used_gb.append(psutil.virtual_memory().used / (1024**3))
            
            try:
                # GPU
                gpu_usage.append(pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu)
            except pynvml.NVMLError:
                gpu_usage.append(0.0)
                
            try:
                # VRAM
                vram_used_gb.append(float(pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used) / (1024**3))
            except pynvml.NVMLError:
                vram_used_gb.append(0.0)
        
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
        avg_ram = sum(ram_used_gb) / len(ram_used_gb)
        avg_gpu = sum(gpu_usage) / len(gpu_usage)
        avg_vram = sum(vram_used_gb) / len(vram_used_gb)
        
        return {
            'cpu_%': avg_cpu,
            'ram_gb': avg_ram,
            'gpu_%': avg_gpu,
            'vram_gb': avg_vram
        }
    else:
         # CPU
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # RAM
        ram_used_gb = psutil.virtual_memory().used / (1024**3)
        
        try:
            # GPU
            gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        except pynvml.NVMLError:
            gpu_usage = 0.0
            # 
        try:
            # VRAM
            vram_used_gb = float(pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used) / (1024**3)
        except pynvml.NVMLError:
            vram_used_gb = 0.0

        # Subtract the resources consumed idle
        return {
            'cpu_%': cpu_usage - initial_consumption['cpu_%'] if initial_consumption else cpu_usage,
            'ram_gb': ram_used_gb - initial_consumption['ram_gb'] if initial_consumption else ram_used_gb,
            'gpu_%': gpu_usage - initial_consumption['gpu_%'] if initial_consumption else gpu_usage,
            'vram_gb': vram_used_gb - initial_consumption['vram_gb'] if initial_consumption else vram_used_gb
        }
    
def main():

    # GPU monitor inicialization
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # Initial comsumption calculations
    initial_consumption = capture_hardware_metrics(gpu_handle)

    # List to storage results
    inference_results = []
    fps_results = []
    hardware_results = []

    cpu_usage = []
    ram_used_gb = []
    gpu_usage = []
    vram_used_gb = []

    # Path to storage results
    final_results_path = Path(__file__).parent.parent.parent / 'results' / 'phase_1'
    final_results_path.mkdir(parents=True, exist_ok=True)

    models_test = ["mediapipe", "yolo"]

    #To get the path to video test
    test_video_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'fase_1_test'

    for video in test_video_path.iterdir():        
        if video.suffix.lower() in ['.mp4', '.avi']:
            # Video inicialization
            videoStreamer = VideoStreamer(video)
            
            #Original video fps and frames number
            video_fps = videoStreamer.get_fps()     
            total_frames = videoStreamer.get_total_frames()

            for model_name in models_test:
                # Model inicialization
                with model_init(model_name) as ia:
                    print(f"Performance test of {model_name} model")
                    
                    # To calculate fps
                    fps_time_start = time.perf_counter()

                    while True:
                        frame = videoStreamer.get_frame()
                        if frame is None: break                    
                        
                        # Timestamp for mediapipe
                        timestamp_ms = int(videoStreamer.get_timestamp())
                        
                        # Set color format to RGB
                        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        
                        inference_time_start = time.perf_counter()

                        # Frame processing
                        results = ia.process_frame(frame=rgb_frame, timestamp_ms=timestamp_ms)
                        
                        # To calculate inference time
                        inference_time_end = time.perf_counter()

                        # Get frame number
                        frame_position = int(videoStreamer.get_frame_position())

                        if frame_position % 10 == 0:
                            # Resource consumption
                            procesing_consumption = capture_hardware_metrics(gpu_handle, initial_consumption, 0)

                            cpu_usage.append(procesing_consumption['cpu_%'])
                            ram_used_gb.append(procesing_consumption['ram_gb'])
                            gpu_usage.append(procesing_consumption['gpu_%'])
                            vram_used_gb.append(procesing_consumption['vram_gb'])

                        inference_time_ms = (inference_time_end - inference_time_start) * 1000
            
                        inference_results.append({
                            "model_name": model_name,
                            "video": video.name,
                            "frame_number": frame_position,
                            "inference_time_ms": inference_time_ms
                        })
                    
                    # Fps calcultations
                    fps_time_end = time.perf_counter()
                    time_elapsed = fps_time_end - fps_time_start
                    fps_process = total_frames / time_elapsed
                    fps_results.append({
                        "model_name": model_name,
                        "video": video.name,
                        "fps_video": video_fps,
                        "total_frames": total_frames,
                        "time_elapsed": time_elapsed,
                        "fps_process": fps_process
                    })

                    # Avg resource consumption by video
                    hardware_results.append({
                        "model_name": model_name,
                        "video": video.name,
                        "cpu_%": sum(cpu_usage) / len(cpu_usage),
                        "ram_gb": sum(ram_used_gb) / len(ram_used_gb),
                        "gpu_%": sum(gpu_usage) / len(gpu_usage),
                        "vram_gb": sum(vram_used_gb) / len(vram_used_gb)
                    })
                    
                    cpu_usage = []
                    ram_used_gb = []
                    gpu_usage = []
                    vram_used_gb = []
                    
                    # Rewind to the beginning of the video
                    videoStreamer.video_rewind()

            videoStreamer.close()
    
    # Shutdown GPU monitor
    pynvml.nvmlShutdown()

    inference_results_dataframe = pd.DataFrame(inference_results)
    inference_results_dataframe.to_csv(final_results_path / 'results_inference.csv', index=False)
    
    fps_results_dataframe = pd.DataFrame(fps_results)
    fps_results_dataframe.to_csv(final_results_path / 'results_fps.csv', index=False)

    hardware_results_dataframe = pd.DataFrame(hardware_results)
    hardware_results_dataframe.to_csv(final_results_path / 'results_hardware.csv', index=False)

if __name__ == "__main__":
    main()