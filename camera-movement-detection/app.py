import streamlit as st
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import io
import zipfile
import tempfile
import os
from typing import List
import movement_detector
import time

# Sayfa yapılandırması
st.set_page_config(
    page_title="🎥 Camera Movement Detector",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile özel stil
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎥 Advanced Camera Movement Detector</h1>', unsafe_allow_html=True)

    st.markdown("""
    **This application uses advanced computer vision techniques to detect camera movements:**
    - 📊 **Frame Differencing**: Analyzes differences between consecutive frames
    - 🔍 **Feature Matching**: Uses ORB features and homography estimation
    - 🌊 **Optical Flow**: Lucas-Kanade optical flow algorithm
    - 🧠 **Smart Fusion**: Combines the results of all methods optimally
    """)

    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Sensitivity settings
        st.subheader("🎚️ Sensitivity Settings")
        diff_threshold = st.slider("Frame Difference Threshold", 10.0, 100.0, 30.0, 5.0)
        feature_threshold = st.slider("Feature Matching Threshold", 0.3, 0.9, 0.7, 0.1)
        min_match_count = st.slider("Minimum Feature Matches", 5, 50, 10, 5)

        # Visualization settings
        st.subheader("📊 Visualization")
        show_detailed_analysis = st.checkbox("Show Detailed Analysis", True)
        show_movement_frames = st.checkbox("Show Movement Frames", True)
        max_frames_to_show = st.slider("Maximum Frames to Display", 1, 50, 5)

        # Demo data
        st.subheader("🎯 Demo")
        if st.button("📥 Load Demo Data"):
            st.session_state.demo_mode = True

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Upload Files")

        # Upload options
        upload_option = st.radio(
            "Select upload type:",
            ["🖼️ Multiple Images", "📹 Video File", "📦 ZIP Archive"]
        )

        frames = []

        if upload_option == "🖼️ Multiple Images":
            uploaded_files = st.file_uploader(
                "Select image files (sequential frames)",
                type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True,
                help="Please upload files in the order of camera movement"
            )

            if uploaded_files:
                frames = load_images(uploaded_files)

        elif upload_option == "📹 Video File":
            uploaded_video = st.file_uploader(
                "Select video file",
                type=["mp4", "avi", "mov", "mkv"],
                help="Video will be automatically split into frames"
            )

            if uploaded_video:
                frames = extract_frames_from_video(uploaded_video)

        elif upload_option == "📦 ZIP Archive":
            uploaded_zip = st.file_uploader(
                "Select ZIP archive (containing images)",
                type=["zip"],
                help="All images in the ZIP will be processed in order"
            )

            if uploaded_zip:
                frames = load_images_from_zip(uploaded_zip)

    with col2:
        st.subheader("📊 Summary")
        if frames:
            st.metric("📸 Total Frames", len(frames))
            st.metric("🖼️ Resolution", f"{frames[0].shape[1]}x{frames[0].shape[0]}")
            st.metric("🎨 Number of Channels", frames[0].shape[2] if len(frames[0].shape) > 2 else 1)

    # Movement detection
    if frames:
        if len(frames) < 2:
            st.error("❌ At least 2 frames are required!")
            return

        with st.spinner("🔍 Detecting camera movement..."):
            detector = movement_detector.MovementDetector(
                diff_threshold=diff_threshold,
                feature_threshold=feature_threshold,
                min_match_count=min_match_count
            )

            result = detector.detect_significant_movement(frames)

        display_results(result, frames, show_detailed_analysis, show_movement_frames, max_frames_to_show)

    # Demo mode
    if hasattr(st.session_state, 'demo_mode') and st.session_state.demo_mode:
        show_demo()


def load_images(uploaded_files) -> List[np.ndarray]:
    """Convert uploaded image files to numpy arrays"""
    frames = []
    
    # Sort files by name
    sorted_files = sorted(uploaded_files, key=lambda x: x.name)
    
    progress_bar = st.progress(0)
    for i, uploaded_file in enumerate(sorted_files):
        try:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            
            # Convert RGBA to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            elif len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
            frames.append(frame)
            progress_bar.progress((i + 1) / len(sorted_files))
        except Exception as e:
            st.error(f"❌ Error uploading {uploaded_file.name}: {str(e)}")
    
    return frames

# def extract_frames_from_video(uploaded_video) -> List[np.ndarray]:
    """Extract frames from uploaded video file"""
    frames = []
    
    # Create a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()
    
    try:
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames if there are too many
        sample_rate = max(1, total_frames // 50)  # Maximum 50 frames
        
        progress_bar = st.progress(0)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        cap.release()
        
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
    finally:
        os.unlink(tfile.name)
    
    return frames



def extract_frames_from_video(uploaded_video) -> List[np.ndarray]:
    """Extract frames from uploaded video file"""
    frames = []
    tfile = None
    cap = None
    
    try:
        # Create a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()  # Dosyayı kapat
        
        # Video capture objesi oluştur
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("❌ Video file could not be opened!")
            return frames
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            st.error("❌ No frames found in video!")
            return frames
        
        # Sample frames if there are too many
        sample_rate = max(1, total_frames // 50)  # Maximum 50 frames
        
        progress_bar = st.progress(0)
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                processed_frames += 1
                
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        st.success(f"✅ Successfully extracted {processed_frames} frames from video!")
        
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        
    finally:
        # Video capture objesini güvenli şekilde kapat
        if cap is not None:
            cap.release()
            cap = None
        
        # Biraz bekle (Windows için gerekli)
        import time
        time.sleep(0.1)
        
        # Temporary dosyayı sil
        if tfile is not None:
            try:
                if os.path.exists(tfile.name):
                    # Dosyanın kullanımda olup olmadığını kontrol et
                    for attempt in range(5):  # 5 kez dene
                        try:
                            os.unlink(tfile.name)
                            break
                        except PermissionError:
                            if attempt < 4:  # Son denemede değilse bekle
                                time.sleep(0.2)
                            else:
                                # Son çare olarak dosyayı işaretle (sistem restart'ta silinir)
                                st.warning("⚠️ Temporary file could not be deleted immediately. It will be cleaned up on system restart.")
            except Exception as cleanup_error:
                st.warning(f"⚠️ Warning: Could not clean up temporary file: {str(cleanup_error)}")
    
    return frames




def load_images_from_zip(uploaded_zip) -> List[np.ndarray]:
    """Load images from a ZIP archive"""
    frames = []

    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist()
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            image_files.sort()  # Sort files by name

            progress_bar = st.progress(0)
            for i, filename in enumerate(image_files):
                try:
                    with zip_ref.open(filename) as file:
                        image = Image.open(io.BytesIO(file.read()))
                        frame = np.array(image)

                        if len(frame.shape) == 3 and frame.shape[2] == 4:
                            frame = frame[:, :, :3]
                        elif len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                        frames.append(frame)
                        progress_bar.progress((i + 1) / len(image_files))
                except Exception as e:
                    st.warning(f"⚠️ Skipped {filename}: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error processing ZIP file: {str(e)}")

    return frames

def display_results(result: dict, frames: List[np.ndarray],
                   show_detailed: bool, show_frames: bool, max_frames: int):
    """Visualize the detection results"""

    movement_indices = result["movement_indices"]
    scores = result["scores"]
    method_results = result["method_results"]

    # Summary results
    st.success(f"✅ Analysis complete! Significant movement detected in {len(movement_indices)} frames.")

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Movement Frames", len(movement_indices))
    with col2:
        st.metric("📊 Total Frames", len(frames))
    with col3:
        st.metric("📈 Movement Rate", f"{len(movement_indices)/len(frames)*100:.1f}%")
    with col4:
        avg_score = np.mean(scores) if scores else 0
        st.metric("⭐ Average Score", f"{avg_score:.1f}")

    if movement_indices:
        st.info(f"🎯 **Frames with detected movement:** {', '.join(map(str, movement_indices))}")
    else:
        st.info("ℹ️ No significant camera movement detected.")

    # Show charts
    if scores:
        create_movement_chart(scores, movement_indices, method_results)

    # Show detailed analysis
    if show_detailed and method_results:
        show_detailed_analysis(method_results, scores)

    # Show movement frames
    if show_frames and movement_indices:
        show_movement_frames(frames, movement_indices[:max_frames])


def create_movement_chart(scores: List[float], movement_indices: List[int], method_results: dict):
    """Create the movement score chart"""

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("🎯 Combined Movement Scores", "📊 Method Comparison"),
        vertical_spacing=0.1
    )

    # Combined scores
    frame_numbers = list(range(1, len(scores) + 1))

    fig.add_trace(
        go.Scatter(
            x=frame_numbers,
            y=scores,
            mode='lines+markers',
            name='Combined Score',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Highlight movement points
    if movement_indices:
        movement_scores = [scores[i - 1] for i in movement_indices if i - 1 < len(scores)]
        fig.add_trace(
            go.Scatter(
                x=movement_indices,
                y=movement_scores,
                mode='markers',
                name='Movement Detected',
                marker=dict(color='red', size=12, symbol='circle-open', line=dict(width=2))
            ),
            row=1, col=1
        )

    # Method-based comparison
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    methods = ['frame_diff', 'feature_matching', 'optical_flow']
    method_names = ['Frame Diff', 'Feature Match', 'Optical Flow']

    for i, (method, name) in enumerate(zip(methods, method_names)):
        if method in method_results and method_results[method]:
            fig.add_trace(
                go.Scatter(
                    x=frame_numbers,
                    y=method_results[method],
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i], width=1.5),
                    opacity=0.7
                ),
                row=2, col=1
            )

    fig.update_layout(
        height=600,
        title_text="📈 Movement Analysis Charts",
        showlegend=True
    )

    fig.update_xaxes(title_text="Frame Number")
    fig.update_yaxes(title_text="Movement Score")

    st.plotly_chart(fig, use_container_width=True)


def show_detailed_analysis(method_results: dict, scores: List[float]):
    """Display detailed analysis results"""

    st.subheader(" Detailed Method Analysis")

    # Create table
    df_data = {
        'Frame': list(range(1, len(scores) + 1)),
        'Combined Score': [f"{score:.2f}" for score in scores]
    }

    methods = ['frame_diff', 'feature_matching', 'optical_flow']
    method_names = ['Frame Diff', 'Feature Match', 'Optical Flow']

    for method, name in zip(methods, method_names):
        if method in method_results and method_results[method]:
            df_data[name] = [f"{score:.2f}" for score in method_results[method]]

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)

    # Show statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("** Frame Differencing**")
        if 'frame_diff' in method_results:
            frame_diff_scores = method_results['frame_diff']
            st.write(f"Average: {np.mean(frame_diff_scores):.2f}")
            st.write(f"Maximum: {np.max(frame_diff_scores):.2f}")
            st.write(f"Standard Deviation: {np.std(frame_diff_scores):.2f}")

    with col2:
        st.markdown("**🔍 Feature Matching**")
        if 'feature_matching' in method_results:
            feature_scores = method_results['feature_matching']
            st.write(f"Average: {np.mean(feature_scores):.2f}")
            st.write(f"Maximum: {np.max(feature_scores):.2f}")
            st.write(f"Standard Deviation: {np.std(feature_scores):.2f}")

    with col3:
        st.markdown("**🌊 Optical Flow**")
        if 'optical_flow' in method_results:
            flow_scores = method_results['optical_flow']
            st.write(f"Average: {np.mean(flow_scores):.2f}")
            st.write(f"Maximum: {np.max(flow_scores):.2f}")
            st.write(f"Standard Deviation: {np.std(flow_scores):.2f}")


def show_movement_frames(frames: List[np.ndarray], movement_indices: List[int]):
    """Display frames with detected movement"""

    st.subheader("🎯 Detected Movement Frames")

    if not movement_indices:
        st.info("No movement frames to display.")
        return

    # Show in grid
    cols_per_row = 3
    for i in range(0, len(movement_indices), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(movement_indices):
                frame_idx = movement_indices[idx]
                if frame_idx < len(frames):
                    with col:
                        st.image(
                            frames[frame_idx],
                            caption=f"Frame {frame_idx} - Movement Detected",
                            use_container_width=True
                        )


def show_demo():
    """Show Demo Data"""
    st.subheader("Demo Mode")
    st.info("Demo mode is under development. Please upload your own data to start testing!")
    st.session_state.demo_mode = False


if __name__ == "__main__":
    main()
    