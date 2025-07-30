import streamlit as st
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import requests
from io import BytesIO
from PIL import Image
import pretty_midi
from music21 import converter, note, chord
import time
import random
import base64
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🎼 Composer Prediction App",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
DATA_DIR = "data/midiclassics"
MODEL_PATH = "data/model/composer_classification_model_best.keras"
ARTIFACTS_PATH = "data/model/composer_classification_model_artifacts.pkl"

# Composer information with Wikipedia images
COMPOSER_INFO = {
    "Bach": {
        "full_name": "Johann Sebastian Bach",
        "years": "1685-1750",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/6/6a/Johann_Sebastian_Bach.jpg",
        "description": "German composer and musician of the Baroque period, known for complex fugues and mathematical precision."
    },
    "Beethoven": {
        "full_name": "Ludwig van Beethoven", 
        "years": "1770-1827",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/6/6f/Beethoven.jpg",
        "description": "German composer and pianist who bridged Classical and Romantic periods, famous for his nine symphonies."
    },
    "Mozart": {
        "full_name": "Wolfgang Amadeus Mozart",
        "years": "1756-1791", 
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/1/1e/Wolfgang-amadeus-mozart_1.jpg",
        "description": "Austrian composer of the Classical period, known for his prodigious talent and elegant compositions."
    },
    "Chopin": {
        "full_name": "Frédéric Chopin",
        "years": "1810-1849",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/e8/Frederic_Chopin_photo.jpeg",
        "description": "Polish composer and virtuoso pianist of the Romantic period, famous for his piano compositions."
    }
}

# --- CACHE FUNCTIONS ---
@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model and artifacts"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at {MODEL_PATH}")
            return None, None
            
        model = keras.models.load_model(MODEL_PATH)
        
        # For artifacts, we'll create a simple mapping since the pickle might have custom classes
        composer_names = ["Bach", "Beethoven", "Chopin", "Mozart"]
        artifacts = {"composer_names": composer_names}
        
        return model, artifacts
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def get_midi_files_for_composer(composer):
    """Get list of MIDI files for a specific composer"""
    folder = os.path.join(DATA_DIR, composer)
    if not os.path.exists(folder):
        return []
    
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith(('.mid', '.midi')):
            files.append(os.path.join(folder, f))
    return files

@st.cache_data
def get_all_composers():
    """Get list of available composers"""
    if not os.path.exists(DATA_DIR):
        return []
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

@st.cache_data
def load_composer_image(composer):
    """Load composer image from Wikipedia"""
    try:
        url = COMPOSER_INFO[composer]["image_url"]
        response = requests.get(url, timeout=10)
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.warning(f"Could not load image for {composer}: {str(e)}")
        return None

# --- FEATURE EXTRACTION ---
def extract_features_for_model(midi_path):
    """Extract features from MIDI file matching the model's expected input format"""
    try:
        midi = converter.parse(midi_path)
        
        # Initialize feature arrays
        musical_features = []  # 17 features
        harmonic_features = []  # 15 features  
        note_sequence = []     # 100 sequence features
        
        # Basic counts
        all_notes = list(midi.flat.notes)
        total_notes = len(all_notes)
        total_duration = float(midi.duration.quarterLength) if midi.duration else 0
        
        # Musical features (17 features expected)
        musical_features.append(total_notes)
        musical_features.append(total_duration)
        
        # Tempo
        tempo_markings = midi.flat.getElementsByClass('MetronomeMark')
        tempo = float(tempo_markings[0].number) if tempo_markings else 120.0
        musical_features.append(tempo)
        
        # Pitch features
        pitches = []
        durations = []
        for n in all_notes[:100]:  # Limit for speed
            if isinstance(n, note.Note):
                pitches.append(n.pitch.midi)
                durations.append(float(n.duration.quarterLength))
            elif isinstance(n, chord.Chord):
                pitches.append(n.root().midi)
                durations.append(float(n.duration.quarterLength))
        
        if pitches:
            pitches_array = np.array(pitches)
            musical_features.append(float(np.mean(pitches_array)))  # avg_pitch
            musical_features.append(float(np.ptp(pitches_array)))   # pitch_range
            musical_features.append(float(np.std(pitches_array)))   # pitch_std
            
            # Intervals
            if len(pitches) > 1:
                intervals = np.abs(np.diff(pitches_array))
                musical_features.append(float(np.mean(intervals)))  # avg_interval
                musical_features.append(float(np.std(intervals)))   # interval_std
            else:
                musical_features.extend([0.0, 0.0])
        else:
            musical_features.extend([60.0, 0.0, 0.0, 0.0, 0.0])
        
        # Duration features
        if durations:
            dur_array = np.array(durations)
            musical_features.append(float(np.mean(dur_array)))  # avg_duration
            musical_features.append(float(np.std(dur_array)))   # duration_std
        else:
            musical_features.extend([1.0, 0.0])
        
        # Additional musical features
        musical_features.append(len(midi.parts) if hasattr(midi, 'parts') else 1)  # num_parts
        
        # Count chords and rests
        limited_elements = list(midi.flat.notesAndRests)[:50]
        chords = [el for el in limited_elements if isinstance(el, chord.Chord)]
        rests = [el for el in limited_elements if isinstance(el, note.Rest)]
        
        musical_features.append(len(chords))  # num_chords
        musical_features.append(len(rests) / total_notes if total_notes > 0 else 0)  # rest_ratio
        
        # Measures approximation
        num_measures = max(1, int(total_duration / 4))
        musical_features.append(total_notes / num_measures if num_measures > 0 else 0)  # avg_notes_per_measure
        
        # Key confidence
        musical_features.append(0.5)  # key_confidence placeholder
        
        # Pad musical features to 17
        while len(musical_features) < 17:
            musical_features.append(0.0)
        musical_features = musical_features[:17]
        
        # Harmonic features (15 features) - simplified
        harmonic_features = [
            tempo / 120.0,  # normalized tempo
            total_duration / 100.0 if total_duration > 0 else 0,  # normalized duration
            total_notes / 1000.0 if total_notes > 0 else 0,  # normalized note count
            len(chords) / total_notes if total_notes > 0 else 0,  # chord density
            len(rests) / total_notes if total_notes > 0 else 0,   # rest density
            musical_features[3] / 127.0 if len(musical_features) > 3 else 0.5,  # normalized avg pitch
            musical_features[4] / 127.0 if len(musical_features) > 4 else 0,    # normalized pitch range
            musical_features[5] / 30.0 if len(musical_features) > 5 else 0,     # normalized pitch std
            musical_features[6] / 12.0 if len(musical_features) > 6 else 0,     # normalized avg interval
            musical_features[9] / 4.0 if len(musical_features) > 9 else 0.25,   # normalized avg duration
            musical_features[11] / 10.0 if len(musical_features) > 11 else 0,   # normalized num parts
            num_measures / 100.0 if num_measures > 0 else 0,  # normalized measures
            0.5,  # placeholder
            0.5,  # placeholder  
            0.5   # placeholder
        ]
        
        # Note sequence (100 features) - simplified pitch sequence
        note_sequence = []
        for n in all_notes[:100]:
            if isinstance(n, note.Note):
                note_sequence.append(n.pitch.midi / 127.0)  # normalized
            elif isinstance(n, chord.Chord):
                note_sequence.append(n.root().midi / 127.0)  # normalized
        
        # Pad note sequence to 100
        while len(note_sequence) < 100:
            note_sequence.append(0.0)
        note_sequence = note_sequence[:100]
        
        return (
            np.array(musical_features).reshape(1, -1),
            np.array(harmonic_features).reshape(1, -1),
            np.array(note_sequence).reshape(1, -1)
        )
        
    except Exception as e:
        st.error(f"Error extracting features from {midi_path}: {str(e)}")
        # Return default features if extraction fails
        return (
            np.zeros((1, 17)),
            np.zeros((1, 15)), 
            np.zeros((1, 100))
        )

def midi_to_wav_data(midi_path, sample_rate=22050):
    """Convert MIDI to WAV audio data for playback"""
    try:
        # Load MIDI file using pretty_midi
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Synthesize audio
        audio = midi_data.synthesize(fs=sample_rate)
        
        # Normalize audio to prevent clipping
        if len(audio) > 0:
            audio = audio / np.max(np.abs(audio))
            # Ensure audio is not too loud
            audio = np.clip(audio * 0.8, -1.0, 1.0)
        
        return audio, sample_rate
    except Exception as e:
        st.warning(f"Could not synthesize audio from MIDI: {str(e)}")
        return None, None

def create_audio_player(midi_path):
    """Create an audio player for MIDI file with WAV preview"""
    try:
        # Convert MIDI to WAV
        audio_data, sample_rate = midi_to_wav_data(midi_path)
        
        if audio_data is not None and len(audio_data) > 0:
            # Display audio player directly with numpy array
            st.audio(audio_data, format='audio/wav', sample_rate=sample_rate)
            
            # Show audio info
            duration = len(audio_data) / sample_rate
            st.caption(f"🎵 Duration: {duration:.1f}s | Sample Rate: {sample_rate}Hz | Synthesized from MIDI")
        else:
            # Fallback: provide download link for original MIDI
            st.info("🎵 Audio synthesis not available. Download MIDI file to play in your preferred music software.")
        
        # Always provide MIDI download option
        with open(midi_path, "rb") as f:
            midi_bytes = f.read()
        
        b64 = base64.b64encode(midi_bytes).decode()
        href = f'<a href="data:audio/midi;base64,{b64}" download="{os.path.basename(midi_path)}">📥 Download Original MIDI File</a>'
        st.markdown(href, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error creating audio player: {str(e)}")
        # Fallback: just provide download link
        try:
            with open(midi_path, "rb") as f:
                midi_bytes = f.read()
            
            b64 = base64.b64encode(midi_bytes).decode()
            href = f'<a href="data:audio/midi;base64,{b64}" download="{os.path.basename(midi_path)}">📥 Download MIDI File</a>'
            st.markdown(href, unsafe_allow_html=True)
        except:
            st.error("Could not process audio file.")

def display_file_info(midi_path):
    """Display information about the MIDI file"""
    try:
        midi = converter.parse(midi_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**File Information:**")
            st.write(f"• Duration: {midi.duration.quarterLength:.1f} quarter notes" if midi.duration else "• Duration: Unknown")
            st.write(f"• Parts: {len(midi.parts) if hasattr(midi, 'parts') else 1}")
            
        with col2:
            # Time signature
            time_sigs = midi.getTimeSignatures()
            time_sig = f"{time_sigs[0].numerator}/{time_sigs[0].denominator}" if time_sigs else "4/4"
            st.write(f"• Time Signature: {time_sig}")
            
            # Tempo
            tempo_markings = midi.flat.getElementsByClass('MetronomeMark')
            tempo = tempo_markings[0].number if tempo_markings else 120
            st.write(f"• Tempo: {tempo} BPM")
            
    except Exception as e:
        st.write(f"Could not analyze file: {str(e)}")

# --- MAIN APP ---
def main():
    st.title("🎼 Classical Composer Prediction")
    st.markdown("**Predict the composer of classical music pieces using deep learning!**")
    
    # Load model and artifacts
    model, artifacts = load_model_and_artifacts()
    if model is None:
        st.error("Failed to load the model. Please check if the model files exist.")
        return
    
    # Get composer names
    composer_names = artifacts.get('composer_names', ['Bach', 'Beethoven', 'Chopin', 'Mozart'])
    
    if not composer_names:
        st.error("No composers found.")
        return
    
    st.sidebar.title("🎵 Navigation")
    
    # Sidebar options
    mode = st.sidebar.radio(
        "Choose Mode:",
        ["🔍 Predict Composer", "📚 Browse Samples", "ℹ️ About Composers"]
    )
    
    if mode == "🔍 Predict Composer":
        st.header("🔍 Composer Prediction")
        st.markdown("Select a composer and piece to test the AI's prediction capabilities!")
        
        # Composer selection
        selected_composer = st.selectbox(
            "Select a composer to test:",
            composer_names,
            help="Choose a composer to select a sample from their works"
        )
        
        # Get MIDI files for selected composer
        midi_files = get_midi_files_for_composer(selected_composer)
        
        if not midi_files:
            st.warning(f"No MIDI files found for {selected_composer}")
            return
        
        # File selection
        selected_file = st.selectbox(
            "Select a piece:",
            midi_files,
            format_func=lambda x: os.path.basename(x).replace('.mid', '').replace('.midi', '').replace('_', ' ')
        )
        
        if selected_file:
            st.subheader(f"🎼 Selected: {os.path.basename(selected_file).replace('.mid', '').replace('.midi', '').replace('_', ' ')}")
            
            # Create two columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display file information
                display_file_info(selected_file)
                
                st.markdown("---")
                st.write(f"**Actual Composer:** {selected_composer}")
                
                # Audio player
                st.subheader("🎵 Audio Preview")
                create_audio_player(selected_file)
            
            with col2:
                # Show composer info
                if selected_composer in COMPOSER_INFO:
                    composer_img = load_composer_image(selected_composer)
                    if composer_img:
                        st.image(composer_img, caption=f"{selected_composer}", width=200)
                    
                    info = COMPOSER_INFO[selected_composer]
                    st.markdown(f"**{info['full_name']}**")
                    st.markdown(f"*{info['years']}*")
            
            st.markdown("---")
            
            # Prediction button
            if st.button("🎯 Predict Composer", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing musical features and making prediction..."):
                    
                    # Extract features
                    musical_features, harmonic_features, note_sequence = extract_features_for_model(selected_file)
                    
                    if musical_features is None:
                        st.error("Failed to extract features from the MIDI file.")
                        return
                    
                    # Make prediction
                    try:
                        prediction = model.predict([musical_features, harmonic_features, note_sequence], verbose=0)
                        predicted_idx = np.argmax(prediction[0])
                        predicted_composer = composer_names[predicted_idx]
                        confidence = float(prediction[0][predicted_idx]) * 100
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        
                        # Create columns for results
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.metric("Predicted Composer", predicted_composer)
                        
                        with col2:
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        with col3:
                            if predicted_composer == selected_composer:
                                st.success("✅ Correct!")
                            else:
                                st.error("❌ Incorrect")
                        
                        # Show all prediction scores
                        st.subheader("📊 All Prediction Scores")
                        scores_df = pd.DataFrame({
                            'Composer': composer_names,
                            'Probability (%)': [float(score) * 100 for score in prediction[0]]
                        }).sort_values('Probability (%)', ascending=False)
                        
                        # Create a nice bar chart
                        st.bar_chart(scores_df.set_index('Composer')['Probability (%)'])
                        
                        # Show detailed scores
                        st.dataframe(scores_df, use_container_width=True)
                        
                        # Show predicted composer info
                        if predicted_composer in COMPOSER_INFO:
                            st.subheader(f"🎼 About the Predicted Composer: {predicted_composer}")
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                composer_img = load_composer_image(predicted_composer)
                                if composer_img:
                                    st.image(composer_img, width=200)
                            
                            with col2:
                                info = COMPOSER_INFO[predicted_composer]
                                st.write(f"**Full Name:** {info['full_name']}")
                                st.write(f"**Years:** {info['years']}")
                                st.write(f"**Description:** {info['description']}")
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.write("Please try with a different MIDI file.")
    
    elif mode == "📚 Browse Samples":
        st.header("📚 Browse Composer Samples")
        st.markdown("Explore sample pieces from each composer in the dataset!")
        
        for composer in composer_names:
            st.subheader(f"🎼 {composer}")
            
            # Create columns for composer info and image
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Show composer info
                if composer in COMPOSER_INFO:
                    info = COMPOSER_INFO[composer]
                    st.write(f"**{info['full_name']}** ({info['years']})")
                    st.write(info['description'])
            
            with col2:
                if composer in COMPOSER_INFO:
                    composer_img = load_composer_image(composer)
                    if composer_img:
                        st.image(composer_img, width=150)
            
            # Get and display sample files
            midi_files = get_midi_files_for_composer(composer)
            
            if midi_files:
                st.write(f"**Available pieces: {len(midi_files)}**")
                
                # Show first few samples
                samples_to_show = min(3, len(midi_files))
                for i in range(samples_to_show):
                    file_path = midi_files[i]
                    file_name = os.path.basename(file_path).replace('.mid', '').replace('.midi', '').replace('_', ' ')
                    
                    with st.expander(f"🎵 {file_name}"):
                        create_audio_player(file_path)
                
                if len(midi_files) > samples_to_show:
                    st.info(f"... and {len(midi_files) - samples_to_show} more pieces available")
            else:
                st.warning("No MIDI files found for this composer.")
            
            st.markdown("---")
    
    elif mode == "ℹ️ About Composers":
        st.header("ℹ️ About the Composers")
        st.markdown("Learn about the four classical composers included in this AI model!")
        
        for composer in composer_names:
            if composer in COMPOSER_INFO:
                st.subheader(f"🎼 {composer}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    composer_img = load_composer_image(composer)
                    if composer_img:
                        st.image(composer_img, width=200)
                
                with col2:
                    info = COMPOSER_INFO[composer]
                    st.markdown(f"**Full Name:** {info['full_name']}")
                    st.markdown(f"**Years:** {info['years']}")
                    st.markdown(f"**Description:** {info['description']}")
                
                # Show number of available samples
                midi_files = get_midi_files_for_composer(composer)
                st.info(f"📊 Available pieces in dataset: {len(midi_files)}")
                
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        🎼 Made with ❤️ using Streamlit | Deep Learning for Music Classification<br>
        <small>This AI model was trained on classical music pieces to identify compositional styles</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
