# import requests
# import json

# url = "http://122.166.46.242:11434/api/generate"
# payload = json.dumps({"model": "llama3.2:latest", "prompt": "Hello"})
# headers = {"Content-Type": "application/json"}

# response = requests.post(url, data=payload, headers=headers, stream=True)

# # Read the streamed response line by line
# for line in response.iter_lines():
#     if line:
#         print(json.loads(line))



import streamlit as st
import time
import os
import torch
import json
import requests
import subprocess
from TTS.api import TTS
import pandas as pd

# Configuration options and settings in the sidebar
st.sidebar.title("Configuration")

# Ollama Server URL with input option
OLLAMA_SERVER_URL = st.sidebar.text_input(
    "Ollama Server URL", 
    value="http://localhost:11434/api/generate",
    help="Enter the URL for your Ollama server"
)

# Model selection
MODEL_OPTIONS = ["llama3:8b-instruct-q4_K_M", "llama3:70b-instruct-q4_K_M", "llama3.2:latest"]
SELECTED_MODEL = st.sidebar.selectbox("Select LLM Model", MODEL_OPTIONS)

# TTS Model selection
TTS_MODELS = ["tts_models/multilingual/multi-dataset/xtts_v2", "tts_models/en/ljspeech/tacotron2-DDC"]
SELECTED_TTS_MODEL = st.sidebar.selectbox("Select TTS Model", TTS_MODELS)

# Debug mode toggle
DEBUG_MODE = st.sidebar.checkbox("Debug Mode", value=False, help="Show additional debug information")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "characters_and_topics_submitted" not in st.session_state:
    st.session_state["characters_and_topics_submitted"] = False
if "character_persona_submitted" not in st.session_state:
    st.session_state["character_persona_submitted"] = False
if "guests" not in st.session_state:
    st.session_state["guests"] = []
if "host_character" not in st.session_state:
    st.session_state["host_character"] = ""
if "podcast_topic" not in st.session_state:
    st.session_state["podcast_topic"] = ""
if "dialog_count" not in st.session_state:
    st.session_state["dialog_count"] = 12
if "generated_dialogs" not in st.session_state:
    st.session_state["generated_dialogs"] = []
if "podcast_timestamp" not in st.session_state:
    st.session_state["podcast_timestamp"] = ""
if "voice_files" not in st.session_state:
    st.session_state["voice_files"] = {}

st.title("🎙️ Podcast Generator 🎙️")

# Restart Button
if st.sidebar.button("Restart the podcast"):
    for key in ["messages", "characters_and_topics_submitted", "character_persona_submitted", 
               "guests", "host_character", "podcast_topic", "dialog_count", 
               "generated_dialogs", "podcast_timestamp", "voice_files"]:
        if key in st.session_state:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else ""
    st.session_state["characters_and_topics_submitted"] = False
    st.session_state["character_persona_submitted"] = False
    st.session_state["dialog_count"] = 12
    st.rerun()

# Function to scan for available voice files
def scan_voice_files():
    voice_files = {}
    voices_dir = "voices"
    if os.path.exists(voices_dir):
        for file in os.listdir(voices_dir):
            if file.endswith(".wav"):
                character_name = file.split(".")[0].replace("_", " ")
                voice_files[character_name] = os.path.join(voices_dir, file)
    return voice_files

# Function to Generate Podcast Transcript
def generate_dialog(number_of_dialogs, podcast_topic, host, guests, personas):
    character_personas = "\n".join([f"- {character} Persona: {persona}" for character, persona in personas.items()])
    guest_introductions = ", ".join(guests)

    podcast_template = f"""## Podcast Outline
    This is a podcast between {host}, {guest_introductions}.
    {host} is the host of the show.
    Podcast Topic: {podcast_topic}

    Character Personas:
    {character_personas}
    """

    instructions = f"""Instructions:
    - The podcast should have approximately {number_of_dialogs} dialogs. Always include a natural closure at the end.
    - Avoid non-verbal cues like *laughs* or *ahem*. Use 'Haha' or natural speech patterns instead.
    - Provide each line in the format: SPEAKER: content
    - Make the conversation natural, with appropriate back-and-forth between participants
    - Ensure each character speaks in accordance with their defined persona
    """

    if not os.path.exists("podcasts"):
        os.makedirs("podcasts")

    timestamp = int(time.time())
    transcript_file_name = f"podcasts/podcast_{timestamp}.txt"
    
    # Prepare the payload for Ollama
    payload = json.dumps({
        "model": SELECTED_MODEL,
        "prompt": f"Generate a full podcast transcript: {podcast_template} {instructions}",
        "stream": False
    })
    headers = {"Content-Type": "application/json"}

    if DEBUG_MODE:
        st.code(podcast_template + instructions, language="text")

    try:
        # Make the API call to Ollama
        with st.spinner("Connecting to Ollama server and generating transcript..."):
            response = requests.post(OLLAMA_SERVER_URL, data=payload, headers=headers, timeout=120)
        
        if response.status_code != 200:
            st.error(f"Error from Ollama API: {response.status_code} - {response.text}")
            return [], timestamp
            
        data = response.json()
        transcript = data.get("response", "")
        
        # Parse the transcript into dialogs
        dialogs = []
        with open(transcript_file_name, "w") as transcript_file:
            transcript_file.write(transcript)
            
            for line in transcript.split("\n"):
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        speaker, content = parts
                        speaker = speaker.strip()
                        
                        # Check if the speaker is valid (either host or one of the guests)
                        valid_speakers = [host] + guests
                        if speaker in valid_speakers:
                            dialogs.append({"speaker": speaker, "content": content.strip()})
        
        return dialogs, timestamp
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        st.info("Please check that your Ollama server is running and accessible.")
        return [], timestamp
    except Exception as e:
        st.error(f"Error generating transcript: {str(e)}")
        return [], timestamp

# Function to Generate Audio from Dialogs
def generate_audio(dialogs, timestamp, voice_mapping):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create necessary directories
    if not os.path.exists("dialogs"):
        os.makedirs("dialogs")
    
    if not os.path.exists("podcasts"):
        os.makedirs("podcasts")
    
    # Create file for FFmpeg concatenation
    concat_file = open("concat.txt", "w")
    
    # Initialize TTS model
    try:
        with st.spinner(f"Loading TTS model to {device}..."):
            tts = TTS(SELECTED_TTS_MODEL).to(device)
    except Exception as e:
        st.error(f"Error loading TTS model: {str(e)}")
        if os.path.exists("concat.txt"):
            os.unlink("concat.txt")
        return None
    
    dialog_files = []
    progress_bar = st.progress(0)
    
    try:
        total_dialogs = len(dialogs)
        for i, dialog in enumerate(dialogs):
            # Skip if dialog has no content
            if not dialog.get("content"):
                continue
                
            filename = f"dialogs/dialog_{timestamp}_{i}.wav"
            
            # Determine if we should split sentences based on content length
            split_sentences = len(dialog["content"]) > 250
            
            # Get the voice file, default to first voice if not found
            speaker = dialog["speaker"]
            voice_file = voice_mapping.get(speaker)
            
            if not voice_file:
                st.warning(f"No voice file found for {speaker}. Using default voice.")
                voice_file = next(iter(voice_mapping.values()))
            
            # Update progress
            progress_text = f"Generating audio for {speaker} ({i+1}/{total_dialogs})"
            st.text(progress_text)
            progress_bar.progress((i + 1) / total_dialogs)
            
            # Generate audio from text
            tts.tts_to_file(
                text=dialog["content"],
                speaker_wav=voice_file,
                language="en",
                split_sentences=split_sentences,
                file_path=filename
            )
            
            # Add to concat file
            concat_file.write(f"file '{filename}'\n")
            dialog_files.append(filename)
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        if len(dialog_files) == 0:
            concat_file.close()
            if os.path.exists("concat.txt"):
                os.unlink("concat.txt")
            return None
    
    concat_file.close()
    
    # Final podcast file path
    podcast_file = f"podcasts/podcast_{timestamp}.wav"
    
    try:
        # Use FFmpeg to concatenate all dialog audio files
        with st.spinner("Combining audio files..."):
            subprocess.run(f"ffmpeg -f concat -safe 0 -i concat.txt -c copy {podcast_file}", shell=True)
        if os.path.exists("concat.txt"):
            os.unlink("concat.txt")
        
        # Clean up individual dialog files unless in debug mode
        if not DEBUG_MODE:
            for file in dialog_files:
                if os.path.exists(file):
                    os.unlink(file)
        
        return podcast_file
    except Exception as e:
        st.error(f"Error finalizing podcast: {str(e)}")
        return None

# Scan for voice files during initialization
if not st.session_state["voice_files"]:
    st.session_state["voice_files"] = scan_voice_files()

# Function to display available characters
def display_available_characters():
    available_characters = list(st.session_state["voice_files"].keys())
    
    if not available_characters:
        st.warning("No voice files found. Please add voice files to the 'voices' directory.")
        available_characters = ["Default Host", "Guest 1", "Guest 2"]
    
    return available_characters

# Upload voice files
with st.sidebar.expander("Upload Voice Files"):
    voice_file = st.file_uploader("Upload Voice Sample", type=["wav"])
    character_name = st.text_input("Character Name")
    
    if voice_file is not None and character_name:
        # Ensure the voices directory exists
        if not os.path.exists("voices"):
            os.makedirs("voices")
            
        # Save the uploaded file
        voice_path = os.path.join("voices", f"{character_name.replace(' ', '_')}.wav")
        with open(voice_path, "wb") as f:
            f.write(voice_file.getbuffer())
            
        # Add to the voice files dictionary
        st.session_state["voice_files"][character_name] = voice_path
        st.success(f"Voice file for {character_name} uploaded successfully!")
        
        # Clear the inputs
        voice_file = None
        character_name = ""

# User Inputs
if not st.session_state["characters_and_topics_submitted"]:
    with st.form("characters_and_topics"):
        available_characters = display_available_characters()
        
        st.session_state["host_character"] = st.selectbox(
            "Select your host character", 
            available_characters,
            index=0 if available_characters else 0
        )
        
        st.session_state["guests"] = st.multiselect(
            "Select guests", 
            [char for char in available_characters if char != st.session_state["host_character"]]
        )
        
        st.session_state["podcast_topic"] = st.text_area(
            "Podcast topic", 
            help="Describe the podcast topic, theme, and any specific directions"
        )
        
        # More dynamic dialog count with explanation
        st.session_state["dialog_count"] = st.slider(
            "Number of dialogs", 
            min_value=5, 
            max_value=30, 
            value=12,
            help="More dialogs will result in a longer podcast"
        )
        
        # Estimated duration
        avg_words_per_dialog = 50
        avg_speaking_rate = 150  # words per minute
        estimated_duration = (st.session_state["dialog_count"] * avg_words_per_dialog) / avg_speaking_rate
        st.info(f"Estimated podcast duration: ~{estimated_duration:.1f} minutes")
        
        submitted = st.form_submit_button("Next: Define Character Personas")
        if submitted:
            if not st.session_state["guests"]:
                st.error("Please select at least one guest for your podcast")
            elif not st.session_state["podcast_topic"]:
                st.error("Please enter a podcast topic")
            else:
                st.session_state["characters_and_topics_submitted"] = True
                st.rerun()

# Persona Input Form
if st.session_state["characters_and_topics_submitted"] and not st.session_state["character_persona_submitted"]:
    with st.form("character_persona"):
        personas = {}
        
        # Host persona
        host_persona = st.text_area(
            f"Enter persona for host ({st.session_state['host_character']})",
            help="Describe the personality, background, and speaking style of this character"
        )
        personas[st.session_state["host_character"]] = host_persona
        
        # Guest personas
        for guest in st.session_state["guests"]:
            persona = st.text_area(
                f"Enter persona for {guest}", 
                help="Describe the personality, background, and speaking style of this character"
            )
            personas[guest] = persona
        
        submitted = st.form_submit_button("Generate Podcast")
        if submitted:
            # Validate personas
            if any(not persona for persona in personas.values()):
                st.error("Please enter personas for all characters")
            else:
                st.session_state["character_persona_submitted"] = True
                
                # Generate the transcript
                with st.spinner("🎙️ Creating your podcast..."):
                    # Generate transcript
                    dialogs, timestamp = generate_dialog(
                        st.session_state["dialog_count"],
                        st.session_state["podcast_topic"],
                        st.session_state["host_character"],
                        st.session_state["guests"],
                        personas
                    )
                    
                    if dialogs:
                        st.session_state["generated_dialogs"] = dialogs
                        st.session_state["podcast_timestamp"] = timestamp
                        st.rerun()
                    else:
                        st.error("Failed to generate podcast transcript. Please try again.")

# Generate Podcast when all inputs are ready
if st.session_state["character_persona_submitted"] and st.session_state["generated_dialogs"]:
    # Display transcript
    st.subheader("Podcast Transcript")
    
    # Create a dataframe for better display
    transcript_data = []
    for dialog in st.session_state["generated_dialogs"]:
        transcript_data.append({
            "Speaker": dialog["speaker"],
            "Content": dialog["content"]
        })
    
    if transcript_data:
        df = pd.DataFrame(transcript_data)
        st.dataframe(df, use_container_width=True)
    
    # Audio Generation
    st.subheader("Podcast Audio")
    
    # Check if audio already generated
    podcast_file = f"podcasts/podcast_{st.session_state['podcast_timestamp']}.wav"
    if os.path.exists(podcast_file):
        st.audio(podcast_file, format="audio/wav")
    else:
        if st.button("Generate Audio"):
            with st.spinner("🎤 Generating podcast audio..."):
                # Build voice mapping dictionary
                voice_mapping = {}
                for character, voice_file in st.session_state["voice_files"].items():
                    voice_mapping[character] = voice_file
                
                podcast_file = generate_audio(
                    st.session_state["generated_dialogs"], 
                    st.session_state["podcast_timestamp"],
                    voice_mapping
                )
                
                if podcast_file and os.path.exists(podcast_file):
                    st.success("🎉 Podcast generated successfully!")
                    st.audio(podcast_file, format="audio/wav")
                    
                    # Download button for the podcast
                    with open(podcast_file, "rb") as file:
                        st.download_button(
                            label="Download Podcast",
                            data=file,
                            file_name=f"podcast_{st.session_state['podcast_timestamp']}.wav",
                            mime="audio/wav"
                        )
                else:
                    st.error("Failed to generate podcast audio. Please try again.")

# Add download transcript button if dialogs exist
if st.session_state.get("generated_dialogs") and st.session_state.get("podcast_timestamp"):
    transcript_file = f"podcasts/podcast_{st.session_state['podcast_timestamp']}.txt"
    if os.path.exists(transcript_file):
        with open(transcript_file, "rb") as file:
            st.sidebar.download_button(
                label="Download Transcript",
                data=file,
                file_name=f"podcast_transcript_{st.session_state['podcast_timestamp']}.txt",
                mime="text/plain"
            )