# import streamlit as st
# import time
# import os
# import torch
# import json
# import requests
# import subprocess
# from TTS.api import TTS

# # Ollama Server URL
# OLLAMA_SERVER_URL = "http://122.166.46.242:11434/api/generate"

# # Ensure all session state variables are initialized
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
# if "characters_and_topics_submitted" not in st.session_state:
#     st.session_state["characters_and_topics_submitted"] = False
# if "character_persona_submitted" not in st.session_state:
#     st.session_state["character_persona_submitted"] = False
# if "guests" not in st.session_state:
#     st.session_state["guests"] = []  # Initialize as empty list
# if "host_character" not in st.session_state:
#     st.session_state["host_character"] = "Tony"  # Default host
# if "podcast_topic" not in st.session_state:
#     st.session_state["podcast_topic"] = ""
# if "dialog_count" not in st.session_state:
#     st.session_state["dialog_count"] = 12  # Default value


# st.title("🎙️ Podcast Generator 🎙️")

# # Restart Button
# if "characters_and_topics_submitted" not in st.session_state or st.sidebar.button("Restart the podcast"):
#     st.session_state["messages"] = []
#     st.session_state["characters_and_topics_submitted"] = False
#     st.session_state["character_persona_submitted"] = False
#     st.rerun()

# # Function to Generate Podcast Transcript
# def generate_dialog(number_of_dialogs, timestamp):
#     character_personas = "\n".join([f"- {guest} Persona: {st.session_state.get(f'{guest}_persona', '')}" for guest in st.session_state["guests"]])
#     guest_introductions = ", ".join(st.session_state["guests"])

#     podcast_template = f"""## Podcast Outline
#     This is a podcast between {st.session_state["host_character"]}, {guest_introductions}.
#     {st.session_state["host_character"]} is the host of the show.
#     {st.session_state['podcast_topic']}

#     Character Personas:
#     {character_personas}
#     """

#     instructions = f"""Instructions:
#     - The podcast should have around {number_of_dialogs} dialogs. Always include a closure.
#     - Avoid non-verbal cues like *laughs* or *ahem*. Use 'Haha' instead.
#     - Provide each line in the format: SPEAKER: content
#     """

#     if not os.path.exists("podcasts"):
#         os.makedirs("podcasts")

#     transcript_file_name = f"podcasts/podcast{timestamp}.txt"

#     payload = json.dumps({
#         "model": "llama3.2:latest",
#         "prompt": f"Generate a full podcast transcript: {podcast_template} {instructions}",
#         "stream": False
#     })
#     headers = {"Content-Type": "application/json"}

#     try:
#         response = requests.post(OLLAMA_SERVER_URL, data=payload, headers=headers)
#         data = response.json()
#         transcript = data.get("response", "")
        
#         dialogs = []
#         with open(transcript_file_name, "w") as transcript_file:
#             transcript_file.write(transcript)
#             for line in transcript.split("\n"):
#                 if ":" in line:
#                     parts = line.split(":", 1)
#                     if len(parts) == 2:
#                         speaker, content = parts
#                         if speaker.strip() in ["Tony", "Mia", "Denzel", "Alex", "Nimbus", "Roland"]:
#                             dialogs.append({"speaker": speaker.strip(), "content": content.strip()})
        
#         return dialogs
#     except Exception as e:
#         st.error(f"Error generating transcript: {str(e)}")
#         return []

# # Function to Generate Audio from Dialogs
# def generate_audio(dialogs, timestamp):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     voice_names = {
#         "Mia": r"voices/Mia.wav",
#         "Denzel": r"voices/Denzel_Wash.wav",
#         "Alex": r"voices/Alex_Danivero.wav",
#         "Nimbus": r"voices/Nimbus.wav",
#         "Tony": r"voices/Tony_King.wav",
#         "Roland": r"voices/Roland.wav",
#     }

#     if not os.path.exists("dialogs"):
#         os.makedirs("dialogs")
    
#     if not os.path.exists("podcasts"):
#         os.makedirs("podcasts")
    
#     concat_file = open("concat.txt", "w")
    
#     tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
#     dialog_files = []
    
#     try:
#         for i, dialog in enumerate(dialogs):
#             # Skip if dialog has no content
#             if not dialog.get("content"):
#                 continue
                
#             filename = f"dialogs/dialog{i}.wav"
            
#             # Determine if we should split sentences based on content length
#             split_sentences = len(dialog["content"]) > 250
            
#             # Get the voice file, with fallback to Tony if not found
#             voice_file = voice_names.get(dialog["speaker"], voice_names["Tony"])
            
#             # Generate audio from text
#             tts.tts_to_file(
#                 text=dialog["content"],
#                 speaker_wav=voice_file,
#                 language="en",
#                 split_sentences=split_sentences,
#                 file_path=filename
#             )
            
#             concat_file.write(f"file '{filename}'\n")
#             dialog_files.append(filename)
#     except Exception as e:
#         st.error(f"Error generating audio: {str(e)}")
#         if len(dialog_files) == 0:
#             concat_file.close()
#             os.unlink("concat.txt")
#             return
    
#     concat_file.close()
    
#     podcast_file = f"podcasts/podcast{timestamp}.wav"
    
#     try:
#         subprocess.run(f"ffmpeg -f concat -safe 0 -i concat.txt -c copy {podcast_file}", shell=True)
#         os.unlink("concat.txt")
        
#         for file in dialog_files:
#             if os.path.exists(file):
#                 os.unlink(file)
        
#         st.audio(podcast_file, format="audio/wav")
#     except Exception as e:
#         st.error(f"Error finalizing podcast: {str(e)}")

# # Function to Generate the Full Podcast
# def generate_podcast():
#     current_time = int(time.time())
#     with st.spinner("📜 Generating the transcript..."):
#         dialogs = generate_dialog(st.session_state["dialog_count"], current_time)
    
#     if not dialogs:
#         st.error("Failed to generate transcript. Please try again.")
#         return
        
#     st.write("✅ Transcript generated successfully")
    
#     with st.spinner("🎤 Generating the audio..."):
#         generate_audio(dialogs, current_time)

# # User Inputs
# if not st.session_state["characters_and_topics_submitted"]:
#     with st.form("characters_and_topics"):
#         st.selectbox("Select your host character", ("Tony", "Mia", "Denzel", "Alex", "Nimbus", "Roland"), key="host_character")
#         st.multiselect("Select guests", ["Denzel", "Alex", "Nimbus", "Tony", "Roland", "Mia"], key="guests")
#         st.text_area("Podcast topic", key="podcast_topic")
#         st.slider("Number of dialogs", 7, 15, 12, key="dialog_count")
#         st.session_state["characters_and_topics_submitted"] = st.form_submit_button("Submit")

# if st.session_state["characters_and_topics_submitted"]:
#     with st.form("character_persona"):
#         for guest in st.session_state["guests"]:
#             st.text_area(f"Enter persona for {guest}", key=f"{guest}_persona")
#         st.session_state["character_persona_submitted"] = st.form_submit_button("Submit")

# if st.session_state["character_persona_submitted"]:
#     generate_podcast()
#     st.write("🎉 Podcast generated successfully!")



import streamlit as st
import time
import os
import torch
import json
import requests
import subprocess
from TTS.api import TTS
import pandas as pd
import random

# Configuration options and settings in the sidebar
st.sidebar.title("Configuration")

# Ollama Server URL with input option
OLLAMA_SERVER_URL = "http://122.166.46.242:11434/api/generate"
    


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

# Function to ensure balanced character participation
def ensure_balanced_participation(dialogs, host, guests, min_dialogs=None):
    if not dialogs:
        return []
    
    # Count participation
    character_counts = {character: 0 for character in [host] + guests}
    for dialog in dialogs:
        if dialog["speaker"] in character_counts:
            character_counts[dialog["speaker"]] += 1
    
    # Check if any character doesn't appear
    all_characters = [host] + guests
    missing_characters = [char for char in all_characters if character_counts[char] == 0]
    
    # Set minimum dialogs based on total dialog count and number of characters
    if min_dialogs is None:
        total_dialogs = len(dialogs)
        min_dialogs = max(1, total_dialogs // (len(all_characters) * 2))
    
    # If characters are missing or have too few lines, fix the transcript
    if missing_characters or any(count < min_dialogs for count in character_counts.values()):
        st.warning(f"The generated transcript had unbalanced character participation. Attempting to fix.")
        
        if DEBUG_MODE:
            st.write("Character counts:", character_counts)
            st.write("Missing characters:", missing_characters)
        
        # Create a new, balanced set of dialogs
        balanced_dialogs = []
        
        # Always start with host introduction
        balanced_dialogs.append({
            "speaker": host,
            "content": f"Welcome to our podcast! Today we're discussing {st.session_state['podcast_topic']}. I'm joined by {', '.join(guests)}."
        })
        
        # Add introductions for each guest
        for guest in guests:
            balanced_dialogs.append({
                "speaker": guest,
                "content": f"Thanks for having me on the show. I'm excited to talk about this topic."
            })
        
        # Add host question to start the conversation
        balanced_dialogs.append({
            "speaker": host,
            "content": f"Let's dive into our topic about {st.session_state['podcast_topic']}. What are your thoughts on this?"
        })
        
        # Use the existing dialogs that are valid, ensuring each character speaks in turn
        remaining_dialogs = [d for d in dialogs if d["speaker"] in all_characters]
        
        # Sort speakers to create a rotation pattern
        speakers_rotation = [host] + guests
        current_speaker_index = 0
        
        # Filter dialogs by speaker to ensure balanced distribution
        speaker_dialogs = {speaker: [] for speaker in speakers_rotation}
        for dialog in remaining_dialogs:
            speaker = dialog["speaker"]
            if speaker in speaker_dialogs:
                speaker_dialogs[speaker].append(dialog)
        
        # Calculate target dialog count based on total desired dialogs
        target_total = min(len(dialogs), st.session_state["dialog_count"])
        target_per_speaker = max(2, target_total // len(speakers_rotation))
        
        # Build a balanced set of dialogs by rotating through speakers
        while len(balanced_dialogs) < target_total and any(len(dialogs) > 0 for dialogs in speaker_dialogs.values()):
            speaker = speakers_rotation[current_speaker_index]
            if speaker_dialogs[speaker]:
                balanced_dialogs.append(speaker_dialogs[speaker].pop(0))
            current_speaker_index = (current_speaker_index + 1) % len(speakers_rotation)
        
        # Ensure the host concludes the podcast
        balanced_dialogs.append({
            "speaker": host,
            "content": f"Thank you everyone for joining us today for this fascinating discussion about {st.session_state['podcast_topic']}. I hope our listeners enjoyed it as much as I did. Until next time!"
        })
        
        return balanced_dialogs
    
    # If participation is already balanced, return original dialogs
    return dialogs

# Function to Generate Podcast Transcript
def generate_dialog(number_of_dialogs, podcast_topic, host, guests, personas):
    character_personas = "\n".join([f"- {character} Persona: {persona}" for character, persona in personas.items()])
    guest_introductions = ", ".join(guests)

    # Create a more structured prompt to ensure all characters participate
    podcast_template = f"""## Podcast Outline
    This is a podcast between host {host} and guests {guest_introductions}.
    The podcast topic is: {podcast_topic}

    Character Personas:
    {character_personas}
    """

    # Enhanced instructions to ensure balanced participation
    instructions = f"""Instructions:
    - Create a natural podcast conversation with approximately {number_of_dialogs} dialogs
    - IMPORTANT: ENSURE ALL CHARACTERS participate roughly equally in the conversation
    - The host ({host}) should facilitate the conversation and engage with EACH guest
    - EACH guest must speak multiple times throughout the podcast
    - Begin with the host introducing the topic and guests
    - End with the host concluding the podcast
    - Format each line as: SPEAKER: content (e.g., "{host}: Welcome to the podcast!")
    - Avoid non-verbal cues like *laughs* or *ahem* - use natural speech patterns instead
    - Make the conversation flow naturally with appropriate back-and-forth
    - Each character should speak according to their defined persona
    """

    if not os.path.exists("podcasts"):
        os.makedirs("podcasts")

    timestamp = int(time.time())
    transcript_file_name = f"podcasts/podcast_{timestamp}.txt"
    
    # Add character participation instructions
    character_instructions = "Character participation requirements:\n"
    character_instructions += f"- Host ({host}) should speak at least {max(3, number_of_dialogs // (len(guests) + 1))} times\n"
    for guest in guests:
        character_instructions += f"- {guest} should speak at least {max(2, number_of_dialogs // (len(guests) + 2))} times\n"
    
    # Prepare the payload for Ollama with enhanced instructions
    payload = json.dumps({
        "model": SELECTED_MODEL,
        "prompt": f"Generate a full podcast transcript:\n{podcast_template}\n{instructions}\n{character_instructions}",
        "stream": False
    })
    headers = {"Content-Type": "application/json"}

    if DEBUG_MODE:
        st.code(podcast_template + instructions + character_instructions, language="text")

    try:
        # Make the API call to Ollama
        with st.spinner("Connecting to Ollama server and generating transcript..."):
            response = requests.post(OLLAMA_SERVER_URL, data=payload, headers=headers, timeout=180)
        
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
        
        # Ensure all characters have balanced participation
        balanced_dialogs = ensure_balanced_participation(dialogs, host, guests)
        
        # Save the balanced transcript
        with open(transcript_file_name, "w") as transcript_file:
            for dialog in balanced_dialogs:
                transcript_file.write(f"{dialog['speaker']}: {dialog['content']}\n")
        
        return balanced_dialogs, timestamp
    
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
        available_characters = ["Tony", "Mia", "Denzel", "Alex", "Nimbus", "Roland"]
    
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
            [char for char in available_characters if char != st.session_state["host_character"]],
            help="Select at least one guest. Each guest will participate in the conversation."
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

# Function to generate sample personas based on character names
def generate_sample_persona(character_name):
    personas = {
        "Tony": "A charismatic tech entrepreneur with a knack for simplifying complex topics. Known for his witty remarks and thoughtful questions.",
        "Mia": "A brilliant scientist with expertise in multiple fields. She explains concepts clearly and has a warm, engaging personality.",
        "Denzel": "A wise philosopher and storyteller who draws from a wealth of life experiences. His deep voice and careful word choice captivate listeners.",
        "Alex": "A quick-witted journalist who asks insightful questions. Known for connecting seemingly unrelated topics in fascinating ways.",
        "Nimbus": "A forward-thinking futurist with unconventional ideas. Enthusiastic about emerging technologies and societal changes.",
        "Roland": "A seasoned historian and cultural critic with a dry sense of humor. Offers historical context to modern discussions."
    }
    
    # Return persona if available, otherwise generate a generic one
    return personas.get(character_name, f"An interesting character with unique perspectives on {st.session_state['podcast_topic']}.")

# Persona Input Form
if st.session_state["characters_and_topics_submitted"] and not st.session_state["character_persona_submitted"]:
    with st.form("character_persona"):
        personas = {}
        
        # Create examples to help users
        st.info("Define how each character will speak and behave. This influences the conversation style.")
        
        # Host persona with sample
        sample_host_persona = generate_sample_persona(st.session_state['host_character'])
        host_persona = st.text_area(
            f"Enter persona for host ({st.session_state['host_character']})",
            value=sample_host_persona,
            help="Describe the personality, background, and speaking style of this character"
        )
        personas[st.session_state["host_character"]] = host_persona
        
        # Guest personas with samples
        for guest in st.session_state["guests"]:
            sample_persona = generate_sample_persona(guest)
            persona = st.text_area(
                f"Enter persona for {guest}", 
                value=sample_persona,
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
    
    # Calculate speaker statistics
    all_characters = [st.session_state["host_character"]] + st.session_state["guests"]
    speaker_counts = {character: 0 for character in all_characters}
    for dialog in st.session_state["generated_dialogs"]:
        if dialog["speaker"] in speaker_counts:
            speaker_counts[dialog["speaker"]] += 1
    
    # Display speaker statistics
    st.write("Speaker participation:")
    for character, count in speaker_counts.items():
        percentage = (count / len(st.session_state["generated_dialogs"])) * 100
        st.write(f"- {character}: {count} lines ({percentage:.1f}%)")
    
    # Create a dataframe for better display
    transcript_data = []
    for i, dialog in enumerate(st.session_state["generated_dialogs"]):
        transcript_data.append({
            "#": i+1,
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
        
        # Download button for the podcast
        with open(podcast_file, "rb") as file:
            st.download_button(
                label="Download Podcast",
                data=file,
                file_name=f"podcast_{st.session_state['podcast_timestamp']}.wav",
                mime="audio/wav"
            )
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