# Talos
Local, secure, and personalized AI Assistant

# Dependencies
* Vosk - Speech recognition kit for speech recognition
* PyAudio - Audio I/O
* Pydantic - Data management

# Setup
1. [Download Ollama](https://ollama.com/download)
2. Install dependencies (recommended method is via [uv](https://github.com/astral-sh/uv))
3. Download the appropriate [Vosk model](https://alphacephei.com/vosk/models) (recommended: pick largest model for your target language)
   1. (Ensure that the code reflects this model change. Code changes in the future can make this more flexible)

# TODO
1. Memory support (i.e. remember past chats)
2. Integrate app support (i.e. manage calendars, push notifications)
3. Run in background with voice activation
4. Video support (i.e. take snapshot at user request and read notes)
5. Screen read support
6. Raspberry PI deployment