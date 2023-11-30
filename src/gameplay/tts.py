import string
from google.cloud import texttospeech

# Text input for tts
rowIndex = {letter:letter for letter in string.ascii_lowercase[:20]}
colIndex = {str(i):str(i) for i in range(1, 21)}
draw = "The board is filled. It is a draw!"
blackWin = "Game over. Black won!"
whiteWin = "Game over. White won!"

# Combine the input
inputDict = {**rowIndex, **colIndex}
inputDict["draw"] = draw
inputDict["blackWin"] = blackWin
inputDict["whiteWin"] = whiteWin

# Set up google tts client
client = texttospeech.TextToSpeechClient()
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# Iterate through the input dictionary
for key, value in inputDict.items():
    # Synthesize speech
    synthesis_input = texttospeech.SynthesisInput(text=value)
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Save the audio
    with open("./sound/{}.mp3".format(key), "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        
print("Done!")