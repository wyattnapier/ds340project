import os
import librosa
import numpy as np
import json
import pandas as pd
import speech_recognition as sr


recognizer = sr.Recognizer()
# input a file_path to a .wav file
# returns the transcribed audio as a string
# we can use BERT like in the homework to then tokenize/make into array and analyze it
def getVectorOfWords(file_path):
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        # print("Transcription:", recognizer.recognize_google(audio))
        return "" + recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None



# removes all files from images folder so subsequent runs don't have weird overlaps
def clearImagesFolder():
    print("Deleting all data from images folder")
    directory = os.getcwd() + "/images"
    for root, dirs, files in os.walk(directory, topdown=False):  # topdown=False to delete files before dirs
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.png'):
                os.remove(file_path)
                # print("" + file_path + " has been removed successfully")
    print("All images removed successfully!")
    
    
import librosa.display
import matplotlib.pyplot as plt
# input a file_path to a .wav file
# returns a png of the spectogram and a filepath to it
def getSpectogram(file_path, emotion_label):
    y, sr = librosa.load(file_path, sr=None) # load in the audio file and preserve its sample rate (replace with 16,000 if needed)
    
    # Compute the spectrogram
    D = librosa.stft(y)                        # Short-Time Fourier Transform
    S_db = librosa.amplitude_to_db(abs(D), ref=np.max)  # Convert to decibel scale

    # Plot and save the spectrogram
    fig = plt.figure(figsize=(6, 6))                # Set the figure size -- > num pixels will be 100 times this
    # can change the cmap to "viridis" or "plasma" for different color themes
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="log", cmap="magma")  # Log frequency scale to mimic human audio perception

    # TODO: at first try hiding as many extra features as possible and compare to when they're included
    # plt.colorbar(format="%+2.0f dB")           # Add a colorbar
    # plt.title("Spectrogram")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    
    # Save the spectrogram as an image file
    processed_path = (file_path.split("/")[-1]).split(".")[0]
    output_image_path = f"./images_all_given_transcript/{emotion_label}/{processed_path}.png"  # TODO: figure out naming conventions for the file -- either use path or just have a counter that we pass in
    plt.savefig(output_image_path, dpi=300)    # Save as PNG with high resolution
    plt.close()                                # Close the figure to free memory
    
    return output_image_path
    
    
def getTargetEmotionFromCSV(audio_file_name):
    # parse audio_file_name to get distinguishing file info for CSV lookup
    dialogueID, utteranceID = (audio_file_name.split(".wav")[0]).split('_')
    dialogueID, utteranceID = int(dialogueID[3:]), int(utteranceID[3:])
    csv = pd.read_csv('./train_sent_emo.csv')
    # Filter the row(s) that satisfy both conditions
    condition1 = (csv['Dialogue_ID'] == dialogueID)  # First column matches 'dialogueID'
    condition2 = (csv['Utterance_ID'] == utteranceID)  # Second column matches 'utteranceID'
    filtered_rows = csv[condition1 & condition2]
    return filtered_rows['Emotion'].iloc[0], filtered_rows['Sentiment'].iloc[0], filtered_rows['Utterance'].iloc[0]


def traverse_audio_files(directory="./train_splits_wav"):
    # clearImagesFolder() # deletes everything from the image folder
    data = []
    
    # Traverse and process .wav files
    print("Starting audio file traversal")
    for file_name in os.listdir(directory):
        # limit the number of loops so this doesn't take THAT long
        file_path = os.path.join(directory, file_name)
        
        if os.path.isfile(file_path) and file_name.endswith('.wav'):
            emotion, sentiment, utterance = getTargetEmotionFromCSV(file_name)
            image_path = getSpectogram(file_path, emotion)
            data.append({"Transcription": utterance, "Spectogram": image_path, "Emotion": emotion, "Sentiment": sentiment})
    df = pd.DataFrame(data)
    print("Finished creating dataframe and traversing audio files")
    return df
    
    
    
df = traverse_audio_files()
print(df)


df.to_csv('data_all_given_transcript.csv', index=False)