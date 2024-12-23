import os
import subprocess
import pandas as pd
import csv
from yt_dlp import YoutubeDL
import whisper
from textblob import TextBlob
from googletrans import Translator

def download_tiktok_video(url, output_dir="downloads"):
    """
    Downloads a TikTok video using yt-dlp.
    """
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'format': 'mp4',
        'quiet': False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info_dict)
        print(f"Downloaded video to: {video_path}")
        return video_path


def extract_audio_with_ffmpeg(video_path, output_dir="downloads"):
    """
    Extracts audio from the video file using FFmpeg.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    command = [
        "ffmpeg", "-i", video_path,
        "-q:a", "0", "-map", "a", audio_path, "-y"
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print(f"Extracted audio to: {audio_path}")
    return audio_path


def transcribe_audio(audio_path, model_type="base"):
    """
    Transcribes audio using OpenAI's Whisper model.
    """
    model = whisper.load_model(model_type)
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    transcript = result["text"]
    print(f"Transcription: {transcript}")
    return transcript


def translate_to_english(transcript, source_lang="id"):
    """
    Translates text to English using Google Translate.
    """
    translator = Translator()
    translated_text = translator.translate(transcript, src=source_lang, dest="en").text
    print(f"Translated Transcript: {translated_text}")
    return translated_text


def analyze_sentiment(transcript, neutral_threshold=0.1):
    """
    Analyzes the sentiment of the transcript using TextBlob.
    Returns 'positive', 'neutral', or 'negative'.
    """
    analysis = TextBlob(transcript)
    polarity = analysis.sentiment.polarity

    if -neutral_threshold < polarity < neutral_threshold:
        sentiment = "neutral"
    elif polarity >= neutral_threshold:
        sentiment = "positive"
    else:
        sentiment = "negative"

    print(f"Polarity: {polarity}, Sentiment: {sentiment}")
    return sentiment


def save_to_csv(results, output_file="tiktok_sentiments.csv"):
    """
    Saves the results (TikTok URL and Sentiment) to a CSV file.
    """
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["TikTok URL", "Sentiment"])
        writer.writerows(results)
    print(f"Saved all results to CSV: {output_file}")


def process_tiktok_links(input_csv, output_csv="tiktok_sentiments.csv"):
    """
    Processes all video links in the CSV file, performs sentiment analysis, and saves results.
    """
    data = pd.read_csv(input_csv)
    video_links = data["Video Link"].tolist()

    results = []
    for link in video_links:
        print(f"Processing: {link}")
        try:
            # Step 1: Download the video
            video_path = download_tiktok_video(link)

            # Step 2: Extract audio using FFmpeg
            audio_path = extract_audio_with_ffmpeg(video_path)

            # Step 3: Transcribe the audio
            transcript = transcribe_audio(audio_path, model_type="base")

            # Step 4: Translate the transcript to English
            translated_transcript = translate_to_english(transcript, source_lang="id")

            # Step 5: Analyze the sentiment of the translated text
            sentiment = analyze_sentiment(translated_transcript)

            # Append result
            results.append([link, sentiment])
        except Exception as e:
            print(f"Failed to process {link}: {e}")
            results.append([link, "error"])

    # Step 6: Save results to CSV
    save_to_csv(results, output_csv)


if __name__ == "__main__":
    input_csv = "mentalhealthsurvivor.csv"  # Path to your input CSV
    output_csv = "tiktok_sentiments.csv"  # Path to the output CSV

    # Process all video links
    process_tiktok_links(input_csv, output_csv)
