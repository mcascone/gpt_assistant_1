# from https://cookbook.openai.com/examples/gpt4o/introduction_to_gpt4o

from openai import OpenAI
from helpers import print_spacer

import os

## Set the API key and model name
MODEL="gpt-4o"
client = OpenAI() # defaults to getting the API key using os.environ.get("OPENAI_API_KEY")

# completion = client.chat.completions.create(
#   model=MODEL,
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant. Help me with my math homework!"}, # <-- This is the system message that provides context to the model
#     {"role": "user", "content": "Hello! Could you solve 2+2?"}  # <-- This is the user message for which the model will generate a response
#   ]
# )

# print(completion) # <-- This is the response from the model
# print_spacer()
# print(completion.choices[0].message.content) # <-- This is the response from the model

#### RAW IMAGE

# from IPython.display import Image, display, Audio, Markdown
# import base64

# IMAGE_PATH = "../data/triangle.png"

# # Preview image for context
# display(Image(IMAGE_PATH))

# # Open the image file and encode it as a base64 string
# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode("utf-8")

# base64_image = encode_image(IMAGE_PATH)

# response = client.chat.completions.create(
#   model=MODEL,
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
#     {"role": "user", "content": [
#       {"type": "text", "text": "What's the area of the triangle?"},
#       {"type": "image_url", "image_url": {
#         "url": f"data:image/png;base64,{base64_image}"}
#       }
#     ]}
#   ],
#   temperature=0.0,
# )

# print(response.choices[0].message.content)

#### URL IMAGE

# response = client.chat.completions.create(
#   model=MODEL,
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
#     {"role": "user", "content": [
#       {"type": "text", "text": "What's the area of the triangle?"},
#       {"type": "image_url", "image_url": {
#         "url": "https://upload.wikimedia.org/wikipedia/commons/e/e2/The_Algebra_of_Mohammed_Ben_Musa_-_page_82b.png"}
#       }
#     ]}
#   ],
#   temperature=0.0,
# )

# print(response.choices[0].message.content)

#### VIDEO
import cv2
from moviepy.editor import VideoFileClip
import time
import base64

# We'll be using the OpenAI DevDay Keynote Recap video. You can review the video here: https://www.youtube.com/watch?v=h02ti0Bl6zk
VIDEO_PATH = "../data/keynote_recap.mp4" # can't download the video

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path

# Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)
