import asyncio
import csv
import random
from pytok.tiktok import PyTok
from datetime import datetime

hashtag_name = 'mentalhealthsurvivor'

async def main():
    async with PyTok(headless=False, manual_captcha_solves=True, request_delay=10) as api:
        try:
            hashtag = api.hashtag(name={hashtag_name})
            print(f"Fetching videos for hashtag: {hashtag_name}")

            with open("mentalhealthsurvivor.csv", "w", newline="", encoding="utf-8") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["Video ID", "Video Link", "Description", "Likes", "Comments", "Shares", "Upload Date"])

                async for video in hashtag.videos(count=500):  # Fetch a limited number of videos for testing
                    retries = 0
                    while retries < 5:
                        try:
                            await asyncio.sleep(random.uniform(2, 5))  # Random delay to mimic human behavior
                            video_data = await video.info()

                            # Extract video details
                            video_id = video_data.get("id", "N/A")
                            video_link = f"https://www.tiktok.com/@{video_data.get('author', {}).get('uniqueId', 'unknown')}/video/{video_id}"
                            description = video_data.get("desc", "N/A")
                            stats = video_data.get("stats", {})
                            likes = stats.get("diggCount", 0)
                            comments = stats.get("commentCount", 0)
                            shares = stats.get("shareCount", 0)

                            # Get upload date and convert to human-readable format
                            create_time = video_data.get("createTime", None)
                            upload_date = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S') if create_time else "N/A"

                            csvwriter.writerow([video_id, video_link, description, likes, comments, shares, upload_date])
                            print(f"Video saved: {video_id} - {video_link} (Uploaded on {upload_date})")
                            break  # Break retry loop on success
                        except Exception as e:
                            retries += 1
                            print(f"Retry {retries}/5 due to error: {e}")
                            await asyncio.sleep(5)  # Wait before retrying
                    else:
                        print("Failed to fetch video after 5 retries.")

            print("Data saved to mentalhealthawarness_videos.csv.")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        print("Operation was cancelled.")
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Explicitly create and set a new event loop if necessary
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
