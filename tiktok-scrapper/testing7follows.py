import asyncio
import csv
import random
from pytok.tiktok import PyTok
from datetime import datetime
import re

# Input and Output CSV files
input_csv = 'tiktok.csv'  # The CSV that contains video links and usernames
output_csv = 'output_video_followers.csv'  # The new CSV file to store user followers

async def main():
    # Start PyTok with headless=False to open a browser window where CAPTCHA can be solved manually
    async with PyTok(headless=False, manual_captcha_solves=True, request_delay=10) as api:
        try:
            with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
                csvreader = csv.DictReader(infile)  # Reading the existing CSV file
                csvwriter = csv.writer(outfile)  # Writing to a new CSV
                csvwriter.writerow(["Video Link", "User Followers"])  # Output CSV columns

                # Iterate through each row in the input CSV
                for row in csvreader:
                    video_link = row.get('Video Link', '').strip()  # Assuming 'Video Link' is the column name
                    if not video_link:
                        continue

                    # Extract username from the video link (assuming it's in the format: tiktok.com/@username/video/{video_id})
                    match = re.search(r'tiktok\.com/@([^/]+)', video_link)
                    if not match:
                        print(f"Skipping invalid link: {video_link}")
                        continue
                    
                    username = match.group(1)  # Extract the username from the link

                    retries = 0
                    while retries < 5:
                        try:
                            await asyncio.sleep(random.uniform(2, 5))  # Random delay to mimic human behavior
                            # Fetch user info using the username
                            user_data = await api.user(username=username).info()
                            user_followers = user_data.get('followerCount', 0)  # Accessing follower count

                            # Write the data to the new CSV (video link and followers count)
                            csvwriter.writerow([video_link, user_followers])
                            print(f"User Followers for {username} fetched: {user_followers}")
                            break  # Break retry loop on success
                        except Exception as e:
                            retries += 1
                            print(f"Retry {retries}/5 due to error: {e}")
                            await asyncio.sleep(5)  # Wait before retrying
                    else:
                        print(f"Failed to fetch user data for {video_link} after 5 retries.")

            print(f"Data saved to {output_csv}.")

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
