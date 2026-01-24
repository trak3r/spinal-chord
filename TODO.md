The code now tries Vision API first (if credentials are available), which performs web detection to find similar images and extract descriptions that might include book titles. If Vision fails or isn't configured, it falls back to Tesseract OCR.

To enable Google Vision reverse image search:

Create a Google Cloud project (if you don't have one)
Enable the Vision API in the Google Cloud Console
Create a service account key:
Go to IAM & Admin > Service Accounts
Create a new service account
Generate a JSON key file
Set the environment variable:
Restart the notebook kernel or run the cells again
The Vision API has a generous free tier ($300 credit for new users) and can identify books by finding similar spine images on the web (e.g., on Amazon, Goodreads, or library sites). This approach might work better than OCR for challenging spine images.