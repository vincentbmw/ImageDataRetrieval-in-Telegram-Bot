# Pawspective Telegram Bot

🐾 **Pawspective** is a Telegram bot designed to help you find your perfect canine companion. Whether you're curious about different dog breeds, seeking detailed information, or want to identify a breed from a photo, Pawspective is here to assist!

## Demo Video
https://github.com/user-attachments/assets/94b499f1-2296-4b08-be3e-15873e7786be

## Features

- **Start Interaction**: Begin your journey with a friendly welcome message using the `/start` command.
- **Help Guide**: Get instructions on how to use the bot with the `/help` command.
- **Text-Based Queries**: Describe the type of dog you're interested in, and the bot will provide breed recommendations and information.
- **Image Recognition**: Send a photo of a dog, and the bot will identify the breed and provide detailed information.
- **Interactive Chat**: Engage in a conversation with the bot to explore various dog breeds and traits.

## Technologies Used

- **Python 3**
- **Telegram Bot API** via `python-telegram-bot`
- **LlamaIndex**: For language model interaction.
- **ChromaDB**: As a vector store for embeddings.
- **HuggingFace Transformers**: Utilizing models like CLIP and ViT.
- **PyTorch**: For deep learning computations.
- **Scikit-learn**: For calculating cosine similarity.
- **Pillow (PIL)**: For image processing.
- **Environment Variables**: Managed with `python-dotenv`.

## Setup and Installation

### Prerequisites

- Python 3.7 or higher.
- Telegram account.
- Telegram Bot Token: Obtain one by creating a bot via [BotFather](https://core.telegram.org/bots#6-botfather).

### Installation Steps and Usage Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/vincentbmw/ImageDataRetrieval-in-Telegram-Bot.git
   cd ImageDataRetrieval-in-Telegram-Bot
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   - Create a `.env` file in the project root directory.
   - Add your Telegram Bot Token and Google API Key:

     ```env
     BOT_API=your-telegram-bot-token
     GOOGLE_API_KEY=your-google-api-key
     ```

5. **Setup Embedding Text**

   - Go to `embedding.ipynb` file.
   - Choose your kernel `venv` that you already do in Step 2.
   - Run all the Cell

6. **Starting the Bot**

   - Ensure that you're in the root folder which is `ImageDataRetrieval-in-Telegram-Bot`.
   - Run this Command in your root folder

     ```bash
     python app.py
     ```
     
7. **Interact via Telegram**

   - Open your Telegram app.
   - Search your username bot which mine is `@pawspective_bot`.
   - Start the conversation by typing `/start` or you can just do Text Message (See examples).

### Commands and Interaction

- **`/start`**: Receive a welcome message and instructions to begin.
- **`/help`**: Get assistance on how to use the bot.
- **Text Messages**: Describe the dog traits you're interested in (e.g., "I want a small, friendly dog").
- **Image Messages**: Send a photo of a dog to identify its breed.

### Examples

- **Text Query**

  ```
  User: I'm looking for a large, energetic dog that's good with kids.
  Bot: Based on your preferences, the Labrador Retriever might be a great match! They are large, energetic, and excellent with children.
  ```

- **Image Query**

  - Send a photo of a dog.
  - The bot responds with the detected breed, accuracy score, and detailed information.

## Project Structure

```
ImageDataRetrieval-in-Telegram-Bot/
├── data/
│   └── Images/
│       └── (Dog breed images organized by breed)
│   └── dogs/
│       └── (Dog breed in pdf format)
├── cache/
│   └── (Pre-trained embedding_text model)
├── app.py
├── embedding.ipynb
├── requirements.txt
├── .env
└── README.md
```

## Notes

**Image Dataset**: To optimize processing time, this project uses only one image per breed. This approach speeds up the embedding and querying processes but may affect the accuracy of breed identification. For better accuracy, you can include more images per breed in the ./data/Images directory. You can download the Image dataset from [Stanford Dogs Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)


## 📝 License

This project is open source and available under the [MIT License](LICENSE).

