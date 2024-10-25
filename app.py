# Import libraries
import os, sys, re
import chromadb
import torch
import time
from io import BytesIO
sys.path.insert(0, '../')
from dotenv import find_dotenv, dotenv_values
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    StorageContext, VectorStoreIndex
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.core.memory import ChatMemoryBuffer
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTForImageClassification
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Define variables
config = dotenv_values(find_dotenv())
embed_model = None
llm = None
index = None
image_model = None
image_processor = None
image_classification_model = None
image_classification_processor = None
image_collection = None
text_collection = None
persist_directory = "./chroma_db"
image_embeddings = None


# Initialization function
def initialize():
    global persist_directory
    os.environ['LLAMA_INDEX_CACHE_DIR'] = os.path.join(os.path.abspath('./'), 'cache')
    persist_directory = os.path.join(os.path.abspath('./'), 'chroma_db')
    print('ChromaDB will store data in:', persist_directory)

def calculate_accuracy(image_embedding, query_embedding):
    similarity = cosine_similarity([image_embedding],[query_embedding])[0][0]
    return similarity

def search_image(query):
    global image_embeddings
    if not query.strip():
        return None, "Oops! You forgot to type something on the query input!", ""

    print(f"\nQuery: {query}")
    query_template = f"a {query}"

    
    start_time = time.time()
    
    inputs = image_processor(text=query_template, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = image_model.get_text_features(**inputs).numpy()

    # Cosine similarity between query and image embeddings
    query_embedding = outputs.tolist()
    
    results = image_collection.query(
        query_embeddings=query_embedding,          
        n_results=1)
    
    result_image_path = results['metadatas'][0][0]['image']
    matched_image_index = int(results['ids'][0][0])
    matched_image_embedding = image_embeddings[matched_image_index]

    accuracy_score = calculate_accuracy(matched_image_embedding, query_embedding[0])
    
    end_time = time.time()
    query_time = end_time - start_time
    
    result_image = Image.open(result_image_path)    
    file_name = result_image_path.split('/')[-1]

    return result_image, accuracy_score, query_time, file_name

def get_one_image_per_breed(image_root_folder):
    selected_images = []
    for breed_folder in os.listdir(image_root_folder):
        breed_path = os.path.join(image_root_folder, breed_folder)
        if os.path.isdir(breed_path):
            image_files = [os.path.join(breed_path, file) for file in os.listdir(breed_path)
                            if os.path.splitext(file.lower())[1] in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']]
        if image_files:
            breed_name = breed_folder.split('-')[-1]
            selected_images.append((image_files[0], breed_name))
        else:
            print(f"Tidak ada gambar yang ditemukan dalam folder: {breed_path}")
    return selected_images

def embed_images():
    global image_embeddings
    image_root_folder = "./data/Images"

    images_and_breeds = get_one_image_per_breed(image_root_folder)

    print(f"Jumlah gambar yang dipilih untuk di-embed: {len(images_and_breeds)}")

    image_paths, breed_names = zip(*images_and_breeds)
    texts = [f"A dog of the breed {breed_name.replace('_', ' ')}." for breed_name in breed_names]

    images = [Image.open(image_path) for image_path in image_paths]
    inputs = image_processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Measure image ingestion time
    start_ingestion_time = time.time()

    with torch.no_grad():
        outputs = image_model(**inputs)

    # Extract image embeddings
    image_embeddings = outputs.image_embeds.cpu().numpy()

    # Measure total ingestion time
    end_ingestion_time = time.time()
    ingestion_time = end_ingestion_time - start_ingestion_time

    image_collection.add(
        embeddings=image_embeddings,
        metadatas=[{"image": image_path} for image_path in image_paths],
        ids=[str(i) for i in range(len(image_paths))]
    )

    # Log the ingestion performance
    print(f"Image Data ingestion time: {ingestion_time:.4f} seconds")

def get_breeds():
    breeds = []
    normalized_breeds = {}
    for breed_folder in os.listdir("./data/Images"):
        breed_name = breed_folder.split('-')[-1]
        breeds.append(breed_name)
        # Normalize the breed name
        normalized_name = breed_name.replace('_', ' ').lower()
        normalized_breeds[normalized_name] = breed_name
    return breeds, normalized_breeds


# LLM Function
def setup():
    api_key = config.get('GOOGLE_API_KEY')
    
    os.environ["GOOGLE_API_KEY"] = api_key

    # Initialize the embedding model
    global embed_model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Initialize the LLM
    global llm
    llm = Gemini(
        model_name="models/gemini-1.5-flash",
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
    )

    global image_model
    global image_processor
    image_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    global image_classification_model
    global image_classification_processor
    image_classification_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    image_classification_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # Test the LLM
    resp = llm.complete("Paul Graham is")
    print(resp)

def get_chat_engine(context):
    if 'chat_engine' not in context.user_data:
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        context.user_data['chat_engine'] = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt="""
            You are an efficient language model designed to respond promptly to user inquiries.
            Responses should be concise and to the point, avoiding unnecessary elaboration unless requested by the user.
            Remember to give other dog breeds if users didn't like it                      
            """,
            llm=llm
        )
    return context.user_data['chat_engine']

# Connect LLM to ChromaDB
def connect_llm():
    global index

    vector_store = ChromaVectorStore(
        chroma_collection=text_collection
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create the index without using ServiceContext
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
        llm=llm
    )

# Start command handler
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_text = """
    🐾 Welcome to Pawspective! 🐾

    To begin, please describe the type of dog you're interested in. For example, you can tell me about the size, temperament, or any specific traits you prefer in a dog.

    Let's find your perfect furry companion! 🐶🐾
    """
    await update.message.reply_text(start_text)

# Help command handler
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
    🐾 Welcome to Pawspective! 🐾

    To use this bot, simply follow these steps:
    1. Type /start to begin.
    2. Ask any questions about dogs that you want to know.
    3. Explore the suggested dog breeds to find your ideal furry companion!

    Feel free to ask any questions about dog breeds or how to use the bot. 🐶🐾
    """
    await update.message.reply_text(help_text)

# Message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_message = update.message.text

        # Get the chat_engine using the helper function
        user_chat_engine = get_chat_engine(context)

        # Check if the message contains any breed names
        breeds, normalized_breeds = get_breeds()
        
        # Normalize the user's message
        normalized_message = user_message.lower().replace('_', ' ').replace('-', ' ')
        normalized_message = re.sub(r'\s+', ' ', normalized_message)

        breeds_in_message = []
        for normalized_breed, original_breed in normalized_breeds.items():
            if normalized_breed in normalized_message:
                breeds_in_message.append(original_breed)
        
        if breeds_in_message:
            # First, get the image search results
            breed_query = breeds_in_message[0]  # Use the first found breed
            result_image, accuracy_score, query_time, file_name = search_image(breed_query)
            
            # Save image to BytesIO to send via telegram
            img_byte_arr = BytesIO()
            result_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Get text information about the breed
            breeds_str = ', '.join(breeds_in_message)
            prompt = f"Please provide detailed information about the following dog breeds: {breeds_str}. Ensure the response is clear and informative."
            response = user_chat_engine.chat(prompt)
            text_response = response.response  # Extract text
            
            # Prepare the caption
            caption = (f"Breed: {breed_query}\n"
                       f"Accuracy Score: {accuracy_score:.4f}\n"
                       f"Query Time: {query_time:.4f} seconds\n"
                       f"{text_response}")
            
            # Send image with caption
            await update.message.reply_photo(
                photo=img_byte_arr,
                caption=caption[:1024]  # Telegram caption limit is 1024 characters
            )
        else:
            # Handle regular chat if no breed is mentioned
            response = user_chat_engine.chat(user_message)
            clean_response_text = re.sub(r'[*_~`]', '', response.response)
            await update.message.reply_text(clean_response_text)
    except Exception as e:
        print(f"Error in handle_message: {str(e)}")
        await update.message.reply_text("Sorry, there was an error processing your message. Please try again.")


# Main function
def main():
    print('Starting the Bot...')
    app = Application.builder().token(config.get('BOT_API')).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(MessageHandler(filters.PHOTO | filters.TEXT & (~filters.COMMAND), handle_message))

    print('Polling...')
    app.run_polling(poll_interval=3)

if __name__ == '__main__':
    initialize()
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    text_collection = chroma_client.get_or_create_collection(name="dog_breeds_text")

    try:
        image_collection = chroma_client.get_collection(name="dog_breeds_image")
    except:
        pass
    else:
        image_collection = chroma_client.delete_collection(name="dog_breeds_image")

    image_collection = chroma_client.create_collection(name="dog_breeds_image")

    setup()
    embed_images()
    connect_llm()
    main()
