"""
ChatPDF - Telegram Bot
A Telegram bot that processes PDF documents and answers questions about them.
Uses tiered search with OpenAI embeddings and GPT-4o for answers.
"""

import os
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Import our PDF search system
from chatpdf_system import TieredPDFSearchSystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
PDFS_DIR = Path("pdfs")
CHROMA_DB_DIR = Path("chroma_db")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ChatPDFBot:
    """Main bot class handling ChatPDF functionality"""
    
    def __init__(self):
        """Initialize the bot with necessary components"""
        self.pdfs_dir = PDFS_DIR
        self.chroma_db_dir = CHROMA_DB_DIR
        self.ensure_directories()
        
        # Initialize the PDF search system
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables!")
            
        self.pdf_system = TieredPDFSearchSystem(
            openai_api_key=OPENAI_API_KEY,
            embedding_model="text-embedding-3-large",
            answer_model="gpt-4o",
            chroma_persist_directory=str(self.chroma_db_dir)
        )
        
        logger.info("ChatPDF system initialized successfully")
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        self.pdfs_dir.mkdir(exist_ok=True)
        self.chroma_db_dir.mkdir(exist_ok=True)
        logger.info(f"PDF directory: {self.pdfs_dir.absolute()}")
        logger.info(f"ChromaDB directory: {self.chroma_db_dir.absolute()}")
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command"""
        # Get current stats
        stats = self.pdf_system.get_processing_stats()
        
        welcome_message = (
            "👋 Welcome to ChatPDF!\n\n"
            "I can help you find information from PDF documents using advanced AI search.\n\n"
            "📊 Current Status:\n"
            f"• Documents loaded: {stats['document_count']}\n"
            f"• Pages indexed: {stats['page_count']}\n\n"
            "Available commands:\n"
            "• /start - Show this welcome message\n"
            "• /reload - Process PDF documents from the pdfs folder\n"
            "• /status - Show detailed system status\n\n"
            "Simply send me a question and I'll search through the documents for you!\n\n"
            "🔍 I use a two-tier search system:\n"
            "1. First, I find the most relevant documents\n"
            "2. Then, I find the best pages within those documents\n"
            "3. Finally, I use GPT-4o to generate a comprehensive answer"
        )
        await update.message.reply_text(welcome_message)
        
    async def reload_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /reload command to refresh document database"""
        await update.message.reply_text("🔄 Processing PDF documents...")
        
        # Count PDF files in the directory
        pdf_files = list(self.pdfs_dir.glob("*.pdf"))
        
        if not pdf_files:
            await update.message.reply_text(
                f"📁 No PDF files found in {self.pdfs_dir}.\n"
                "Please add some PDF files to the pdfs folder and try again."
            )
            return
        
        try:
            # Process the PDFs
            processing_msg = await update.message.reply_text(
                f"🔧 Processing {len(pdf_files)} PDF file(s)...\n"
                "This may take a few minutes depending on the number and size of documents."
            )
            
            # Actually process the PDFs
            results = self.pdf_system.process_pdf_folder(str(self.pdfs_dir), force_reprocess=False)
            
            # Format results message
            success_message = (
                f"✅ PDF Processing Complete!\n\n"
                f"📊 Results:\n"
                f"• Total files found: {results['total_files']}\n"
                f"• Successfully processed: {results['processed_files']}\n"
                f"• Skipped (already processed): {results['skipped_files']}\n"
                f"• Failed: {results['failed_files']}\n\n"
            )
            
            if results['processed_files'] > 0:
                success_message += "🎉 Ready to answer questions about your documents!"
            elif results['skipped_files'] > 0:
                success_message += "ℹ️ All documents were already processed. Ready to answer questions!"
            
            if results['errors']:
                success_message += f"\n\n⚠️ Errors encountered:\n"
                for error in results['errors'][:3]:  # Show first 3 errors
                    success_message += f"• {error}\n"
                if len(results['errors']) > 3:
                    success_message += f"• ... and {len(results['errors']) - 3} more errors"
            
            await processing_msg.edit_text(success_message)
            
        except Exception as e:
            error_message = f"❌ Error processing PDFs: {str(e)}"
            logger.error(f"Error in reload command: {e}")
            await update.message.reply_text(error_message)
        
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /status command to show system status"""
        try:
            pdf_files = list(self.pdfs_dir.glob("*.pdf"))
            stats = self.pdf_system.get_processing_stats()
            
            status_message = (
                f"📊 ChatPDF Status:\n\n"
                f"📁 PDF Directory: {self.pdfs_dir.absolute()}\n"
                f"📄 PDF Files in folder: {len(pdf_files)}\n"
                f"🗄️ Documents in database: {stats['document_count']}\n"
                f"📖 Pages indexed: {stats['page_count']}\n"
                f"🤖 Bot Status: Online and ready!\n"
                f"🧠 AI Models: text-embedding-3-large + GPT-4o\n\n"
            )
            
            if stats['processed_files']:
                status_message += "📚 Processed Documents:\n"
                for i, file_info in enumerate(stats['processed_files'][:5]):  # Show first 5
                    status_message += f"• {file_info['file_name']} ({file_info['total_pages']} pages)\n"
                
                if len(stats['processed_files']) > 5:
                    status_message += f"• ... and {len(stats['processed_files']) - 5} more documents\n"
            else:
                status_message += "📚 No documents processed yet. Use /reload to process PDFs."
                
            await update.message.reply_text(status_message)
            
        except Exception as e:
            error_message = f"❌ Error getting status: {str(e)}"
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text(error_message)
        
    async def handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle user queries using the PDF search system"""
        user_query = update.message.text
        username = update.effective_user.first_name or "User"
        
        # Check if we have any documents loaded
        stats = self.pdf_system.get_processing_stats()
        
        if stats['document_count'] == 0:
            await update.message.reply_text(
                "📚 No documents have been processed yet!\n\n"
                "Please use /reload to process PDF documents first, "
                "then I'll be able to answer questions about them."
            )
            return
        
        # Send "thinking" message
        thinking_msg = await update.message.reply_text(
            f"🔍 Searching through {stats['document_count']} documents for: \"{user_query}\"\n\n"
            "Please wait while I find the most relevant information..."
        )
        
        try:
            # Perform search and generate answer
            result = self.pdf_system.search_and_answer(
                query=user_query,
                top_documents=3,
                pages_per_document=5
            )
            
            # Format the response
            answer = result['answer']
            search_results = result['search_results']
            
            # Add search metadata
            response_message = f"🤖 **Answer:**\n{answer}\n\n"
            
            if search_results['total_documents_found'] > 0:
                response_message += (
                    f"📊 **Search Results:**\n"
                    f"• Found information in {search_results['total_documents_found']} documents\n"
                    f"• Analyzed {search_results['total_pages_found']} relevant pages\n\n"
                    f"📚 **Sources:**\n"
                )
                
                # Add document sources
                for doc in search_results['documents']:
                    response_message += f"• {doc['file_name']}\n"
            
            # Split long messages if necessary
            if len(response_message) > 4096:  # Telegram message limit
                # Send answer first
                await thinking_msg.edit_text(f"🤖 **Answer:**\n{answer}")
                
                # Send metadata separately
                metadata_msg = (
                    f"📊 **Search Results:**\n"
                    f"• Found information in {search_results['total_documents_found']} documents\n"
                    f"• Analyzed {search_results['total_pages_found']} relevant pages\n\n"
                    f"📚 **Sources:**\n"
                )
                for doc in search_results['documents']:
                    metadata_msg += f"• {doc['file_name']}\n"
                
                await update.message.reply_text(metadata_msg)
            else:
                await thinking_msg.edit_text(response_message)
                
        except Exception as e:
            error_message = (
                f"❌ Sorry {username}, I encountered an error while processing your query:\n\n"
                f"{str(e)}\n\n"
                "Please try rephrasing your question or contact support if the issue persists."
            )
            logger.error(f"Error handling query '{user_query}': {e}")
            await thinking_msg.edit_text(error_message)
        
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Exception while handling an update: {context.error}")

def main():
    """Main function to run the bot"""
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables!")
        logger.error("Please create a .env file with your bot token:")
        logger.error("TELEGRAM_BOT_TOKEN=your_bot_token_here")
        return
        
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables!")
        logger.error("Please add your OpenAI API key to the .env file:")
        logger.error("OPENAI_API_KEY=your_openai_api_key_here")
        return
        
    try:
        # Initialize bot
        bot = ChatPDFBot()
        
        # Create application
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", bot.start_command))
        application.add_handler(CommandHandler("reload", bot.reload_command))
        application.add_handler(CommandHandler("status", bot.status_command))
        
        # Add message handler for queries
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_query))
        
        # Add error handler
        application.add_error_handler(bot.error_handler)
        
        # Start the bot
        logger.info("Starting ChatPDF Bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")

if __name__ == "__main__":
    main()

