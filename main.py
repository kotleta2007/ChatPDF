"""
ChatPDF - Telegram Bot
A Telegram bot that processes PDF documents and answers questions about them.
Currently implements basic echo functionality.
"""

import os
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

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
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

class ChatPDFBot:
    """Main bot class handling ChatPDF functionality"""
    
    def __init__(self):
        """Initialize the bot with necessary components"""
        self.pdfs_dir = PDFS_DIR
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        self.pdfs_dir.mkdir(exist_ok=True)
        logger.info(f"PDF directory: {self.pdfs_dir.absolute()}")
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command"""
        welcome_message = (
            "👋 Welcome to ChatPDF!\n\n"
            "I can help you find information from PDF documents.\n\n"
            "Available commands:\n"
            "• /start - Show this welcome message\n"
            "• /reload - Reload PDF documents from the pdfs folder\n"
            "• /status - Show current system status\n\n"
            "Simply send me a question and I'll search through the documents for you!"
        )
        await update.message.reply_text(welcome_message)
        
    async def reload_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /reload command to refresh document database"""
        await update.message.reply_text("🔄 Reloading documents...")
        
        # Count PDF files in the directory
        pdf_files = list(self.pdfs_dir.glob("*.pdf"))
        
        if not pdf_files:
            await update.message.reply_text(
                f"📁 No PDF files found in {self.pdfs_dir}.\n"
                "Please add some PDF files to the pdfs folder and try again."
            )
            return
            
        # TODO: Implement actual document processing
        await update.message.reply_text(
            f"✅ Found {len(pdf_files)} PDF file(s):\n" + 
            "\n".join(f"• {pdf.name}" for pdf in pdf_files[:10]) +
            (f"\n... and {len(pdf_files) - 10} more" if len(pdf_files) > 10 else "") +
            "\n\n🔧 Document processing will be implemented in the next step!"
        )
        
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /status command to show system status"""
        pdf_files = list(self.pdfs_dir.glob("*.pdf"))
        
        status_message = (
            f"📊 ChatPDF Status:\n\n"
            f"📁 PDF Directory: {self.pdfs_dir.absolute()}\n"
            f"📄 PDF Files: {len(pdf_files)}\n"
            f"🗄️ Vector Database: Not initialized yet\n"
            f"🤖 Bot Status: Online and ready to echo!"
        )
        await update.message.reply_text(status_message)
        
    async def handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle user queries - currently just echoes back"""
        user_query = update.message.text
        username = update.effective_user.first_name or "User"
        
        # Simple echo response for now
        response = (
            f"👋 Hi {username}!\n\n"
            f"You asked: \"{user_query}\"\n\n"
            f"🔧 I'm currently in development mode and just echoing your message back.\n"
            f"Soon I'll be able to search through PDF documents to answer your questions!"
        )
        
        await update.message.reply_text(response)
        
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

if __name__ == "__main__":
    main()

