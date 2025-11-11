#!/usr/bin/env python3
"""
Cyber Sentinel Model Application
Main entry point for the application
"""

import logging
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app import create_app
from config.settings import Config

def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('cyber_sentinel.log')
        ]
    )

def main():
    """Main application entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        app = create_app()
        
        logger.info("Starting Cyber Sentinel Model Application")
        logger.info(f"Debug mode: {Config.DEBUG}")
        logger.info(f"Host: {Config.API_HOST}")
        logger.info(f"Port: {Config.API_PORT}")
        
        app.run(
            host=Config.API_HOST,
            port=Config.API_PORT,
            debug=Config.DEBUG
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()