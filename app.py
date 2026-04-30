import sys
import os
from dotenv import load_dotenv

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from src.gui import create_ui

def main():
    """
    Main entry point for the Performance-Aware Model Compression for Circuit Analysis Using Prerequisite Graphs Framework.
    """
    app = create_ui()
    
    # Run the Gradio app
    app.launch(
        share=True
    )

if __name__ == "__main__":
    main()
