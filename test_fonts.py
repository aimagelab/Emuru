import os
import gc
import psutil
import logging
from PIL import ImageFont

# Set up logging
logging.basicConfig(filename='font_loading.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class FontRenderer:
    def __init__(self, font_path, font_size):
        try:
            if not os.path.isabs(font_path):
                # Convert relative path to absolute path
                font_path = os.path.abspath(font_path)
            self.font_path = font_path
            self.font_size = font_size

            # Debugging: print the font path
            logging.info(f"Loading font from: {self.font_path}")

            # Check if the file exists
            if not os.path.exists(self.font_path):
                raise FileNotFoundError(f"Font file does not exist: {self.font_path}")
            
            # Check file permissions
            if not os.access(self.font_path, os.R_OK):
                raise PermissionError(f"Font file is not readable: {self.font_path}")

            # Attempt to load the font
            self.font = ImageFont.truetype(self.font_path, self.font_size)
            logging.info(f"Font loaded successfully: {self.font_path}")

        except Exception as e:
            logging.error(f"An error occurred while loading the font: {e}")
            raise

        finally:
            # Explicitly release resources
            del self.font
            gc.collect()  # Collect garbage

def log_resource_usage(process, step_description=""):
    mem_info = process.memory_info()
    open_files = len(process.open_files())
    logging.info(
        f"{step_description} - Memory usage: RSS={mem_info.rss}, VMS={mem_info.vms}, "
        f"Open files: {open_files}"
    )

def load_fonts(font_paths, font_size, batch_size=1000):
    process = psutil.Process(os.getpid())
    for i in range(0, len(font_paths), batch_size):
        batch = font_paths[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1} / {len(font_paths)//batch_size + 1}")
        for font_path in batch:
            try:
                FontRenderer(font_path, font_size)
            except Exception as e:
                logging.error(f"Failed to load font {font_path}: {e}")
        
        # Log resource usage after each batch
        log_resource_usage(process, step_description=f"After batch {i//batch_size + 1}")

# Example usage
# font_directory = '/work/FoMo_AIISDH/fquattrini/Emuru/files/font_square/clean_fonts'
font_paths = ['/home/fquattrini/emuru/files/font_square/backgrounds/1000_F_110099353_9TJL7cxNQq2tJbK0KjPkBcCe5AvT0A5z.jpg'] * 115961  # Using the same font path
font_size = 32

# Loading fonts in batches with resource cleanup and monitoring
load_fonts(font_paths, font_size)
