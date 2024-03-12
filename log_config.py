import logging

def setup_logging(debug_mode=False):
    # Define the logging level based on the debug mode
    level = logging.DEBUG if debug_mode else logging.INFO

    # Basic configuration for logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
