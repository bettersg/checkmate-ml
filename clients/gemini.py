from google import genai
import os
from dotenv import load_dotenv
import functools
import time
from logger import StructuredLogger

logger = StructuredLogger("gemini_client")

load_dotenv()


def retry_once_per_model(wait_time=2, fallback_models=None):
    if fallback_models is None:
        fallback_models = ["gemini-2.0-flash-exp", "gemini-1.5-pro"]

    def decorator(func):  # Synchronous decorator
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # Synchronous wrapper
            model_index = 0

            while model_index < len(fallback_models):
                kwargs["model"] = fallback_models[model_index]
                logger.info(f"Generating using model: {fallback_models[model_index]}")
                try:
                    return func(*args, **kwargs)  # First attempt (no await needed)
                except Exception as e:
                    if "429" in str(e):
                        logger.warning(
                            f"Resource exhausted (429) for model {fallback_models[model_index]}. Retrying once..."
                        )
                        time.sleep(wait_time)  # Blocking sleep for synchronous function
                        try:
                            return func(*args, **kwargs)  # Retry once
                        except Exception as retry_error:
                            logger.warning(
                                f"Retry failed for model {fallback_models[model_index]}: {retry_error}"
                            )
                    else:
                        raise  # Raise non-429 errors immediately

                model_index += 1  # Move to the next model if retry fails

            raise Exception("All models and retries exhausted")

        return wrapper

    return decorator


# Initialize the gemini client
gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Apply the decorator to `models.generate_content`
gemini_client.models.generate_content = retry_once_per_model(wait_time=2)(
    gemini_client.models.generate_content
)

__all__ = ["gemini_client"]
