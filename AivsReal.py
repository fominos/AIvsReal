import logging
import dill
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    CallbackQueryHandler,
    Application,
    ApplicationBuilder,
    ContextTypes,
)
IMG_SIZE = (224, 224)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PHOTO = range(1)


def load_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path)[..., ::-1]
    img = cv2.resize(img, target_size)
    return preprocess_input(img)


def predict_generator(files):
    while True:
        for path in files:
            img = np.array([load_image(path)])
            yield (img,)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        'Welcome! I am your virtual image detective. Ready to find out the truth about your photo?\n\n'
        'Upload an image, and I will tell you whether it is real or created by Artificial Intelligence or press /cancel to end the conversation')
    return PHOTO


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    # Await the coroutine to get the file object
    photo_file = await update.message.photo[-1].get_file()
    # Await the download
    await photo_file.download_to_drive(f'{user.id}_photo.jpg')
    logger.info("Фотография %s: %s", user.id,
                f'{user.id}_photo.jpg')
    await update.message.reply_text(
        'Great! Give me a moment to analyze your image...')
    with open("model.pkl", "rb") as dill_file:
        model = dill.load(dill_file)
        test_pred = model.predict(predict_generator(
            [f'{user.id}_photo.jpg']), steps=len([f'{user.id}_photo.jpg']))
        print(test_pred[0][0])
        if test_pred[0][0] > 0.5:
            result = "This image is real! Want to know more? Upload another image, and I will help you uncover its secrets again!"
        else:
            result = "This image was created by Artificial Intelligence! Want to know more? Upload another image, and I will help you uncover its secrets again!"

    await update.message.reply_text(
        result
    )

    return PHOTO


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:

    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)

    await update.message.reply_text(
        'See you later!\n'
        'Press command /start to continue'
    )

    return ConversationHandler.END


def main() -> None:
    application = (
        Application.builder()
        .token("7135655915:AAEzIcn9FfpBbzsbXw69c9hcgqCQ5BXrBYE")
        .build()
    )

    conv_handler = ConversationHandler(  # здесь строится логика разговора
        entry_points=[CommandHandler('start', start)],
        states={
            PHOTO: [MessageHandler(filters.PHOTO, photo)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
