import os
import requests
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

# Constants and global variables
ADMIN_ID = 6425757396  # Replace with your actual admin Telegram ID
TOKEN = "7442020780:AAFKQPHfFJpeNRlQEB1gqwX1Al18PY6VPmw"  # Replace with your actual bot token
MODEL_PATH = 'luckyAI.keras'  # Path to the pre-trained model

# User license management
user_licenses = {}

def is_license_valid(user_id):
    """Check if a user's license is valid."""
    if user_id in user_licenses:
        expiry = user_licenses[user_id]
        if expiry is None:
            return False
        if expiry == -1 or expiry > datetime.now():
            return True
    return False

def issue_license(user_id, duration_hours):
    """Issue a license to a user for a specified duration."""
    if duration_hours == -1:
        user_licenses[user_id] = -1  # Permanent license
    else:
        user_licenses[user_id] = datetime.now() + timedelta(hours=duration_hours)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    user_id = update.message.from_user.id
    if is_license_valid(user_id):
        keyboard = [[InlineKeyboardButton("START", callback_data='start_signal')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("<b>ADMIN PANEL</b>", reply_markup=reply_markup, parse_mode='HTML')
    else:
        user_licenses[user_id] = None  # Initialize license status for the user
        await update.message.reply_text(
            "<b>Hi, this is the best bot for the LUCKY JET game</b>\n"
            f"- <b>Your ID:</b> {user_id}\n"
            "- <b>To start using the bot, you must buy access from @shyne1x</b>",
            parse_mode='HTML'
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages."""
    user_id = update.message.from_user.id
    if is_license_valid(user_id):
        keyboard = [[InlineKeyboardButton("START", callback_data='start_signal')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("<b>ADMIN PANEL</b>", reply_markup=reply_markup, parse_mode='HTML')
    else:
        await update.message.reply_text("Invalid key. Please contact @shyne1x to buy a key.")

async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin command to issue licenses."""
    user_id = update.message.from_user.id
    if user_id != ADMIN_ID:
        await update.message.reply_text("You are not authorized to issue licenses.")
        return

    try:
        args = context.args
        target_user_id = int(args[0])
        duration = args[1]

        if duration.endswith('h'):
            duration_hours = int(duration[:-1])
        elif duration == "permanent":
            duration_hours = -1
        else:
            raise ValueError("Invalid duration format")

        issue_license(target_user_id, duration_hours)
        if duration_hours == -1:
            await update.message.reply_text(f"Issued a permanent license to user {target_user_id}.")
        else:
            await update.message.reply_text(f"Issued a {duration_hours}-hour license to user {target_user_id}.")

        await context.bot.send_message(
            chat_id=target_user_id,
            text=f"<b>You've been issued a license: {duration}</b>",
            parse_mode='HTML'
        )
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /admin <user_id> <duration(h/permanent)>")

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks."""
    query = update.callback_query
    user_id = query.from_user.id

    if query.data == 'start_signal':
        if is_license_valid(user_id):
            await query.edit_message_text("<b>Connecting...</b>", parse_mode='HTML')
            forecast, confidence = await get_forecast()
            if forecast == "SKIP":
                keyboard = [[InlineKeyboardButton("START", callback_data='start_signal')]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text("<b>SKIP</b>\n<b>@shyne1x panel</b>", reply_markup=reply_markup, parse_mode='HTML')
            else:
                keyboard = [[InlineKeyboardButton("START", callback_data='start_signal')]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    f"<b>X {forecast:.2f}</b>\n<b>@shyne1x panel</b>",
                    reply_markup=reply_markup,
                    parse_mode='HTML'
                )
        else:
            await query.edit_message_text("<b>Your license has expired</b>\n- <b>buy a key @shyne1x</b>", parse_mode='HTML')

async def get_forecast():
    """Fetch forecast data and predict the next value."""
    url = 'https://lucky-jet-history.gamedev-atech.cc/public/history/api/history'
    headers = {'session': 'demo'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        coefficients = [item['coefficient'] for item in data[:20]]  # Get the last 20 coefficients
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch data from the server: {e}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    coefficients_scaled = scaler.fit_transform(np.array(coefficients).reshape(-1, 1))

    look_back = 10
    X_train = []
    y_train = []
    for i in range(look_back, len(coefficients_scaled)):
        X_train.append(coefficients_scaled[i-look_back:i, 0])
        y_train.append(coefficients_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = build_model((look_back, 1))
        early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0, callbacks=[early_stopping])
        model.save(MODEL_PATH)

    X_test = np.array(coefficients_scaled[-look_back:])
    X_test = np.reshape(X_test, (1, X_test.shape[0], 1))
    prediction_scaled = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction_scaled)
    confidence = np.std(prediction_scaled)

    if prediction[0][0] <= 1.00:
        return "SKIP", confidence
    else:
        return prediction[0][0], confidence

def build_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

def main() -> None:
    """Main function to start the bot."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("admin", admin))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button))

    application.run_polling()

if __name__ == '__main__':
    main()
