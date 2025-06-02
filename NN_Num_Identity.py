#!/usr/bin/env python3
"""
Train a CNN on MNIST (or load a saved model) and open a Tkinter window to draw digits and predict.
Note1: Softmax for output layer is not recommended. Future iterations would have the output use a linear activation
Note2: With linear output change compiler to loss=SparseCategoricalCrossentropy(from_logits=True) and use a softmax conversion
Conversion: def my_softmax(z):
                ez = np.exp(z)
                sm = ez/np.sum(ez)
                return(sm)

Dependencies:
    pip install tensorflow numpy pillow
"""

import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from tensorflow import keras
from tensorflow.keras import layers

# Path to save/load the trained model
MODEL_PATH = "mnist_cnn.h5"

# Determine the correct resampling filter for Pillow
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

# --- 1) Load & preprocess MNIST data ---
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = np.expand_dims(x_train, -1)  # (60000,28,28,1)
    x_test  = np.expand_dims(x_test,  -1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test  = keras.utils.to_categorical(y_test,  10)
    return (x_train, y_train), (x_test, y_test)

# --- 2) Build the CNN model ---
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --- 3) GUI app for drawing ---
class DrawPredictApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Draw a Digit (0–9) and Predict")
        self.resizable(False, False)
        self.model = model

        self.canvas_size = 200
        self.bg_color = "white"
        self.pen_width = 12
        self.pen_color = "black"

        # Canvas for drawing
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size,
                                bg=self.bg_color)
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # PIL image for capturing
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw  = ImageDraw.Draw(self.image)

        # Buttons
        self.predict_btn = tk.Button(self, text="Predict", command=self.predict_digit)
        self.predict_btn.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,10))
        self.clear_btn   = tk.Button(self, text="Clear",   command=self.clear_canvas)
        self.clear_btn.grid(row=1, column=1, sticky="ew", padx=10, pady=(0,10))

        # Label for result
        self.result_label = tk.Label(self, text="Draw a digit and click Predict", font=("Helvetica", 14))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=(0,10))

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_paint)

        self.last_x, self.last_y = None, None

    def on_button_press(self, event):
        self.last_x, self.last_y = event.x, event.y

    def on_paint(self, event):
        x, y = event.x, event.y
        # Draw on the Tkinter canvas
        self.canvas.create_line(self.last_x, self.last_y, x, y,
                                width=self.pen_width, fill=self.pen_color,
                                capstyle=tk.ROUND, smooth=True)
        # Draw on the PIL image
        self.draw.line([self.last_x, self.last_y, x, y],
                       fill=0, width=self.pen_width)
        self.last_x, self.last_y = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size],
                            fill=255)
        self.result_label.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        # Prepare the image: resize to 28x28, invert colors, normalize
        img = self.image.resize((28,28), resample=RESAMPLE)
        img = ImageOps.invert(img)
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))  # (1,28,28,1)

        # Predict
        preds = self.model.predict(arr)
        digit = np.argmax(preds)
        confidence = preds[0][digit]

        self.result_label.config(
            text=f"Prediction: {digit} (Confidence: {confidence*100:.1f}%)"
        )

# --- 4) Main function ---
def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Load or train the model
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        model = build_model()
        model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=5,
            validation_split=0.1,
            verbose=2
        )
        model.save(MODEL_PATH)
        print(f"Trained new model and saved to {MODEL_PATH}")

    # Evaluate on test set
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc*100:.2f}% (loss: {loss:.4f})")

    # Launch drawing GUI
    app = DrawPredictApp(model)
    app.mainloop()

if __name__ == "__main__":
    main()
